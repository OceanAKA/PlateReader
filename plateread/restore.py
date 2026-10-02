"""Image restoration: the part that undoes blur, as far as blur is undoable.

Deconvolution recovers detail only while the blur is mild enough that the
signal survives above the noise floor. Past that, every method here starts
inventing structure - which is why nothing is trusted on its own; each output
is one hypothesis fed to the OCR vote.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.signal import fftconvolve

from .util import clip8, norm8


# --------------------------------------------------------------------------
# point spread functions
# --------------------------------------------------------------------------

def motion_psf(length: float, angle_deg: float, size: int | None = None) -> np.ndarray:
    """Anti-aliased line PSF for linear camera or subject motion."""
    length = max(1.0, float(length))
    if size is None:
        size = int(length) + 4
    size = max(3, size | 1)
    psf = np.zeros((size, size), np.float64)
    c = size // 2
    th = np.deg2rad(angle_deg)
    dx, dy = np.cos(th), -np.sin(th)
    n = max(8, int(length * 6))
    for t in np.linspace(-length / 2.0, length / 2.0, n):
        x, y = c + dx * t, c + dy * t
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        fx, fy = x - x0, y - y0
        for yy, wy in ((y0, 1 - fy), (y0 + 1, fy)):
            for xx, wx in ((x0, 1 - fx), (x0 + 1, fx)):
                if 0 <= yy < size and 0 <= xx < size:
                    psf[yy, xx] += wy * wx
    s = psf.sum()
    return psf / s if s > 0 else psf


def gaussian_psf(sigma: float, size: int | None = None) -> np.ndarray:
    """Isotropic defocus / atmospheric blur approximation."""
    sigma = max(0.3, float(sigma))
    if size is None:
        size = int(np.ceil(sigma * 6))
    size = max(3, size | 1)
    ax = np.arange(size, dtype=np.float64) - size // 2
    g = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    psf = np.outer(g, g)
    return psf / psf.sum()


def disc_psf(radius: float, size: int | None = None) -> np.ndarray:
    """Circular aperture - true out-of-focus blur."""
    radius = max(0.5, float(radius))
    if size is None:
        size = int(np.ceil(radius * 2)) + 3
    size = max(3, size | 1)
    c = size // 2
    yy, xx = np.ogrid[:size, :size]
    d = np.hypot(yy - c, xx - c)
    psf = np.clip(radius + 0.5 - d, 0, 1).astype(np.float64)
    s = psf.sum()
    return psf / s if s > 0 else psf


# --------------------------------------------------------------------------
# blur estimation
# --------------------------------------------------------------------------

@dataclass
class BlurEstimate:
    length: float          # motion extent in pixels
    angle: float           # degrees, 0 = horizontal
    strength: float        # how much to believe the above (peak / noise)
    lap_var: float         # variance of Laplacian, absolute sharpness proxy

    @property
    def trustworthy(self) -> bool:
        return self.strength >= 3.0 and 2.5 <= self.length <= 60.0


def estimate_motion_blur(gray: np.ndarray) -> BlurEstimate:
    """Read motion length and direction off the cepstrum.

    A linear blur multiplies the spectrum by a sinc, whose periodic nulls show
    up in the cepstrum as a pair of negative spikes at +-L along the motion
    direction. Finding the strongest spike gives both parameters at once.
    """
    g = gray.astype(np.float64)
    h, w = g.shape[:2]
    lap = float(cv2.Laplacian(g, cv2.CV_64F).var())
    if min(h, w) < 24:
        return BlurEstimate(0.0, 0.0, 0.0, lap)

    g = g - g.mean()
    win = np.outer(np.hanning(h), np.hanning(w))
    spec = np.fft.fft2(g * win)
    logmag = np.log(np.abs(spec) + 1e-8)
    cep = np.fft.fftshift(np.real(np.fft.ifft2(logmag)))

    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - cy, xx - cx)
    rmax = min(h, w) * 0.45
    ring = (r > 3.0) & (r < rmax)
    if not ring.any():
        return BlurEstimate(0.0, 0.0, 0.0, lap)

    neg = -cep                      # the spikes are negative in the cepstrum
    vals = neg[ring]
    med = float(np.median(vals))
    sd = float(vals.std()) or 1e-9
    scored = np.where(ring, neg, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(scored)), scored.shape)
    peak = float(neg[iy, ix])

    dy, dx = iy - cy, ix - cx
    length = float(np.hypot(dx, dy))
    angle = float(np.degrees(np.arctan2(-dy, dx))) % 180.0
    return BlurEstimate(length, angle, (peak - med) / sd, lap)


# --------------------------------------------------------------------------
# deconvolution
# --------------------------------------------------------------------------

def edge_taper(gray: np.ndarray, k: int = 13) -> np.ndarray:
    """Blend the borders into a blurred copy so the FFT sees no hard seam.

    Without this, deconvolution rings along every image edge and the ripples
    get mistaken for character strokes.
    """
    img = gray.astype(np.float64)
    h, w = img.shape[:2]
    blurred = cv2.GaussianBlur(img, (0, 0), max(1.0, k / 2.0))
    k = max(1, int(k))
    rx = np.minimum(np.arange(w), w - 1 - np.arange(w)) / float(k)
    ry = np.minimum(np.arange(h), h - 1 - np.arange(h)) / float(k)
    a = np.clip(np.minimum.outer(ry, rx), 0.0, 1.0)
    a = 0.5 - 0.5 * np.cos(np.pi * a)           # smoothstep ramp
    return a * img + (1.0 - a) * blurred


def _psf_otf(psf: np.ndarray, shape) -> np.ndarray:
    pad = np.zeros(shape, np.float64)
    ph, pw = psf.shape
    pad[:ph, :pw] = psf
    pad = np.roll(pad, -(ph // 2), axis=0)
    pad = np.roll(pad, -(pw // 2), axis=1)
    return np.fft.fft2(pad)


def wiener_deconv(gray: np.ndarray, psf: np.ndarray, K: float = 0.01) -> np.ndarray:
    """Wiener inverse filter. K is the noise-to-signal guess: bigger = safer."""
    img = edge_taper(gray) / 255.0
    otf = _psf_otf(psf, img.shape)
    spec = np.fft.fft2(img)
    est = np.conj(otf) / (np.abs(otf) ** 2 + max(1e-6, K)) * spec
    out = np.real(np.fft.ifft2(est))
    return clip8(out * 255.0)


def richardson_lucy(gray: np.ndarray, psf: np.ndarray, iters: int = 20) -> np.ndarray:
    """Iterative RL deconvolution - gentler on noise than Wiener, slower."""
    img = np.clip(edge_taper(gray) / 255.0, 1e-4, 1.0)
    est = np.full(img.shape, 0.5, np.float64)
    flip = psf[::-1, ::-1]
    for _ in range(max(1, iters)):
        conv = np.maximum(fftconvolve(est, psf, mode="same"), 1e-6)
        est = est * fftconvolve(img / conv, flip, mode="same")
        est = np.clip(est, 0.0, 1.0)
    return clip8(est * 255.0)


# --------------------------------------------------------------------------
# periodic noise: photographs of screens
# --------------------------------------------------------------------------

def _spectrum(gray: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))


def _find_spectral_peaks(gray: np.ndarray, threshold: float = 4.0,
                         max_peaks: int = 24) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Isolated spikes in the spectrum, ignoring content and edge artefacts."""
    mag = np.log1p(np.abs(_spectrum(np.asarray(gray, np.float64))))
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    r = np.hypot(yy - cy, xx - cx)
    # skip the low-frequency core (that is real picture content) and the axis
    # cross (the FFT's own edge artefact, present in every image)
    band = (r > 0.10 * min(h, w)) & (np.abs(yy - cy) > 3) & (np.abs(xx - cx) > 3)
    if not band.any():
        return [], mag

    vals = mag[band]
    med = float(np.median(vals))
    sd = float(vals.std()) or 1e-9
    cutoff = med + threshold * sd

    work = np.where(band, mag, -np.inf)
    guard = max(3, int(0.02 * min(h, w)))
    peaks: list[tuple[int, int]] = []
    for _ in range(max_peaks):
        iy, ix = np.unravel_index(int(np.argmax(work)), work.shape)
        if not np.isfinite(work[iy, ix]) or work[iy, ix] < cutoff:
            break
        peaks.append((int(iy), int(ix)))
        work[max(0, iy - guard):iy + guard + 1,
             max(0, ix - guard):ix + guard + 1] = -np.inf
    return peaks, mag


def periodic_noise_strength(gray: np.ndarray, threshold: float = 4.0) -> float:
    """How much regular, screen-like texture is riding on this image.

    Measured as the *number* of isolated spectral spikes, not the height of
    the tallest one. Height alone does not separate a photographed screen from
    ordinary motion blur - both push a single peak up - but a display's pixel
    grid beating against the sensor produces a whole lattice of spikes
    (the two grid frequencies plus their sum and difference terms), and
    nothing in a natural scene does that.
    """
    g = np.asarray(gray)
    if min(g.shape[:2]) < 48:
        return 0.0
    peaks, _ = _find_spectral_peaks(g, threshold)
    return float(len(peaks))


def _notch_mask(gray: np.ndarray, max_notches: int, sigma: float,
                threshold: float) -> np.ndarray | None:
    """Frequency-domain mask that removes the periodic spikes."""
    peaks, mag = _find_spectral_peaks(gray, threshold, max_notches)
    if not peaks:
        return None
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.ogrid[:h, :w]
    mask = np.ones((h, w), np.float64)
    for iy, ix in peaks:
        for py, px in ((iy, ix), (2 * cy - iy, 2 * cx - ix)):
            if 0 <= py < h and 0 <= px < w:
                d2 = (yy - py) ** 2 + (xx - px) ** 2
                mask *= 1.0 - np.exp(-d2 / (2.0 * sigma ** 2))
    return mask


def suppress_periodic_bgr(bgr: np.ndarray, max_notches: int = 24,
                          sigma: float = 2.6, threshold: float = 4.0) -> np.ndarray:
    """Descreen a colour image, keeping its colour.

    The spikes are located once on the luminance and the same notch mask is
    applied to every channel, so the pattern goes and the colour balance
    stays - which matters because the body-colour cross-check reads it later.
    """
    if bgr.ndim != 3:
        return suppress_periodic(bgr, max_notches, sigma, threshold)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if min(gray.shape[:2]) < 48:
        return bgr
    mask = _notch_mask(gray, max_notches, sigma, threshold)
    if mask is None:
        return bgr
    out = np.empty_like(bgr)
    for c in range(3):
        spec = _spectrum(bgr[..., c].astype(np.float64))
        out[..., c] = clip8(np.real(np.fft.ifft2(np.fft.ifftshift(spec * mask))))
    return out


def suppress_periodic(gray: np.ndarray, max_notches: int = 24,
                      sigma: float = 2.6, threshold: float = 4.0) -> np.ndarray:
    """Notch out screen-door / moire patterning from a grayscale image.

    Subtractive, not generative: it removes an interference pattern that was
    added by photographing a display. It does not invent detail.
    """
    g = np.asarray(gray, np.float64)
    if min(g.shape[:2]) < 48:
        return np.asarray(gray, np.uint8)
    mask = _notch_mask(g, max_notches, sigma, threshold)
    if mask is None:
        return clip8(g)
    return clip8(np.real(np.fft.ifft2(np.fft.ifftshift(_spectrum(g) * mask))))


def descreen_if_needed(gray: np.ndarray, force: bool | None = None,
                       threshold: float = 6.0) -> tuple[np.ndarray, float]:
    """Apply periodic-noise suppression when the image looks like a screen shot.

    Returns the (possibly unchanged) image and the measured strength.
    """
    strength = periodic_noise_strength(gray)
    if force is False:
        return np.asarray(gray, np.uint8), strength
    if force is True or strength >= threshold:
        return suppress_periodic(gray), strength
    return np.asarray(gray, np.uint8), strength


# --------------------------------------------------------------------------
# blind PSF search
# --------------------------------------------------------------------------

def _otsu_separability(gray: np.ndarray) -> float:
    """How cleanly the histogram splits into two classes, 0..1.

    Text on a plate is bimodal: ink and background, with little in between.
    The more a deconvolution recovers that split, the closer it is to the
    true PSF - and unlike a sharpness metric this does not simply reward
    whatever output has the loudest ringing.
    """
    g = np.asarray(gray, np.uint8).ravel()
    if g.size < 16:
        return 0.0
    hist = np.bincount(g, minlength=256).astype(np.float64)
    p = hist / hist.sum()
    levels = np.arange(256, dtype=np.float64)
    omega = np.cumsum(p)
    mu = np.cumsum(p * levels)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = np.where(denom > 1e-12, (mu_t * omega - mu) ** 2 / denom, 0.0)
    total_var = float((p * (levels - mu_t) ** 2).sum())
    if total_var < 1e-9:
        return 0.0
    return float(np.clip(np.nanmax(sigma_b) / total_var, 0.0, 1.0))


def _text_rhythm(gray: np.ndarray) -> float:
    """Does this look like separated characters rather than one smear?

    Counting how often the ink profile crosses its own mean is not enough: a
    flat, noisy profile crosses constantly and scores well while being one
    illegible band. Count instead the distinct runs of ink and how empty the
    gaps between them are - a plate has five to eight bodies separated by
    near-empty columns.
    """
    g = np.asarray(gray, np.uint8)
    if min(g.shape[:2]) < 8:
        return 0.0
    g = cv2.resize(g, (200, 48), interpolation=cv2.INTER_AREA)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if b.mean() > 127:
        b = 255 - b
    prof = (b > 0).sum(axis=0).astype(np.float64)
    mean = float(prof.mean())
    if mean < 1e-6:
        return 0.0

    kern = np.ones(5, np.float64) / 5.0
    sm = np.convolve(prof, kern, mode="same")
    on = sm > 0.5 * mean
    runs = int(np.count_nonzero(np.diff(on.astype(np.int8)) == 1) + (1 if on[0] else 0))
    if runs == 0 or on.all():
        return 0.0

    gaps = sm[~on]
    depth = 1.0 - float(gaps.mean()) / mean if gaps.size else 0.0
    count_score = float(np.clip(1.0 - abs(runs - 6.5) / 6.5, 0.0, 1.0))
    return count_score * float(np.clip(depth, 0.0, 1.0))


def deblur_quality(gray: np.ndarray) -> float:
    """Score a deconvolution without a reference image.

    Separability says the ink came back as ink; rhythm says the characters
    came back as *separate* characters, which is the thing heavy blur
    destroys and the thing separability alone will not notice - a smeared
    band can be beautifully bimodal and still be one illegible blob. The
    saturation term penalises the ringing an over-confident PSF produces.
    """
    g = np.asarray(gray, np.uint8)
    sep = _otsu_separability(g)
    rhythm = _text_rhythm(g)
    saturated = float(((g <= 1) | (g >= 254)).mean())
    return (0.5 * sep + 0.5 * rhythm) * float(np.clip(1.0 - 3.0 * saturated, 0.0, 1.0))


def search_defocus_psf(gray: np.ndarray, top_k: int = 2, K: float = 0.012,
                       sigmas=None, radii=None) -> list[tuple[float, str, np.ndarray]]:
    """Blind search over out-of-focus PSFs.

    Motion and defocus fail differently and need different kernels, and the
    fixed sigma-1/sigma-2 variants elsewhere only cover mild softness. A badly
    focused shot can be several pixels of blur radius while still having large,
    well-lit characters - plenty of signal, just spread out - so it is worth
    searching properly rather than giving up.

    Returns (score, label, deconvolved image), best first.
    """
    if min(gray.shape[:2]) < 24:
        return []
    if sigmas is None:
        sigmas = (1.0, 1.6, 2.4, 3.2, 4.2, 5.5)
    if radii is None:
        radii = (2.0, 3.5, 5.0, 7.0)

    scored: list[tuple[float, str, np.ndarray]] = []
    for sig in sigmas:
        try:
            out = wiener_deconv(gray, gaussian_psf(float(sig)), K)
        except Exception:
            continue
        scored.append((deblur_quality(out), f"gauss{sig:.1f}", out))
    for rad in radii:
        try:
            out = wiener_deconv(gray, disc_psf(float(rad)), K)
        except Exception:
            continue
        scored.append((deblur_quality(out), f"disc{rad:.1f}", out))

    scored.sort(key=lambda t: -t[0])
    return scored[:top_k]


def search_motion_psf(gray: np.ndarray, top_k: int = 3, K: float = 0.012,
                      angles=None, lengths=None) -> list[tuple[float, float, float, np.ndarray]]:
    """Try motion PSFs directly instead of trusting a single estimate.

    The cepstrum reads blur length and direction off one spectral pattern,
    which is reliable on clean images and increasingly not so as blur grows -
    exactly when it matters. Deconvolving with a grid of PSFs and keeping
    whichever outputs actually look like text is slower but does not depend on
    that estimate being right.

    Returns (score, length, angle, deconvolved image), best first.
    """
    if min(gray.shape[:2]) < 24:
        return []
    if angles is None:
        angles = np.arange(0.0, 180.0, 15.0)
    if lengths is None:
        lengths = (5.0, 8.0, 12.0, 17.0, 23.0, 30.0)

    scored: list[tuple[float, float, float, np.ndarray]] = []
    for ang in angles:
        for length in lengths:
            try:
                out = wiener_deconv(gray, motion_psf(length, float(ang)), K)
            except Exception:
                continue
            scored.append((deblur_quality(out), float(length), float(ang), out))

    scored.sort(key=lambda t: -t[0])
    # keep the best few, but spread across distinct directions so the shortlist
    # is not five near-identical angles
    picked: list[tuple[float, float, float, np.ndarray]] = []
    for item in scored:
        if all(min(abs(item[2] - p[2]), 180 - abs(item[2] - p[2])) >= 20.0
               for p in picked):
            picked.append(item)
        if len(picked) >= top_k:
            break
    return picked or scored[:top_k]


def deblur_bgr(bgr: np.ndarray, search_side: int = 640,
               K: float = 0.015) -> tuple[np.ndarray, str]:
    """Deconvolve a whole photograph before anything looks for a plate in it.

    Every geometry step keys on edges: the detector on gradient density, corner
    refinement on Hough lines, the rotation search on how sharply the text band
    steps. Blur removes exactly that, so on a blurred *and* angled shot the
    plate is either missed or found with a quad too rough to unwarp - and the
    per-plate deconvolution downstream never gets a straight plate to work on.
    Restoring first breaks that ordering.

    The PSF is searched on a downscaled copy (cheap) and applied at full
    resolution (accurate), with the length rescaled to match.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    h, w = gray.shape[:2]
    if min(h, w) < 48:
        return bgr, "too-small"

    scale = min(1.0, search_side / float(max(h, w)))
    small = (cv2.resize(gray, (int(w * scale), int(h * scale)),
                        interpolation=cv2.INTER_AREA) if scale < 1.0 else gray)

    best_psf, label, best_score = None, "none", -1.0
    for score, length, angle, _ in search_motion_psf(small, top_k=1):
        if score > best_score:
            best_psf = motion_psf(length / max(scale, 1e-6), angle)
            label, best_score = f"motion L{length / max(scale, 1e-6):.0f}@{angle:.0f}", score
    for score, lab, _ in search_defocus_psf(small, top_k=1):
        if score > best_score:
            sigma = float(lab.replace("gauss", "").replace("disc", ""))
            big = sigma / max(scale, 1e-6)
            best_psf = (gaussian_psf(big) if lab.startswith("gauss")
                        else disc_psf(big))
            label, best_score = f"{lab} -> {big:.1f}", score

    if best_psf is None:
        return bgr, "none"
    if bgr.ndim == 2:
        return wiener_deconv(bgr, best_psf, K), label
    out = np.empty_like(bgr)
    for c in range(3):
        out[..., c] = wiener_deconv(bgr[..., c], best_psf, K)
    return out, label


# --------------------------------------------------------------------------
# cheap enhancers - no inverse model, so far safer than deconvolution
# --------------------------------------------------------------------------

def unsharp(gray: np.ndarray, sigma: float = 1.2, amount: float = 1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(gray.astype(np.float64), (0, 0), sigma)
    return clip8(gray.astype(np.float64) * (1 + amount) - blur * amount)


def clahe(gray: np.ndarray, clip: float = 3.0, grid: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(gray.astype(np.uint8))


def denoise(gray: np.ndarray, h: float = 7.0) -> np.ndarray:
    return cv2.fastNlMeansDenoising(gray.astype(np.uint8), None, h, 7, 21)


def upscale(gray: np.ndarray, factor: float) -> np.ndarray:
    """Resample larger. Adds no information - it only gives OCR room to work."""
    if factor <= 1.0:
        return gray
    h, w = gray.shape[:2]
    return cv2.resize(gray, (int(round(w * factor)), int(round(h * factor))),
                      interpolation=cv2.INTER_CUBIC)


def normalize_illumination(gray: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Divide out a slow background: fixes shadow, glare and angled lighting."""
    g = gray.astype(np.float64) + 1.0
    bg = cv2.GaussianBlur(g, (0, 0), sigma)
    return norm8(g / np.maximum(bg, 1.0))

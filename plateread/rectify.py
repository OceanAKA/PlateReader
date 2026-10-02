"""Undoing geometry: perspective, rotation and shear.

These are the distortions that are genuinely, losslessly invertible - a plate
photographed from 40 degrees off-axis still contains every pixel of the
characters, just resampled onto a trapezoid. Getting the geometry right is
usually worth more than any amount of sharpening.
"""
from __future__ import annotations

import cv2
import numpy as np

from .detect import order_quad
from .util import smooth1d, to_gray


def pad_quad(quad: np.ndarray, pad_x: float = 0.13, pad_y: float = 0.16) -> np.ndarray:
    """Grow a quad along its own axes.

    Detectors routinely clip the first or last glyph by a few pixels, and a
    half-eaten H reads as an I. Padding outward costs nothing because
    trim_border crops back to the real ink afterwards.
    """
    q = order_quad(quad).astype(np.float64)
    tl, tr, br, bl = q
    u = tr - tl
    v = bl - tl
    wu, hv = np.linalg.norm(u), np.linalg.norm(v)
    if wu < 1e-6 or hv < 1e-6:
        return quad
    u, v = u / wu, v / hv
    dx, dy = u * (pad_x * wu), v * (pad_y * hv)
    return np.array([tl - dx - dy, tr + dx - dy, br + dx + dy, bl - dx + dy], np.float32)


def warp_quad(img: np.ndarray, quad: np.ndarray, target_h: int = 128,
              max_aspect: float = 9.0) -> np.ndarray:
    """Perspective-unwarp a quadrilateral to a front-on rectangle."""
    q = order_quad(quad)
    w_top = np.linalg.norm(q[1] - q[0])
    w_bot = np.linalg.norm(q[2] - q[3])
    h_lef = np.linalg.norm(q[3] - q[0])
    h_rig = np.linalg.norm(q[2] - q[1])
    src_w = max(w_top, w_bot)
    src_h = max(h_lef, h_rig)
    if src_w < 4 or src_h < 3:
        return to_gray(img)

    aspect = float(np.clip(src_w / src_h, 0.8, max_aspect))
    out_h = int(target_h)
    out_w = max(8, int(round(out_h * aspect)))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    m = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    return cv2.warpPerspective(img, m, (out_w, out_h),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def rotate_bound(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre, growing the canvas so nothing is clipped."""
    h, w = img.shape[:2]
    c = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(c, angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw = int(h * sin + w * cos)
    nh = int(h * cos + w * sin)
    m[0, 2] += nw / 2.0 - c[0]
    m[1, 2] += nh / 2.0 - c[1]
    return cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _row_profile_energy(gray: np.ndarray) -> float:
    """Sharpness of the horizontal text band.

    When text sits level, the row-wise ink profile has steep shoulders: rows
    inside the band are full, rows outside are empty. Tilt smears that step,
    so the squared row-to-row difference peaks at the correct angle.
    """
    g = gray.astype(np.float64)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    prof = np.abs(cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)).sum(axis=1)
    if prof.size < 4:
        return 0.0
    return float((np.diff(prof) ** 2).sum())


def estimate_rotation(gray: np.ndarray, max_angle: float = 46.0,
                      coarse: float = 2.0, fine: float = 0.25) -> float:
    """Coarse-to-fine search for the angle that makes the text band sharpest."""
    if min(gray.shape[:2]) < 12:
        return 0.0
    small = gray
    if max(gray.shape[:2]) > 400:
        s = 400.0 / max(gray.shape[:2])
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    def score(a: float) -> float:
        return _row_profile_energy(rotate_bound(small, a))

    angles = np.arange(-max_angle, max_angle + 1e-9, coarse)
    best = max(angles, key=score)
    lo, hi = best - coarse, best + coarse
    angles = np.arange(lo, hi + 1e-9, fine)
    best = max(angles, key=score)
    return float(best)


def apply_shear(img: np.ndarray, shear: float) -> np.ndarray:
    """Horizontal shear about the vertical centre (undoes italic slant)."""
    h, w = img.shape[:2]
    dx = abs(shear) * h
    m = np.array([[1.0, shear, -shear * h / 2.0 + dx / 2.0], [0.0, 1.0, 0.0]], np.float64)
    return cv2.warpAffine(img, m, (int(w + dx), h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def estimate_shear(binary: np.ndarray, limit: float = 0.6, step: float = 0.04) -> float:
    """Find the shear that makes the gaps between characters deepest.

    Slanted glyphs overlap in projection and fill the valleys between them.
    The correct de-slant maximises the variance of the column ink profile.
    """
    if min(binary.shape[:2]) < 12:
        return 0.0
    best_s, best_v = 0.0, -1.0
    for s in np.arange(-limit, limit + 1e-9, step):
        sheared = apply_shear(binary, float(s))
        prof = (sheared > 127).sum(axis=0).astype(np.float64)
        if prof.size < 8:
            continue
        v = float(prof.var() + 0.5 * (np.diff(prof) ** 2).mean())
        if v > best_v:
            best_v, best_s = v, float(s)
    return best_s


def trim_border(gray: np.ndarray, binary: np.ndarray | None = None,
                margin: float = 0.04) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop the plate frame, mounting screws and state banner off the edges.

    Keeps whatever horizontal band actually contains a dense run of ink.
    """
    h, w = gray.shape[:2]
    if binary is None:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if binary.mean() > 127:
            binary = 255 - binary

    rows = smooth1d((binary > 0).sum(axis=1).astype(np.float64), 3)
    thr = max(2.0, 0.12 * w)
    on = rows > thr
    if not on.any():
        return gray, (0, 0, w, h)

    # longest contiguous run of "inky" rows
    best_len, best_start = 0, 0
    run_len, run_start = 0, 0
    for i, v in enumerate(on):
        if v:
            if run_len == 0:
                run_start = i
            run_len += 1
            if run_len > best_len:
                best_len, best_start = run_len, run_start
        else:
            run_len = 0
    y0, y1 = best_start, best_start + best_len
    if best_len < h * 0.25:            # band too thin to trust - keep it all
        y0, y1 = 0, h
    pad = int(round(h * margin))
    y0, y1 = max(0, y0 - pad), min(h, y1 + pad)

    band = binary[y0:y1]
    cols = smooth1d((band > 0).sum(axis=0).astype(np.float64), 3)
    on_c = cols > max(1.0, 0.06 * (y1 - y0))
    xs = np.flatnonzero(on_c)
    if xs.size:
        padx = int(round(w * 0.01))
        x0, x1 = max(0, int(xs[0]) - padx), min(w, int(xs[-1]) + 1 + padx)
    else:
        x0, x1 = 0, w
    if x1 - x0 < w * 0.3:
        x0, x1 = 0, w
    return gray[y0:y1, x0:x1], (x0, y0, x1, y1)


def rectify(img: np.ndarray, quad: np.ndarray | None = None, target_h: int = 128,
            do_rotate: bool = True, do_shear: bool = True,
            pad: bool = True, pad_x: float = 0.13, pad_y: float = 0.16) -> np.ndarray:
    """Full geometric normalisation: unwarp, level, de-slant, rescale."""
    gray = to_gray(img)
    if quad is not None:
        q = pad_quad(quad, pad_x, pad_y) if pad else quad
        out = warp_quad(gray, q, target_h=target_h)
    else:
        out = gray

    if do_rotate:
        ang = estimate_rotation(out, max_angle=46.0)
        if abs(ang) > 0.3:
            out = rotate_bound(out, ang)

    if do_shear:
        _, b = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if b.mean() > 127:
            b = 255 - b
        s = estimate_shear(b)
        if abs(s) > 0.02:
            out = apply_shear(out, s)

    if out.shape[0] != target_h and out.shape[0] > 0:
        scale = target_h / float(out.shape[0])
        interp = cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
        out = cv2.resize(out, (max(8, int(round(out.shape[1] * scale))), target_h),
                         interpolation=interp)
    return out

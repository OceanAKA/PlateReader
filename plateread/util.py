"""Small shared helpers used across the pipeline."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
DIGITS = "0123456789"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def imread_any(path) -> np.ndarray:
    """cv2.imread that survives non-ASCII Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise IOError(f"empty or unreadable file: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"not a decodable image: {path}")
    return img


def imwrite_any(path, img) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".png", img)
    if not ok:
        raise IOError(f"could not encode {path}")
    buf.tofile(str(path))


def to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def norm8(a: np.ndarray) -> np.ndarray:
    """Rescale any float array to a full-range uint8 image."""
    a = np.asarray(a, dtype=np.float64)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return np.zeros(a.shape, np.uint8)
    return np.clip((a - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def clip8(a: np.ndarray) -> np.ndarray:
    """Clamp to uint8 without rescaling (preserves absolute brightness)."""
    return np.clip(np.asarray(a), 0, 255).astype(np.uint8)


def resize_max(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return img
    s = max_side / float(m)
    return cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)


def scale_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h or h == 0:
        return img
    s = target_h / float(h)
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    return cv2.resize(img, (max(1, int(round(w * s))), target_h), interpolation=interp)


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray.astype(np.float64), cv2.CV_64F).var())


def rms_contrast(gray: np.ndarray) -> float:
    g = gray.astype(np.float64) / 255.0
    return float(g.std())


def percentile_spread(gray: np.ndarray, lo: float = 5, hi: float = 95) -> float:
    a, b = np.percentile(gray, [lo, hi])
    return float(b - a)


def stroke_width(binary: np.ndarray) -> float:
    """Median stroke half-width*2 of the ink, via the distance transform."""
    ink = (binary > 0).astype(np.uint8)
    if ink.sum() < 20:
        return 0.0
    dist = cv2.distanceTransform(ink, cv2.DIST_L2, 3)
    vals = dist[dist > 0]
    if vals.size == 0:
        return 0.0
    # ridge pixels carry the half-width; take the upper quartile to skip edges
    return float(2.0 * np.percentile(vals, 75))


def ink_fraction(binary: np.ndarray) -> float:
    return float((binary > 0).mean())


def box_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / float(ua) if ua > 0 else 0.0


def smooth1d(a: np.ndarray, k: int) -> np.ndarray:
    k = max(1, int(k) | 1)
    if k == 1:
        return a.astype(np.float64)
    kern = np.ones(k, np.float64) / k
    return np.convolve(a.astype(np.float64), kern, mode="same")

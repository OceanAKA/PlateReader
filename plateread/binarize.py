"""Turning grey pixels into ink.

Every method here returns ink=255 on a 0 background, with polarity resolved
automatically, so downstream code never has to care whether the plate was
dark-on-light or light-on-dark.
"""
from __future__ import annotations

import cv2
import numpy as np


def _as_ink(binary: np.ndarray) -> np.ndarray:
    """Force the ink (the characters) to be 255.

    Characters cover less of a *plate* than its background does, so the
    minority class is the text - but only when what you are measuring is
    actually the plate. Judge it on the middle of the crop: a crop that still
    carries dark bumper or shadow around the edges can easily have more dark
    pixels than light overall, and then a whole-image minority test decides
    the plate's own background is the ink and inverts everything downstream.
    """
    b = (binary > 0).astype(np.uint8)
    h, w = b.shape[:2]
    y0, y1 = int(h * 0.20), max(int(h * 0.80), int(h * 0.20) + 1)
    x0, x1 = int(w * 0.20), max(int(w * 0.80), int(w * 0.20) + 1)
    core = b[y0:y1, x0:x1]
    frac = float(core.mean()) if core.size else float(b.mean())
    if frac > 0.5:
        b = 1 - b
    return (b * 255).astype(np.uint8)


def otsu(gray: np.ndarray) -> np.ndarray:
    _, b = cv2.threshold(gray.astype(np.uint8), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _as_ink(b)


def adaptive_gaussian(gray: np.ndarray, block: int | None = None, c: int = 9) -> np.ndarray:
    h = gray.shape[0]
    if block is None:
        block = max(11, int(h * 0.35) | 1)
    block = max(3, int(block) | 1)
    b = cv2.adaptiveThreshold(gray.astype(np.uint8), 255,
                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY, block, c)
    return _as_ink(b)


def sauvola(gray: np.ndarray, window: int | None = None, k: float = 0.2,
            r: float = 128.0) -> np.ndarray:
    """Sauvola local thresholding - the strongest option under uneven light.

    T = mean * (1 + k * (std / R - 1)), computed with box filters so it runs
    in O(pixels) regardless of window size.
    """
    g = gray.astype(np.float64)
    h = g.shape[0]
    if window is None:
        window = max(15, int(h * 0.5) | 1)
    window = max(3, int(window) | 1)
    ksz = (window, window)
    mean = cv2.boxFilter(g, cv2.CV_64F, ksz, normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(g * g, cv2.CV_64F, ksz, normalize=True, borderType=cv2.BORDER_REPLICATE)
    var = np.maximum(mean_sq - mean * mean, 0.0)
    std = np.sqrt(var)
    thresh = mean * (1.0 + k * (std / r - 1.0))
    return _as_ink((g > thresh).astype(np.uint8) * 255)


def niblack(gray: np.ndarray, window: int | None = None, k: float = -0.2) -> np.ndarray:
    g = gray.astype(np.float64)
    if window is None:
        window = max(15, int(g.shape[0] * 0.5) | 1)
    window = max(3, int(window) | 1)
    ksz = (window, window)
    mean = cv2.boxFilter(g, cv2.CV_64F, ksz, normalize=True, borderType=cv2.BORDER_REPLICATE)
    mean_sq = cv2.boxFilter(g * g, cv2.CV_64F, ksz, normalize=True, borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    return _as_ink((g > mean + k * std).astype(np.uint8) * 255)


def clean(binary: np.ndarray, min_area_frac: float = 0.0008) -> np.ndarray:
    """Drop speckle and close 1px stroke breaks left by aggressive sharpening."""
    b = (binary > 0).astype(np.uint8)
    # bridge hairline breaks - a thresholded crossbar that drops out turns one
    # H into two 1s downstream, which no amount of voting recovers from
    k = int(np.clip(round(b.shape[0] / 40.0), 2, 4))
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    if n <= 1:
        return (b * 255).astype(np.uint8)
    total = b.shape[0] * b.shape[1]
    keep = np.zeros(n, bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= max(4, total * min_area_frac):
            keep[i] = True
    return (keep[labels] * 255).astype(np.uint8)


ALL_METHODS = {
    "otsu": otsu,
    "sauvola": sauvola,
    "adaptive": adaptive_gaussian,
    "niblack": niblack,
}

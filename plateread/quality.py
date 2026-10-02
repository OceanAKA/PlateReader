"""How much information is actually left in these pixels.

This is the part that keeps the tool honest. Deconvolution and upscaling can
make an unreadable plate *look* readable, and a classifier will always return
its best guess with no sense of whether the evidence justified one. These
checks are computed on the plate at its native resolution - before any
enhancement - so they describe the photograph, not the processing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .restore import estimate_motion_blur
from .util import percentile_spread, rms_contrast, stroke_width, variance_of_laplacian

# Rules of thumb from OCR practice: characters need ~20px of height to be read
# reliably, ~12px to be guessed at, and below that the glyph simply does not
# contain enough samples to distinguish 8 from B no matter what you run on it.
CHAR_H_GOOD = 20.0
CHAR_H_MARGINAL = 12.0

VERDICT_GOOD = "good"
VERDICT_MARGINAL = "marginal"
VERDICT_INSUFFICIENT = "insufficient"


@dataclass
class Quality:
    char_height_px: float
    stroke_px: float
    blur_length: float
    blur_angle: float
    blur_confident: bool
    sharpness: float
    contrast: float
    dynamic_range: float
    score: float                       # 0..1 information budget
    verdict: str
    char_height_measured: bool = True  # False when the number is a guess
    notes: list[str] = field(default_factory=list)

    @property
    def readable(self) -> bool:
        return self.verdict != VERDICT_INSUFFICIENT


def assess(native_plate: np.ndarray, binary: np.ndarray | None = None,
           char_height_px: float | None = None) -> Quality:
    """Judge a plate crop at its original scale.

    `char_height_px` should be the height of the characters actually
    segmented, measured in the original photograph's pixels. Pass it whenever
    it is available: the fallback below assumes the crop really is a plate,
    and if the detector locked onto a door panel instead, that assumption
    reports a large character height - and a reassuring verdict - for a region
    containing no characters at all.
    """
    g = native_plate if native_plate.ndim == 2 else cv2.cvtColor(native_plate, cv2.COLOR_BGR2GRAY)
    h = float(g.shape[0])
    measured = bool(char_height_px)
    # characters occupy roughly 55% of plate height on every common format
    char_h = float(char_height_px) if measured else h * 0.55

    if binary is None:
        _, binary = cv2.threshold(g.astype(np.uint8), 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if binary.mean() > 127:
            binary = 255 - binary
    stroke = stroke_width(binary)

    be = estimate_motion_blur(g)
    sharp = variance_of_laplacian(g)
    contrast = rms_contrast(g)
    drange = percentile_spread(g, 2, 98)

    notes: list[str] = []

    # --- resolution: the hard limit nothing can undo ---------------------
    if char_h >= CHAR_H_GOOD:
        res_score = 1.0
    elif char_h >= CHAR_H_MARGINAL:
        res_score = 0.35 + 0.65 * (char_h - CHAR_H_MARGINAL) / (CHAR_H_GOOD - CHAR_H_MARGINAL)
        notes.append(
            f"Characters are only ~{char_h:.0f}px tall. Below ~{CHAR_H_GOOD:.0f}px "
            "similar glyphs (8/B, 0/O, 5/S) stop being separable.")
    else:
        res_score = max(0.0, 0.35 * char_h / CHAR_H_MARGINAL)
        notes.append(
            f"Characters are ~{char_h:.0f}px tall. That is below the resolution "
            "at which a plate can be read; upscaling cannot add the missing detail.")

    # --- stroke vs blur: is the ink still distinguishable from the gaps ---
    if stroke > 0:
        if stroke < 1.5:
            notes.append(f"Stroke width is ~{stroke:.1f}px - strokes are thinner "
                         "than the sampling grid, so shapes are aliased.")
        stroke_score = float(np.clip((stroke - 1.0) / 2.5, 0.0, 1.0))
    else:
        stroke_score = 0.0
        notes.append("No coherent ink found - the crop may not be a plate.")

    if be.trustworthy:
        notes.append(f"Motion blur detected: ~{be.length:.0f}px at {be.angle:.0f} degrees. "
                     "Deconvolution will be attempted along that direction.")
        severity = be.length / max(stroke, 1.0)
        blur_score = float(np.clip(1.0 - (severity - 1.0) / 6.0, 0.0, 1.0))
        if severity > 4:
            notes.append("Blur extent is several times the stroke width; recovered "
                         "characters here are inference, not measurement.")
    else:
        blur_score = float(np.clip(sharp / 120.0, 0.0, 1.0))
        if sharp < 25:
            notes.append(f"Very low high-frequency content (Laplacian variance "
                         f"{sharp:.0f}); the image is soft or heavily compressed.")

    # --- contrast --------------------------------------------------------
    contrast_score = float(np.clip(drange / 90.0, 0.0, 1.0))
    if drange < 40:
        notes.append(f"Low dynamic range ({drange:.0f}/255) - glare, dusk or "
                     "heavy compression flattening the plate.")

    score = float(np.clip(
        0.45 * res_score + 0.20 * stroke_score + 0.20 * blur_score + 0.15 * contrast_score,
        0.0, 1.0))
    # Resolution is not one factor among four - it is a gate. A crop whose
    # characters are eight pixels tall cannot be read no matter how sharp,
    # well lit or high contrast it is, and letting the other terms carry it to
    # a middling score lets an unreadable crop outrank a good one.
    if char_h < CHAR_H_MARGINAL:
        score = min(score, 0.30)

    if char_h < CHAR_H_MARGINAL or score < 0.30:
        verdict = VERDICT_INSUFFICIENT
    elif char_h < CHAR_H_GOOD or score < 0.60:
        verdict = VERDICT_MARGINAL
    else:
        verdict = VERDICT_GOOD

    return Quality(
        char_height_px=char_h, stroke_px=stroke,
        blur_length=be.length, blur_angle=be.angle, blur_confident=be.trustworthy,
        sharpness=sharp, contrast=contrast, dynamic_range=drange,
        score=score, verdict=verdict, notes=notes,
        char_height_measured=measured,
    )


def mark_no_plate_found(q: Quality, reason: str) -> None:
    """Withdraw the quality claim entirely.

    The image score answers "did the photograph carry enough detail" - a
    question that only means anything once a plate has actually been found.
    When the reading itself shows we have not found one, reporting a
    confident score about whatever region was cropped is worse than useless,
    because it reads as a guarantee.
    """
    q.score = min(q.score, 0.25)
    q.verdict = VERDICT_INSUFFICIENT
    q.notes.append(reason)


def verdict_sentence(q: Quality) -> str:
    if q.verdict == VERDICT_GOOD:
        return "Enough detail is present for a reliable read."
    if q.verdict == VERDICT_MARGINAL:
        return ("Marginal: a read is possible but individual characters may be wrong. "
                "Treat the result as a lead, not an identification.")
    return ("Insufficient: the pixels no longer carry enough information to identify "
            "this plate. Any string below is a guess and should not be relied on.")

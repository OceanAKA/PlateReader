"""Finding the plate inside a full photo.

Two independent detectors vote:

  * gradient density - a row of characters produces a dense band of vertical
    edges packed far tighter than anything else in a street scene. Closing
    that band with a wide kernel fuses it into one solid blob shaped like a
    plate.
  * MSER text clustering - stable regions the size and shape of characters,
    then grouped into collinear runs.

Candidates are scored on edge density, aspect plausibility and "rhythm" (the
regular peak-valley beat that only a row of glyphs produces), then merged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .util import box_iou, to_gray


# Plates run from ~1.4:1 (stacked motorcycle) to ~5.5:1 (EU single-line).
ASPECT_MIN, ASPECT_MAX = 1.3, 8.0
ASPECT_IDEAL_LO, ASPECT_IDEAL_HI = 2.0, 5.2


@dataclass
class Candidate:
    quad: np.ndarray                     # (4,2) float32, ordered TL TR BR BL
    score: float
    source: str
    detail: dict = field(default_factory=dict)

    @property
    def bbox(self):
        x0, y0 = self.quad.min(axis=0)
        x1, y1 = self.quad.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],   # TL has the smallest x+y
        pts[np.argmin(d)],   # TR has the smallest y-x
        pts[np.argmax(s)],   # BR has the largest x+y
        pts[np.argmax(d)],   # BL has the largest y-x
    ], np.float32)


def _rect_quad(rect) -> np.ndarray:
    return order_quad(cv2.boxPoints(rect))


def _aspect_prior(aspect: float) -> float:
    """1.0 across the plausible plate band, tapering outside it."""
    if aspect <= 0:
        return 0.0
    if ASPECT_IDEAL_LO <= aspect <= ASPECT_IDEAL_HI:
        return 1.0
    if aspect < ASPECT_IDEAL_LO:
        return max(0.0, 1.0 - (ASPECT_IDEAL_LO - aspect) / (ASPECT_IDEAL_LO - ASPECT_MIN))
    return max(0.0, 1.0 - (aspect - ASPECT_IDEAL_HI) / (ASPECT_MAX - ASPECT_IDEAL_HI))


def _rhythm_score(gray_crop: np.ndarray) -> float:
    """How much the crop beats like a row of characters.

    Binarise, take the column-wise ink profile, and count how often it crosses
    its own mean. A plate with 6-8 glyphs crosses 12-16 times; a wall, a
    bumper or a shadow crosses far less.
    """
    if gray_crop.size < 64 or gray_crop.shape[0] < 8:
        return 0.0
    g = cv2.resize(gray_crop, (160, 48), interpolation=cv2.INTER_AREA)
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if b.mean() > 127:
        b = 255 - b
    prof = (b > 0).sum(axis=0).astype(np.float64)
    if prof.std() < 1e-6:
        return 0.0
    centred = prof - prof.mean()
    crossings = int((np.diff(np.sign(centred)) != 0).sum())
    # 10-20 crossings is the sweet spot for 5-8 characters
    return float(np.clip(1.0 - abs(crossings - 15) / 15.0, 0.0, 1.0))


def _warp_for_score(gray: np.ndarray, quad: np.ndarray,
                    out_h: int = 48) -> tuple[np.ndarray | None, float, float]:
    """Straighten a candidate before judging it.

    A plate tilted 35 degrees fills barely half of its axis-aligned bounding
    box; the rest is background. Measuring density, rhythm or aspect on that
    box describes the background as much as the plate, so every tilted
    candidate scored badly and lost to upright clutter.
    """
    q = order_quad(quad)
    src_w = max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))
    src_h = max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))
    if src_w < 8 or src_h < 5:
        return None, float(src_w), float(src_h)
    aspect = float(np.clip(src_w / src_h, 0.5, 12.0))
    out_w = max(12, int(round(out_h * aspect)))
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   np.float32)
    m = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    warped = cv2.warpPerspective(gray, m, (out_w, out_h), flags=cv2.INTER_AREA,
                                 borderMode=cv2.BORDER_REPLICATE)
    return warped, float(src_w), float(src_h)


def _score_candidate(gray: np.ndarray, quad: np.ndarray) -> tuple[float, dict]:
    crop, src_w, src_h = _warp_for_score(gray, quad)
    if crop is None or src_h < 6:
        return 0.0, {}

    aspect = src_w / max(src_h, 1e-6)
    prior = _aspect_prior(aspect)
    if prior <= 0.0:
        return 0.0, {"aspect": round(aspect, 2)}

    sob = np.abs(cv2.Sobel(crop, cv2.CV_32F, 1, 0, ksize=3))
    sob = cv2.normalize(sob, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, e = cv2.threshold(sob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    density = float((e > 0).mean())
    rhythm = _rhythm_score(crop)
    contrast = float(np.clip(crop.std() / 60.0, 0.0, 1.0))

    score = prior * (0.40 * rhythm + 0.30 * min(density * 2.5, 1.0) + 0.30 * contrast)
    return float(score), {
        "aspect": round(aspect, 2),
        "edge_density": round(density, 3),
        "rhythm": round(rhythm, 3),
        "contrast": round(contrast, 3),
        "height_px": int(round(src_h)),
    }


# --------------------------------------------------------------------------
# detector A: gradient density
# --------------------------------------------------------------------------

def detect_by_gradient(bgr: np.ndarray) -> list[Candidate]:
    gray = to_gray(bgr)
    h, w = gray.shape[:2]
    g = cv2.bilateralFilter(gray, 7, 40, 40)

    sob = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)   # vertical strokes
    sob = np.abs(sob)
    sob = cv2.normalize(sob, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges = cv2.threshold(sob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    out: list[Candidate] = []
    # The closing kernel has to bridge the gaps between characters, and that
    # gap depends on how big the plate is in frame - which is what we are
    # trying to find out. So sweep a ladder of widths rather than guessing one
    # from the image size: narrow kernels catch a plate far away, wide ones
    # catch a plate that fills the frame.
    ladder = (5, 9, 15, 23, 33, 45, 61)
    kws = sorted({max(3, k | 1) for k in ladder if k < w * 0.6})
    for kw in kws:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, max(3, kw // 4) | 1))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        found = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]
        for c in contours:
            area = cv2.contourArea(c)
            if area < (h * w) * 2e-4 or area > (h * w) * 0.6:
                continue
            rect = cv2.minAreaRect(c)
            (_, _), (rw, rh), _ = rect
            if rw < 1 or rh < 1:
                continue
            long_s, short_s = max(rw, rh), min(rw, rh)
            if short_s < 8:
                continue
            aspect = long_s / short_s
            if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
                continue
            if area / (rw * rh) < 0.35:      # contour must actually fill its box
                continue
            quad = _rect_quad(rect)
            s, detail = _score_candidate(gray, quad)
            if s > 0.12:
                out.append(Candidate(quad, s, "gradient", detail))
    return out


# --------------------------------------------------------------------------
# detector B: MSER character clustering
# --------------------------------------------------------------------------

def detect_by_mser(bgr: np.ndarray) -> list[Candidate]:
    gray = to_gray(bgr)
    h, w = gray.shape[:2]
    try:
        mser = cv2.MSER_create()
        try:
            # generous upper bound: when the plate fills the frame a single
            # character is a large fraction of the image
            mser.setMinArea(max(10, int(h * w * 1e-5)))
            mser.setMaxArea(int(h * w * 0.08))
        except AttributeError:
            pass
        regions, _ = mser.detectRegions(gray)
    except Exception:
        return []

    boxes = []
    for r in regions:
        x, y, bw, bh = cv2.boundingRect(r.reshape(-1, 1, 2))
        if bh < 8 or bh > h * 0.5 or bw < 2:
            continue
        ar = bw / float(bh)
        if not (0.08 <= ar <= 1.3):          # glyph-shaped
            continue
        boxes.append((x, y, bw, bh))
    if len(boxes) < 3:
        return []

    # de-duplicate heavily overlapping MSER hits
    boxes.sort(key=lambda b: (-b[3], b[0]))
    kept: list[tuple] = []
    for b in boxes:
        bb = (b[0], b[1], b[0] + b[2], b[1] + b[3])
        if all(box_iou(bb, (k[0], k[1], k[0] + k[2], k[1] + k[3])) < 0.5 for k in kept):
            kept.append(b)
    kept.sort(key=lambda b: b[0])

    # group into collinear, similarly-sized runs = a line of plate characters
    groups: list[list[tuple]] = []
    for b in kept:
        placed = False
        for grp in groups:
            last = grp[-1]
            same_size = abs(b[3] - last[3]) <= 0.4 * max(b[3], last[3])
            same_line = abs((b[1] + b[3] / 2) - (last[1] + last[3] / 2)) <= 0.5 * last[3]
            near = 0 <= (b[0] - (last[0] + last[2])) <= 2.2 * last[3]
            if same_size and same_line and near:
                grp.append(b)
                placed = True
                break
        if not placed:
            groups.append([b])

    sob = np.abs(cv2.Sobel(cv2.bilateralFilter(gray, 5, 30, 30), cv2.CV_32F, 1, 0, ksize=3))
    sob = cv2.normalize(sob, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, edges = cv2.threshold(sob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    out: list[Candidate] = []
    for grp in groups:
        if len(grp) < 3:
            continue
        xs0 = min(b[0] for b in grp)
        ys0 = min(b[1] for b in grp)
        xs1 = max(b[0] + b[2] for b in grp)
        ys1 = max(b[1] + b[3] for b in grp)
        pad_x = int(0.06 * (xs1 - xs0))
        pad_y = int(0.35 * (ys1 - ys0))
        xs0, ys0 = max(0, xs0 - pad_x), max(0, ys0 - pad_y)
        xs1, ys1 = min(w, xs1 + pad_x), min(h, ys1 + pad_y)
        if xs1 - xs0 < 20 or ys1 - ys0 < 10:
            continue
        quad = np.array([[xs0, ys0], [xs1, ys0], [xs1, ys1], [xs0, ys1]], np.float32)
        s, detail = _score_candidate(gray, quad)
        detail["n_glyphs"] = len(grp)
        s *= min(1.0, 0.5 + 0.1 * len(grp))      # more glyphs = more confidence
        if s > 0.12:
            out.append(Candidate(quad, s, "mser", detail))
    return out


# --------------------------------------------------------------------------
# corner refinement
# --------------------------------------------------------------------------

def refine_quad(gray: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Snap a box onto the plate's real border using its four dominant lines.

    minAreaRect only models rotation. A plate shot from an angle is a general
    quadrilateral, and getting the true corners is what makes the perspective
    unwarp put the characters back on a rectangle.
    """
    x0, y0 = np.floor(quad.min(axis=0)).astype(int)
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int)
    h, w = gray.shape[:2]
    mx = int(0.12 * (x1 - x0)) + 3
    my = int(0.25 * (y1 - y0)) + 3
    x0, y0 = max(0, x0 - mx), max(0, y0 - my)
    x1, y1 = min(w, x1 + mx), min(h, y1 + my)
    crop = gray[y0:y1, x0:x1]
    if crop.size < 400 or crop.shape[0] < 12:
        return quad

    edges = cv2.Canny(cv2.GaussianBlur(crop, (5, 5), 0), 40, 130)
    ch, cw = crop.shape[:2]
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360,
                            threshold=max(18, int(cw * 0.18)),
                            minLineLength=int(cw * 0.35), maxLineGap=int(cw * 0.12))
    if lines is None or len(lines) == 0:
        return quad
    # OpenCV 4 returns (N,1,4); OpenCV 5 returns (N,4)
    segs = np.asarray(lines).reshape(-1, 4)

    horiz, vert = [], []
    for lx0, ly0, lx1, ly1 in segs:
        ang = abs(np.degrees(np.arctan2(ly1 - ly0, lx1 - lx0)))
        if ang > 90:
            ang = 180 - ang
        if ang < 30:
            horiz.append((lx0, ly0, lx1, ly1))
        elif ang > 60:
            vert.append((lx0, ly0, lx1, ly1))
    if len(horiz) < 2 or len(vert) < 2:
        return quad

    top = min(horiz, key=lambda l: (l[1] + l[3]) / 2)
    bot = max(horiz, key=lambda l: (l[1] + l[3]) / 2)
    lef = min(vert, key=lambda l: (l[0] + l[2]) / 2)
    rig = max(vert, key=lambda l: (l[0] + l[2]) / 2)

    def inter(a, b):
        x1a, y1a, x2a, y2a = a
        x1b, y1b, x2b, y2b = b
        d = (x1a - x2a) * (y1b - y2b) - (y1a - y2a) * (x1b - x2b)
        if abs(d) < 1e-6:
            return None
        pa = x1a * y2a - y1a * x2a
        pb = x1b * y2b - y1b * x2b
        return ((pa * (x1b - x2b) - (x1a - x2a) * pb) / d,
                (pa * (y1b - y2b) - (y1a - y2a) * pb) / d)

    pts = [inter(top, lef), inter(top, rig), inter(bot, rig), inter(bot, lef)]
    if any(p is None for p in pts):
        return quad
    pts = np.array(pts, np.float32)
    # reject nonsense: refined corners must stay near the original box
    if (pts[:, 0].min() < -cw * 0.2 or pts[:, 0].max() > cw * 1.2 or
            pts[:, 1].min() < -ch * 0.2 or pts[:, 1].max() > ch * 1.2):
        return quad
    ref = order_quad(pts + np.array([x0, y0], np.float32))
    rw = np.linalg.norm(ref[1] - ref[0])
    rh = np.linalg.norm(ref[3] - ref[0])
    if rh < 6 or rw < 12 or not (ASPECT_MIN <= rw / rh <= ASPECT_MAX):
        return quad
    return ref


# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------

def whole_image_candidate(bgr: np.ndarray) -> Candidate:
    h, w = bgr.shape[:2]
    quad = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    return Candidate(quad, 0.05, "whole-image", {"aspect": round(w / max(1, h), 2)})


def detect_plates(bgr: np.ndarray, max_candidates: int = 5,
                  refine: bool = True) -> list[Candidate]:
    """Return plate candidates, best first. Always includes a whole-image
    fallback so an already-cropped plate still works."""
    cands = detect_by_gradient(bgr) + detect_by_mser(bgr)
    cands.sort(key=lambda c: -c.score)

    merged: list[Candidate] = []
    for c in cands:
        if all(box_iou(c.bbox, m.bbox) < 0.45 for m in merged):
            merged.append(c)
        if len(merged) >= max_candidates:
            break

    if refine:
        gray = to_gray(bgr)
        for c in merged:
            c.quad = refine_quad(gray, c.quad)

    merged.append(whole_image_candidate(bgr))
    return merged

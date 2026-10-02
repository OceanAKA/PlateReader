"""Cutting a plate into individual characters.

Two segmenters that fail in opposite ways, reconciled:

  * connected components - exact when glyphs are separated, but merges
    touching characters and shatters broken strokes;
  * column ink density - the classic projection profile. Immune to touching
    strokes, but blind to a gap that never reaches zero.

Run components first, then use the density profile to split anything too wide
and merge anything too narrow.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .util import smooth1d


@dataclass
class CharBox:
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0


def _components(binary: np.ndarray) -> list[CharBox]:
    h, w = binary.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats((binary > 0).astype(np.uint8), 8)
    boxes: list[CharBox] = []
    for i in range(1, n):
        x, y, bw, bh, area = (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
                              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT],
                              stats[i, cv2.CC_STAT_AREA])
        if bh < h * 0.30 or bh > h * 1.02:        # glyphs fill most of the band
            continue
        if bw < 2 or bw > w * 0.45:
            continue
        if area < 0.10 * bw * bh:                 # hollow noise
            continue
        boxes.append(CharBox(int(x), int(y), int(x + bw), int(y + bh)))
    boxes.sort(key=lambda b: b.x0)
    return boxes


def _merge_vertical_fragments(boxes: list[CharBox]) -> list[CharBox]:
    """Rejoin pieces of one glyph broken apart by thresholding."""
    out: list[CharBox] = []
    for b in boxes:
        if out:
            p = out[-1]
            overlap = min(p.x1, b.x1) - max(p.x0, b.x0)
            if overlap > 0.55 * min(p.w, b.w):
                out[-1] = CharBox(min(p.x0, b.x0), min(p.y0, b.y0),
                                  max(p.x1, b.x1), max(p.y1, b.y1))
                continue
        out.append(b)
    return out


def _valley_cuts(prof: np.ndarray, min_part: int, thr_ratio: float = 0.50) -> list[int]:
    """Centres of the runs where the ink profile drops away.

    Depth is judged relatively, not against a fixed threshold. Inside a K the
    profile thins between the bar and the arms, and on some binarisations that
    dip goes below any absolute cutoff that still catches a real boundary. But
    the true gap between two characters is several times deeper than any
    internal thin spot, so each valley is compared against the deepest one.
    """
    n = prof.size
    if n < 2 * min_part + 2:
        return []
    mean = float(prof.mean())
    if mean <= 1e-6:
        return []
    low = prof < thr_ratio * mean

    runs: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if low[i]:
            j = i
            while j < n and low[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1

    cand = []
    for a, b in runs:
        c = (a + b) // 2
        if min_part <= c <= n - min_part:
            cand.append((c, float(prof[a:b].min())))
    if not cand:
        return []

    deepest = min(d for _, d in cand)
    # comparably deep as the best gap, or unambiguously empty on its own
    limit = max(2.0 * deepest, 0.10 * mean)
    cand = [(c, d) for c, d in cand if d <= limit or d <= 0.22 * mean]
    cand.sort(key=lambda cd: cd[1])

    chosen: list[int] = []
    for c, _ in cand:
        if all(abs(c - k) >= min_part for k in chosen):
            chosen.append(c)
    return sorted(chosen)


def _merge_broken(boxes: list[CharBox], med_h: float) -> list[CharBox]:
    """Rejoin glyphs that thresholding shattered horizontally.

    A blurred K loses the join between its bar and its arms; a W can part at
    the middle vertex. The tell is the gap: spacing between real characters on
    a plate is regular, so a gap far below the median is a break inside one
    glyph rather than a boundary between two.
    """
    if len(boxes) < 3:
        return boxes
    gaps = [boxes[i + 1].x0 - boxes[i].x1 for i in range(len(boxes) - 1)]
    med_gap = float(np.median(gaps))
    if med_gap <= 0.5:
        return boxes

    out = [boxes[0]]
    for b, gap in zip(boxes[1:], gaps):
        p = out[-1]
        combined = b.x1 - p.x0
        if gap < 0.45 * med_gap and combined <= 1.15 * med_h:
            out[-1] = CharBox(min(p.x0, b.x0), min(p.y0, b.y0),
                              max(p.x1, b.x1), max(p.y1, b.y1))
        else:
            out.append(b)
    return out


def _split_wide(binary: np.ndarray, box: CharBox, target_w: float) -> list[CharBox]:
    """Split a merged blob at the deepest interior valley of its ink profile.

    Width alone is not evidence of a merge: W, M and H are legitimately half
    again as wide as a digit. Two touching characters leave a near-empty
    column between them, a single wide glyph does not - so require the valley
    before cutting.
    """
    if box.w <= target_w * 1.95 or box.w < 10:
        return [box]
    # no single glyph is meaningfully wider than it is tall - W and M come
    # close to square, two touching characters do not
    if box.h > 0 and box.w / box.h < 1.05:
        return [box]

    sub = binary[box.y0:box.y1, box.x0:box.x1]
    prof = smooth1d((sub > 0).sum(axis=0).astype(np.float64), 3)
    interior = prof[int(0.22 * box.w):int(0.78 * box.w)]
    if interior.size == 0 or interior.min() > 0.35 * max(prof.mean(), 1e-6):
        return [box]                       # no real gap - one wide character

    # Cut where the ink actually thins out, not at even spacing: a merged KW
    # is not two equal halves, and dividing by the median width would slice
    # the W in two.
    min_part = max(3, int(0.38 * target_w))
    cuts = _valley_cuts(prof, min_part)
    if not cuts:
        n_parts = max(2, min(8, int(round(box.w / max(1.0, target_w)))))
        for k in range(1, n_parts):
            ideal = int(round(box.w * k / n_parts))
            lo = max(2, ideal - int(0.28 * target_w))
            hi = min(box.w - 2, ideal + int(0.28 * target_w))
            if hi > lo:
                cuts.append(lo + int(np.argmin(prof[lo:hi])))
    cuts = sorted(set(cuts))

    parts, prev = [], 0
    for c in cuts + [box.w]:
        if c - prev >= 3:
            parts.append(CharBox(box.x0 + prev, box.y0, box.x0 + c, box.y1))
        prev = c
    return parts or [box]


def _profile_cuts(binary: np.ndarray) -> list[CharBox]:
    """Pure density segmentation - the fallback when components fail."""
    h, w = binary.shape[:2]
    prof = smooth1d((binary > 0).sum(axis=0).astype(np.float64), max(3, int(w * 0.01) | 1))
    thr = max(1.0, 0.055 * h)
    on = prof > thr
    boxes: list[CharBox] = []
    start = None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= 3:
                boxes.append(CharBox(start, 0, i, h))
            start = None
    if start is not None and w - start >= 3:
        boxes.append(CharBox(start, 0, w, h))

    # tighten each slice vertically
    tight: list[CharBox] = []
    for b in boxes:
        sub = binary[:, b.x0:b.x1]
        ys = np.flatnonzero((sub > 0).sum(axis=1) > 0)
        if ys.size == 0:
            continue
        tight.append(CharBox(b.x0, int(ys[0]), b.x1, int(ys[-1]) + 1))
    return tight


def segment(binary: np.ndarray, min_chars: int = 3, max_chars: int = 10) -> list[CharBox]:
    """Return character boxes left-to-right, or [] if nothing plausible."""
    if binary.ndim != 2 or min(binary.shape) < 8:
        return []
    h, w = binary.shape[:2]

    boxes = _merge_vertical_fragments(_components(binary))
    if len(boxes) < min_chars:
        alt = _profile_cuts(binary)
        if len(alt) > len(boxes):
            boxes = alt
    if not boxes:
        return []

    med_h = float(np.median([b.h for b in boxes]))
    boxes = _merge_broken(boxes, med_h)
    med_h = float(np.median([b.h for b in boxes]))
    # measure typical width from full-height glyphs only, so stray fragments
    # do not drag the median down and trigger spurious splits
    full = [b for b in boxes if b.h >= 0.8 * med_h] or boxes
    med_w = float(np.median([b.w for b in full]))
    # When blur has fused the whole plate into one or two blobs, that median
    # is the width of a blob, not of a character, and every split decision
    # downstream inherits the error. Fall back to geometry: a plate glyph is
    # roughly 0.55 as wide as it is tall in every common font.
    if med_w > 1.3 * med_h:
        med_w = 0.55 * med_h

    split: list[CharBox] = []
    for b in boxes:
        split.extend(_split_wide(binary, b, med_w))
    split.sort(key=lambda b: b.x0)

    # a real glyph is close to the median height and not a sliver
    # Characters on a plate share one height. Anything markedly taller is the
    # frame, a bracket or a shadow edge - never a glyph.
    keep = [b for b in split
            if 0.55 * med_h <= b.h <= 1.35 * med_h
            and max(2, 0.16 * med_w) <= b.w <= 2.3 * med_w]
    # Drop border remnants. A horizontal smear drags the plate frame into the
    # crop as a bar that segments as an I. Two tells, neither of which any
    # real glyph shows: a sliver pinned to the edge, or a wide block that
    # fills nearly its whole bounding box (letters leave counters and gaps).
    def fill(b: CharBox) -> float:
        sub = binary[b.y0:b.y1, b.x0:b.x1]
        return float((sub > 0).mean()) if sub.size else 0.0

    edge = [b for b in keep
            if not ((b.x0 <= 1 or b.x1 >= w - 2) and b.w < 0.30 * med_w)
            and not (fill(b) > 0.88 and b.w > 0.50 * med_w)]
    if len(edge) >= min_chars:
        keep = edge

    if len(keep) < min_chars:
        keep = split

    if len(keep) > max_chars:                      # keep the tallest run
        keep = sorted(sorted(keep, key=lambda b: -b.h)[:max_chars], key=lambda b: b.x0)
    return keep


def normalize_char(binary: np.ndarray, box: CharBox, size: tuple[int, int] = (32, 48),
                   pad: int = 2) -> np.ndarray:
    """Crop, tighten and letterbox one glyph into a fixed canvas.

    Aspect ratio is preserved - stretching an I to fill a 32x48 box would make
    it indistinguishable from an H.
    """
    tw, th = size
    x0 = max(0, box.x0)
    x1 = min(binary.shape[1], box.x1)
    y0 = max(0, box.y0)
    y1 = min(binary.shape[0], box.y1)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return np.zeros((th, tw), np.uint8)

    sub = (binary[y0:y1, x0:x1] > 0).astype(np.uint8) * 255
    ys, xs = np.nonzero(sub)
    if ys.size:
        sub = sub[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sh, sw = sub.shape[:2]

    inner_w, inner_h = tw - 2 * pad, th - 2 * pad
    scale = min(inner_w / max(1, sw), inner_h / max(1, sh))
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    sub = cv2.resize(sub, (nw, nh), interpolation=interp)

    canvas = np.zeros((th, tw), np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = sub
    return (canvas > 96).astype(np.uint8) * 255

"""The ensemble.

No single preprocessing chain is right for every photograph, and on a hard
image the choice of threshold changes the answer. So rather than guessing one
chain, run many - each combination of restoration and binarisation - read all
of them, and let the readings vote position by position.

Agreement across independent chains is the confidence signal: characters that
survive every variant are solid, characters that flip between variants are
exactly the ones worth flagging.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import cv2
import numpy as np

from . import binarize as bz
from . import restore as rs
from .detect import Candidate, detect_plates, order_quad
from .formats import FormatSet, PlateFormat
from .ocr import ambiguity_note, default_engines
from .quality import Quality, assess, mark_no_plate_found, verdict_sentence
from .rectify import rectify, trim_border
from .segment import normalize_char, segment
from .util import DIGITS, LETTERS, to_gray

WORK_HEIGHT = 128            # plate height the pipeline works at

LETTER_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "L": "1", "J": "1",
                   "Z": "2", "S": "5", "B": "8", "G": "6", "A": "4", "T": "7"}
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G",
                   "4": "A", "7": "T"}


@dataclass
class Hypothesis:
    text: str
    confs: list[float]
    variant: str
    engine: str
    weight: float = 1.0

    @property
    def mean_conf(self) -> float:
        return float(np.mean(self.confs)) if self.confs else 0.4


@dataclass
class PlateResult:
    text: str
    confidence: float
    agreement: float
    per_char: list[dict]
    alternatives: list[tuple[str, float]]
    quality: Quality
    candidate: Candidate
    n_hypotheses: int
    hypotheses: list[Hypothesis] = field(default_factory=list)
    stages: dict = field(default_factory=dict)
    rotation: int = 0            # quarter turns applied before this read, in degrees
    rank_score: float = 0.0
    plate_format: str | None = None      # name of the grammar it satisfies
    format_region: str | None = None
    raw_text: str = ""                   # the reading before any grammar applied

    @property
    def trustworthy(self) -> bool:
        return self.quality.readable and self.confidence >= 0.55 and self.agreement >= 0.5


# --------------------------------------------------------------------------
# variant generation
# --------------------------------------------------------------------------

def build_variants(plate: np.ndarray, aggressive: bool = False) -> list[tuple[str, np.ndarray, float]]:
    """(name, image, trust weight) for each restoration to try.

    Deconvolution outputs are down-weighted: they recover real detail, but
    they also manufacture ringing that can read as a stroke.
    """
    out: list[tuple[str, np.ndarray, float]] = [
        ("raw", plate, 1.0),
        ("clahe", rs.clahe(plate), 1.0),
        ("illum", rs.clahe(rs.normalize_illumination(plate)), 1.0),
        ("unsharp", rs.unsharp(plate, 1.2, 1.4), 0.95),
    ]
    if aggressive:
        out.append(("unsharp-hard", rs.unsharp(rs.clahe(plate), 2.0, 2.0), 0.85))
        try:
            out.append(("denoise", rs.clahe(rs.denoise(plate, 6.0)), 0.95))
        except Exception:
            pass

    be = rs.estimate_motion_blur(plate)
    if be.trustworthy:
        psf = rs.motion_psf(be.length, be.angle)
        out.append((f"wiener-motion-L{be.length:.0f}", rs.wiener_deconv(plate, psf, 0.010), 0.85))
        if aggressive:
            out.append((f"wiener-motion-soft", rs.wiener_deconv(plate, psf, 0.035), 0.85))
            out.append((f"rl-motion", rs.richardson_lucy(plate, psf, 18), 0.80))

    if aggressive or be.lap_var < 90:
        for sigma in (1.0, 2.0):
            g = rs.gaussian_psf(sigma)
            out.append((f"wiener-gauss{sigma}", rs.wiener_deconv(plate, g, 0.012), 0.85))
        if aggressive:
            out.append(("rl-gauss", rs.richardson_lucy(plate, rs.gaussian_psf(1.6), 22), 0.80))

    if aggressive:
        # Blind search: do not trust the cepstrum to have found the blur, try
        # a grid of PSFs and keep the ones whose output actually looks like
        # text. This is what rescues heavy motion blur, where the single
        # estimate above is least reliable.
        for score, length, angle, img in rs.search_motion_psf(plate, top_k=3):
            out.append((f"search-L{length:.0f}@{angle:.0f}", img, 0.80))
            if score > 0.45:
                # promising direction - also try a gentler regularisation,
                # which keeps more real detail when the PSF is close to right
                out.append((f"search-soft-L{length:.0f}@{angle:.0f}",
                            rs.wiener_deconv(plate, rs.motion_psf(length, angle), 0.035),
                            0.78))
        # Defocus needs its own kernels - a line PSF will not undo a disc - and
        # its own range: the fixed sigma-1/sigma-2 variants above only cover
        # mild softness, while a badly focused shot can be several pixels of
        # blur radius and still have large, well-lit characters worth
        # recovering.
        for score, label, img in rs.search_defocus_psf(plate, top_k=2):
            out.append((f"search-{label}", img, 0.80))
    return out


def _binarizations(gray: np.ndarray, aggressive: bool) -> list[tuple[str, np.ndarray]]:
    methods = ["otsu", "sauvola", "adaptive"]
    if aggressive:
        methods.append("niblack")
    out = []
    for m in methods:
        try:
            out.append((m, bz.clean(bz.ALL_METHODS[m](gray))))
        except Exception:
            continue
    return out


# --------------------------------------------------------------------------
# voting
# --------------------------------------------------------------------------

def _beam_candidates(per_pos: list[list[tuple[str, float]]], limit: int = 12,
                     beam: int = 200) -> list[tuple[str, float]]:
    """Top whole-string readings, from the per-position distributions."""
    paths: list[tuple[str, float]] = [("", 1.0)]
    for pos in per_pos:
        nxt = []
        for text, p in paths:
            for ch, q in pos[:3]:
                nxt.append((text + ch, p * max(q, 1e-4)))
        nxt.sort(key=lambda kv: -kv[1])
        paths = nxt[:beam]
    return paths[:limit]


def _coerce(text: str, pattern: re.Pattern) -> str | None:
    """Nudge letters/digits into the shape a known plate format demands."""
    if pattern.fullmatch(text):
        return text
    for maps in (LETTER_TO_DIGIT, DIGIT_TO_LETTER):
        swapped = "".join(maps.get(c, c) for c in text)
        if pattern.fullmatch(swapped):
            return swapped
    return None


def vote(hyps: list[Hypothesis], pattern: re.Pattern | None = None,
         formats: FormatSet | None = None):
    if not hyps:
        return "", 0.0, 0.0, [], [], None, ""

    by_len: dict[int, list[Hypothesis]] = defaultdict(list)
    for h in hyps:
        by_len[len(h.text)].append(h)

    def group_weight(group: list[Hypothesis]) -> float:
        return sum(h.weight * h.mean_conf for h in group)

    if pattern is not None:
        matching = {L: g for L, g in by_len.items()
                    if any(_coerce(h.text, pattern) for h in g)}
        if matching:
            by_len = matching

    if formats is not None:
        # A reading whose length no real plate format has is usually an
        # over- or under-segmentation, so prefer lengths a grammar can accept.
        legal = {L: g for L, g in by_len.items() if L in formats.lengths}
        if legal:
            by_len = legal

    best_len = max(by_len, key=lambda L: group_weight(by_len[L]))
    group = by_len[best_len]

    per_pos: list[list[tuple[str, float]]] = []
    for i in range(best_len):
        tally: dict[str, float] = defaultdict(float)
        for h in group:
            conf = h.confs[i] if i < len(h.confs) else 0.5
            tally[h.text[i]] += h.weight * max(0.05, conf)
        total = sum(tally.values()) or 1.0
        ranked = sorted(((c, v / total) for c, v in tally.items()), key=lambda kv: -kv[1])
        per_pos.append(ranked)

    text = "".join(p[0][0] for p in per_pos)
    vote_share = float(np.mean([p[0][1] for p in per_pos])) if per_pos else 0.0
    agreement = sum(h.weight for h in group if h.text == text) / max(
        1e-9, sum(h.weight for h in hyps))

    # Vote share alone is not confidence: if every variant is degraded the
    # same way they will agree unanimously on the same wrong character. Fold
    # in how well the glyphs actually matched a real letterform, which is an
    # absolute measure rather than a relative one.
    winners = [h for h in group if h.text == text] or group
    wsum = sum(h.weight for h in winners) or 1e-9
    ocr_fit = sum(h.weight * h.mean_conf for h in winners) / wsum
    confidence = float(np.clip(vote_share * ocr_fit, 0.0, 1.0))

    candidates = _beam_candidates(per_pos)
    if pattern is not None:
        fixed = []
        for cand, p in candidates:
            c = _coerce(cand, pattern)
            if c:
                fixed.append((c, p))
        if fixed:
            candidates = fixed
            if text != candidates[0][0]:
                text = candidates[0][0]

    free_text = text          # what the vote said before any grammar applied
    matched: PlateFormat | None = None
    if formats is not None:
        got = formats.constrain(per_pos)
        if got is not None:
            ctext, _rank, mean_share, matched = got
            if ctext != text:
                # The grammar overrode at least one position, so confidence
                # comes from the constrained read. Agreement is deliberately
                # NOT recomputed: it measures whether the independent chains
                # concur, and the corrected string is synthesised from
                # per-position picks, so no chain need contain it verbatim.
                # Rescoring it here drove agreement to zero on exactly the
                # reads the grammar had just improved, which in turn made the
                # result look untrustworthy and triggered the rotation retry.
                text = ctext
                confidence = float(np.clip(mean_share * ocr_fit, 0.0, 1.0))

    return text, confidence, float(agreement), per_pos, candidates, matched, free_text


# --------------------------------------------------------------------------
# reading one plate
# --------------------------------------------------------------------------

def read_candidate(bgr: np.ndarray, cand: Candidate, engines: dict,
                   aggressive: bool = False, pattern: re.Pattern | None = None,
                   collect_stages: bool = False,
                   formats: FormatSet | None = None) -> PlateResult | None:
    gray = to_gray(bgr)

    # native crop, used only to judge how much information the photo holds
    x0, y0, x1, y1 = [int(round(v)) for v in cand.bbox]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
    if x1 - x0 < 16 or y1 - y0 < 8:
        return None
    native = gray[y0:y1, x0:x1]

    def build_plate(pad_x: float) -> np.ndarray | None:
        p = rectify(gray, cand.quad, target_h=WORK_HEIGHT, pad_x=pad_x)
        p, _ = trim_border(p)
        if p.size == 0 or p.shape[0] < 24:
            return None
        if p.shape[0] != WORK_HEIGHT:
            s = WORK_HEIGHT / float(p.shape[0])
            p = cv2.resize(p, (max(16, int(p.shape[1] * s)), WORK_HEIGHT),
                           interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        return p

    plate = build_plate(0.13)
    if plate is None:
        return None

    # Character height, measured in the ORIGINAL photograph's pixels, from the
    # characters actually segmented. Warping at the plate's own native scale
    # makes one pixel here one pixel there, so the box heights are directly
    # comparable to the resolution thresholds in quality.py.
    def native_char_height() -> float | None:
        q = order_quad(cand.quad)
        qh = max(float(np.linalg.norm(q[3] - q[0])),
                 float(np.linalg.norm(q[2] - q[1])))
        target = qh * 1.32                      # pad_quad grows height by pad_y
        if target < 12:
            return None
        scale = 1.0
        if target > 420:                        # keep the measuring warp cheap
            scale = target / 420.0
            target = 420.0
        try:
            meas = rectify(gray, cand.quad, target_h=int(round(target)), pad_x=0.13)
            meas, _ = trim_border(meas)
            boxes = segment(bz.clean(bz.otsu(meas)))
        except Exception:
            return None
        if len(boxes) < 3:
            return None
        return float(np.median([b.h for b in boxes])) * scale

    measured_char_h = native_char_height()
    quality = assess(native, char_height_px=measured_char_h)
    if measured_char_h is None:
        # Could not find three character-shaped things at native scale. The
        # fallback inside assess() would now assume this crop is a plate and
        # report a character height of 55% of it - which for a whole-image
        # candidate means "600px characters, excellent quality" about a
        # photograph of a car. Refuse to make the claim instead.
        mark_no_plate_found(
            quality,
            "No row of character-shaped regions was found at the original "
            "resolution, so there is nothing here to vouch for.")

    # If ink runs right up to the edge of the crop, the detector probably cut
    # a character in half. Nothing downstream can notice a glyph that is not
    # in the image, so the read can be unanimously confident and still be
    # missing its first or last character.
    def is_clipped(p: np.ndarray) -> bool:
        _, eb = cv2.threshold(p, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if eb.mean() > 127:
            eb = 255 - eb
        cols = (eb > 0).mean(axis=0)
        return bool(cols[:2].max() > 0.15 or cols[-2:].max() > 0.15)

    def char_count(p: np.ndarray) -> int:
        try:
            return len(segment(bz.clean(bz.otsu(p))))
        except Exception:
            return 0

    clipped = is_clipped(plate)
    # Two different ways the crop can be too tight, and the finished reading
    # shows neither: ink running off the edge (a glyph cut in half), or a glyph
    # missing outright - which happens at the compressed far end of a steeply
    # angled plate and leaves a perfectly self-consistent short read. Widening
    # cannot lose anything, since trim_border crops straight back to the ink,
    # so take the wider crop whenever it actually finds more characters.
    if clipped or char_count(plate) < 6:
        wider = build_plate(0.34)
        if wider is not None and char_count(wider) > char_count(plate):
            plate = wider
            clipped = is_clipped(plate)

    stages: dict = {"native": native, "rectified": plate} if collect_stages else {}
    if clipped:
        quality.notes.append(
            "Ink reaches the edge of the plate crop, so a leading or trailing "
            "character may be cut off and simply absent from this read.")

    hyps: list[Hypothesis] = []
    template = engines.get("template")
    tess = engines.get("tesseract")
    crnn = engines.get("crnn")

    for vname, vimg, vweight in build_variants(plate, aggressive):
        # The sequence model works on the greyscale strip and never needs a
        # threshold or a character box, so it runs once per restoration rather
        # than once per binarisation - and it is the only engine here that can
        # read a plate whose characters have merged into each other.
        if crnn is not None:
            try:
                for text, confs in crnn.read_image(vimg):
                    if 3 <= len(text) <= 10:
                        hyps.append(Hypothesis(text, confs, f"{vname}|gray",
                                               "crnn", vweight * 1.20))
            except Exception:
                pass

        for bname, binary in _binarizations(vimg, aggressive):
            boxes = segment(binary)
            if len(boxes) < 3:
                continue
            if collect_stages and f"{vname}|{bname}" not in stages:
                stages[f"bin:{vname}|{bname}"] = binary

            if template is not None:
                chars = [normalize_char(binary, b) for b in boxes]
                text, confs, _ = template.read(chars)
                if 3 <= len(text) <= 10:
                    hyps.append(Hypothesis(text, confs, f"{vname}|{bname}",
                                           "template", vweight))
            if tess is not None:
                for text, confs in tess.read_image(255 - binary):
                    if not confs:
                        confs = [0.5] * len(text)
                    hyps.append(Hypothesis(text, confs, f"{vname}|{bname}",
                                           "tesseract", vweight * 1.15))

    if not hyps:
        return None

    text, conf, agreement, per_pos, alts, matched, free_text = vote(
        hyps, pattern, formats)

    # A reading made almost entirely of one confusable glyph is what a region
    # containing no text looks like after thresholding: vertical edges become
    # a row of I and 1. Combined with the chains failing to agree, that is not
    # a hard-to-read plate, it is not a plate - so withdraw the quality claim
    # rather than reporting a healthy score for whatever got cropped.
    if len(text) >= 4:
        distinct = len(set(text))
        strokey = sum(ch in "1IJLT" for ch in text) / len(text)
        if (distinct <= 2 and agreement < 0.5) or (strokey > 0.75 and agreement < 0.35):
            mark_no_plate_found(
                quality,
                "The characters found here are nearly all the same stroke-like "
                "glyph and the processing chains did not agree, which is what a "
                "region with no text in it looks like. Treat this as 'no plate "
                "found', not as a difficult plate.")
    if clipped:
        conf *= 0.85
    per_char = []
    for i, ch in enumerate(text):
        dist = per_pos[i] if i < len(per_pos) else [(ch, conf)]
        per_char.append({
            "char": ch,
            "confidence": round(float(dist[0][1]), 3),
            "runner_up": dist[1][0] if len(dist) > 1 else None,
            "runner_up_confidence": round(float(dist[1][1]), 3) if len(dist) > 1 else None,
            "confusable_with": ambiguity_note(ch),
        })

    return PlateResult(
        text=text, confidence=conf, agreement=agreement, per_char=per_char,
        alternatives=[(t, round(float(p), 4)) for t, p in alts],
        quality=quality, candidate=cand, n_hypotheses=len(hyps),
        hypotheses=hyps, stages=stages,
        plate_format=(matched.name if matched else None),
        format_region=(matched.region if matched else None),
        raw_text=free_text,
    )


def _read_pass(bgr: np.ndarray, engines: dict, pat, aggressive: bool,
               max_candidates: int, collect_stages: bool,
               rotation: int, formats: FormatSet | None) -> list[PlateResult]:
    cands = detect_plates(bgr, max_candidates=max_candidates)


    results: list[PlateResult] = []
    for c in cands:
        try:
            r = read_candidate(bgr, c, engines, aggressive, pat, collect_stages,
                               formats)
        except Exception:
            continue
        if r is not None and len(r.text) >= 3:
            r.rotation = rotation
            results.append(r)

    texts = [r.text for r in results]
    for r in results:
        fmt = 0.0
        if any(ch in DIGITS for ch in r.text) and any(ch in LETTERS for ch in r.text):
            fmt = 0.08                       # mixed alphanumeric looks like a plate
        # Reward reading more of the plate. Detectors clip a leading character
        # far more often than they invent one, and it is always easier to be
        # confident about less - so confidence alone would favour the crop.
        fmt += 0.045 * min(len(r.text), 8)
        # A read that strictly contains another read is the same plate, seen
        # more completely.
        if any(t != r.text and len(t) < len(r.text) and t in r.text for t in texts):
            fmt += 0.10
        base = (0.40 * r.confidence + 0.22 * r.agreement +
                0.13 * r.candidate.score + fmt)
        # Image quality multiplies rather than adds: a read off an unreadable
        # crop should not be able to win on confidence, which is precisely
        # what a tiny or blank crop produces - it is easy to be sure about
        # nothing.
        r.rank_score = base * (0.55 + 0.45 * r.quality.score)

    results.sort(key=lambda r: -r.rank_score)
    return results


def read_image(bgr: np.ndarray, aggressive: bool = False, pattern: str | None = None,
               max_candidates: int = 4, engines: dict | None = None,
               collect_stages: bool = False,
               try_rotations: bool = True,
               formats: FormatSet | None = None,
               descreen: bool | None = None) -> list[PlateResult]:
    """Detect and read every plate-like region, best result first.

    The geometric correction inside rectify() handles tilt up to about 25
    degrees. A photograph taken in the other orientation is a different
    problem - the plate is a quarter turn out and no amount of fine angle
    search will find it - so if the upright pass produces nothing worth
    trusting, the whole image is retried at each quarter turn.
    """
    engines = engines if engines is not None else default_engines()
    pat = re.compile(pattern.upper()) if pattern else None

    def settled(found: list[PlateResult]) -> bool:
        # A short read is never reason enough to stop looking: a plate stood on
        # end reads as two or three vertical strokes, and that answer can look
        # perfectly self-consistent while being the wrong orientation entirely.
        return bool(found) and found[0].trustworthy and len(found[0].text) >= 5

    if descreen is True:
        bgr = rs.suppress_periodic_bgr(bgr)

    best = _read_pass(bgr, engines, pat, aggressive, max_candidates,
                      collect_stages, 0, formats)
    if settled(best):
        return best

    # A photograph of a screen carries the display's pixel grid, and that
    # regular texture is exactly the signal the gradient detector keys on, at
    # the scale it keys on it - so it has to come off before detection, not
    # after, or the plate is never found at all.
    #
    # Deciding up front whether an image needs it does not work: every measure
    # tried (peak prominence, peak count) put heavy motion blur and sensor
    # noise in the same range as real moire, so an automatic threshold either
    # missed screens or descreened clean photographs. Retrying only when the
    # first pass failed to settle costs nothing on an image that read fine and
    # needs no threshold at all.
    if descreen is not False and not settled(best):
        cleaned = rs.suppress_periodic_bgr(bgr)
        alt = _read_pass(cleaned, engines, pat, aggressive, max_candidates,
                         collect_stages, 0, formats)
        if alt and (not best or alt[0].rank_score > best[0].rank_score):
            best = alt
            bgr = cleaned          # any rotation retry below works on this too
        if settled(best):
            return best

    # Deblur the whole frame and look again. This is separate from the
    # per-plate deconvolution: that one only runs after the plate has been
    # found and straightened, and on a blurred angled shot neither of those
    # happens. Restoring the frame first gives the detector edges to find and
    # the rectifier lines to square up.
    if not settled(best):
        try:
            cleaned, how = rs.deblur_bgr(bgr)
        except Exception:
            cleaned, how = None, "failed"
        if cleaned is not None and how not in ("none", "too-small", "failed"):
            alt = _read_pass(cleaned, engines, pat, aggressive, max_candidates,
                             collect_stages, 0, formats)
            if alt and (not best or alt[0].rank_score > best[0].rank_score):
                best = alt
                bgr = cleaned
            if settled(best):
                return best

    if not try_rotations:
        return best

    for k in (1, 2, 3):
        turned = np.ascontiguousarray(np.rot90(bgr, k))
        alt = _read_pass(turned, engines, pat, aggressive, max_candidates,
                         collect_stages, k * 90, formats)
        # Rotation is a last resort, so a turned pass has to win clearly, not
        # by a rounding error - otherwise noise read out of a sideways crop
        # displaces a sound upright reading.
        if alt and (not best or alt[0].rank_score > best[0].rank_score * 1.15):
            best = alt
        if settled(best):
            break
    return best


def summarize(r: PlateResult) -> str:
    lines = [
        f"  plate      : {r.text}",
        f"  confidence : {r.confidence:.0%}  (agreement {r.agreement:.0%} across "
        f"{r.n_hypotheses} independent reads)",
        f"  image      : {r.quality.verdict.upper()} - {verdict_sentence(r.quality)}",
    ]
    return "\n".join(lines)

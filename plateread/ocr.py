"""Character recognition.

The default engine is self-contained: it renders A-Z and 0-9 from the fonts
already on the machine and matches segmented glyphs against them on four
features -

  * normalised cross-correlation of the bitmap (overall shape)
  * zoning density - ink fraction in each cell of a 4x6 grid, which is the
    "pixel density" signature of a glyph and survives blur well
  * row and column ink profiles (where the mass sits)
  * topological hole count (0 for C, 1 for D, 2 for B) - very discriminative
    and cheap, though blur can fill holes, so it is weighted softly

Tesseract is used instead when it is installed, with this engine still run
alongside so the vote has two independent opinions.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .util import ALPHABET

CHAR_SIZE = (32, 48)          # (w, h)

# Pairs that are genuinely hard to separate in a degraded image. Reported to
# the user rather than silently resolved.
AMBIGUITY_GROUPS = [
    set("0OQD"), set("1IJlT"), set("8B"), set("5S"), set("2Z"),
    set("6G"), set("4A"), set("7T"), set("UV"), set("MW"), set("EF"), set("PR"),
]

WINDOWS_FONT_CANDIDATES = [
    "arialbd.ttf", "arial.ttf", "consolab.ttf", "consola.ttf",
    "courbd.ttf", "verdanab.ttf", "tahomabd.ttf", "seguisb.ttf",
    "calibrib.ttf", "trebucbd.ttf",
]
UNIX_FONT_DIRS = [
    "/usr/share/fonts", "/usr/local/share/fonts",
    os.path.expanduser("~/.fonts"), "/Library/Fonts", "/System/Library/Fonts",
]
UNIX_FONT_NAMES = [
    "DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "LiberationSans-Bold.ttf",
    "LiberationSans-Regular.ttf", "Arial Bold.ttf", "Arial.ttf", "Helvetica.ttc",
]


def find_fonts(limit: int = 4, extra: list[str] | None = None) -> list[str]:
    """Locate a few sans/mono faces to build templates from."""
    found: list[str] = []
    for p in (extra or []):
        if Path(p).is_file():
            found.append(str(p))

    win = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    if win.is_dir():
        for name in WINDOWS_FONT_CANDIDATES:
            p = win / name
            if p.is_file():
                found.append(str(p))

    for d in UNIX_FONT_DIRS:
        base = Path(d)
        if not base.is_dir():
            continue
        for name in UNIX_FONT_NAMES:
            hits = list(base.rglob(name))
            if hits:
                found.append(str(hits[0]))

    seen, out = set(), []
    for f in found:
        if f.lower() not in seen:
            seen.add(f.lower())
            out.append(f)
        if len(out) >= limit:
            break
    return out


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------

def hole_count(char_img: np.ndarray) -> int:
    """Enclosed background regions: 0 for C, 1 for D/O, 2 for B/8."""
    ink = (char_img > 0).astype(np.uint8)
    bg = 1 - ink
    h, w = bg.shape
    flood = bg.copy()
    mask = np.zeros((h + 2, w + 2), np.uint8)
    for x in range(w):
        for y in (0, h - 1):
            if flood[y, x]:
                cv2.floodFill(flood, mask, (x, y), 0)
    for y in range(h):
        for x in (0, w - 1):
            if flood[y, x]:
                cv2.floodFill(flood, mask, (x, y), 0)
    n, _, stats, _ = cv2.connectedComponentsWithStats(flood, 4)
    return sum(1 for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 6)


def zoning(char_img: np.ndarray, cols: int = 4, rows: int = 6) -> np.ndarray:
    """Ink fraction per grid cell - the density fingerprint of the glyph."""
    ink = (char_img > 0).astype(np.float64)
    h, w = ink.shape
    ys = np.linspace(0, h, rows + 1).astype(int)
    xs = np.linspace(0, w, cols + 1).astype(int)
    out = np.empty(rows * cols, np.float64)
    k = 0
    for r in range(rows):
        for c in range(cols):
            cell = ink[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
            out[k] = cell.mean() if cell.size else 0.0
            k += 1
    return out


def profiles(char_img: np.ndarray) -> np.ndarray:
    ink = (char_img > 0).astype(np.float64)
    col = ink.sum(axis=0)
    row = ink.sum(axis=1)
    return np.concatenate([col / (ink.shape[0] or 1), row / (ink.shape[1] or 1)])


@dataclass
class Features:
    bitmap: np.ndarray       # float, zero-mean unit-norm, for NCC
    zone: np.ndarray
    prof: np.ndarray
    holes: int


def extract(char_img: np.ndarray) -> Features:
    b = (char_img > 0).astype(np.float64)
    b = b - b.mean()
    n = np.linalg.norm(b)
    b = b / n if n > 1e-9 else b
    return Features(b, zoning(char_img), profiles(char_img), hole_count(char_img))


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


def compare(a: Features, b: Features) -> float:
    ncc = float(np.clip((a.bitmap * b.bitmap).sum(), -1.0, 1.0))
    zone = _cos(a.zone, b.zone)
    prof = _cos(a.prof, b.prof)
    holes = 1.0 - min(abs(a.holes - b.holes), 2) / 2.0
    return 0.45 * ncc + 0.25 * zone + 0.15 * prof + 0.15 * holes


# --------------------------------------------------------------------------
# template engine
# --------------------------------------------------------------------------

def _render_glyph(font_path: str, ch: str, px: int = 96) -> np.ndarray | None:
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype(font_path, px)
    except Exception:
        return None
    img = Image.new("L", (px * 2, int(px * 1.8)), 0)
    d = ImageDraw.Draw(img)
    d.text((px // 2, px // 4), ch, fill=255, font=font)
    a = np.array(img, np.uint8)
    ys, xs = np.nonzero(a > 96)
    if ys.size == 0:
        return None
    return (a[ys.min():ys.max() + 1, xs.min():xs.max() + 1] > 96).astype(np.uint8) * 255


def _fit_canvas(glyph: np.ndarray, size=CHAR_SIZE, pad: int = 2) -> np.ndarray:
    tw, th = size
    sh, sw = glyph.shape[:2]
    scale = min((tw - 2 * pad) / max(1, sw), (th - 2 * pad) / max(1, sh))
    nw, nh = max(1, int(round(sw * scale))), max(1, int(round(sh * scale)))
    g = cv2.resize(glyph, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw), np.uint8)
    ox, oy = (tw - nw) // 2, (th - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = g
    return (canvas > 96).astype(np.uint8) * 255


@lru_cache(maxsize=4)
def _build_templates(font_key: str) -> dict:
    fonts = [f for f in font_key.split("|") if f]
    templates: dict[str, list[Features]] = {c: [] for c in ALPHABET}
    for fp in fonts:
        for ch in ALPHABET:
            g = _render_glyph(fp, ch)
            if g is None:
                continue
            canvas = _fit_canvas(g)
            templates[ch].append(extract(canvas))
            # Stroke weight is not a fixed property of a glyph once an image
            # has been blurred and thresholded: heavy blur fattens strokes
            # until a K fills in, light exposure thins them until it breaks up.
            # Carrying dilated and eroded copies lets a degraded glyph match
            # the right letter at the wrong weight, rather than the wrong
            # letter at the right weight.
            templates[ch].append(extract(cv2.dilate(canvas, np.ones((2, 2), np.uint8))))
            templates[ch].append(extract(cv2.dilate(canvas, np.ones((3, 3), np.uint8))))
            thin = cv2.erode(canvas, np.ones((2, 2), np.uint8))
            if thin.any():
                templates[ch].append(extract(thin))
    return {k: v for k, v in templates.items() if v}


class TemplateOCR:
    """Font-template classifier. No model download, no network, no Tesseract."""

    name = "template"

    def __init__(self, fonts: list[str] | None = None):
        self.fonts = fonts or find_fonts()
        if not self.fonts:
            raise RuntimeError("no usable TrueType fonts found for template OCR")
        self.templates = _build_templates("|".join(self.fonts))

    def classify(self, char_img: np.ndarray, top_k: int = 3) -> list[tuple[str, float]]:
        f = extract(char_img)
        scored = []
        for ch, tmpls in self.templates.items():
            scored.append((ch, max(compare(f, t) for t in tmpls)))
        scored.sort(key=lambda kv: -kv[1])
        return scored[:top_k]

    def read(self, char_imgs: list[np.ndarray]) -> tuple[str, list[float], list[list[tuple[str, float]]]]:
        text, confs, alts = "", [], []
        for ci in char_imgs:
            ranked = self.classify(ci, top_k=3)
            if not ranked:
                continue
            best, score = ranked[0]
            runner = ranked[1][1] if len(ranked) > 1 else 0.0
            # confidence blends absolute fit with the margin over the runner-up
            conf = float(np.clip(0.65 * max(0.0, score) + 0.35 * max(0.0, score - runner) * 3.0, 0.0, 1.0))
            text += best
            confs.append(conf)
            alts.append(ranked)
        return text, confs, alts


# --------------------------------------------------------------------------
# optional tesseract backend
# --------------------------------------------------------------------------

def tesseract_available() -> bool:
    try:
        import pytesseract  # noqa: F401
    except Exception:
        return False
    if shutil.which("tesseract"):
        return True
    for p in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
              r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if Path(p).is_file():
            try:
                import pytesseract
                pytesseract.pytesseract.tesseract_cmd = p
                return True
            except Exception:
                return False
    return False


class TesseractOCR:
    name = "tesseract"
    WHITELIST = ALPHABET

    def __init__(self, psm_modes=(7, 8, 13)):
        import pytesseract
        self._pt = pytesseract
        self.psm_modes = psm_modes

    def read_image(self, plate_img: np.ndarray) -> list[tuple[str, list[float]]]:
        out = []
        for psm in self.psm_modes:
            cfg = (f"--oem 3 --psm {psm} "
                   f"-c tessedit_char_whitelist={self.WHITELIST} "
                   f"-c classify_bln_numeric_mode=0")
            try:
                data = self._pt.image_to_data(plate_img, config=cfg,
                                              output_type=self._pt.Output.DICT)
            except Exception:
                continue
            text, confs = "", []
            for word, conf in zip(data.get("text", []), data.get("conf", [])):
                word = "".join(c for c in str(word).upper() if c in self.WHITELIST)
                try:
                    c = float(conf)
                except (TypeError, ValueError):
                    c = -1.0
                if word and c >= 0:
                    text += word
                    confs.extend([c / 100.0] * len(word))
            if 3 <= len(text) <= 10:
                out.append((text, confs))
        return out


DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "crnn.pt"


def default_engines(fonts: list[str] | None = None,
                    model_path: str | None = None,
                    use_model: bool = True):
    """Template engine always; Tesseract and the trained model when available.

    All three are kept even when the model is present. They fail in different
    ways - the template matcher on unusual fonts, the model on anything far
    from its training distribution - and a vote between disagreeing engines is
    worth more than whichever one is best on average.
    """
    engines = {"template": TemplateOCR(fonts)}
    if tesseract_available():
        try:
            engines["tesseract"] = TesseractOCR()
        except Exception:
            pass
    if use_model:
        path = Path(model_path) if model_path else DEFAULT_MODEL
        if path.is_file():
            try:
                from .model import CRNNOCR
                engines["crnn"] = CRNNOCR(path)
            except Exception:
                pass
    return engines


def ambiguity_note(ch: str) -> str:
    for grp in AMBIGUITY_GROUPS:
        if ch in grp:
            others = sorted(grp - {ch})
            if others:
                return "/".join(others)
    return ""

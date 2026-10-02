"""Training data for the sequence recogniser.

The model reads a *rectified plate crop* - exactly what the existing pipeline
already produces after detection and unwarping - and emits the whole string.
So the training images have to look like that stage's output, including its
imperfections: crops that are slightly too tight or too loose, residual tilt
the rotation search did not quite remove, and a frame edge left in shot.

Two sources, mixed:

  * synthetic, generated on demand, unlimited and perfectly labelled;
  * real photographs you supply with their true plate text.

Synthetic data alone gets a model that works on synthetic data. The
augmentation here is deliberately harsher than reality to narrow that gap, but
it does not close it - a handful of real labelled crops is worth a great many
generated ones, which is why `load_real` exists and why training mixes them.
"""
from __future__ import annotations

import csv
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .formats import BUILTIN_FORMATS
from .ocr import find_fonts
from .util import ALPHABET, clip8, imread_any

# CTC reserves index 0 for the blank label.
BLANK = 0
CHARS = ALPHABET
CHAR_TO_IDX = {c: i + 1 for i, c in enumerate(CHARS)}
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1

IMG_H, IMG_W = 48, 160
MAX_LEN = 10


def encode(text: str) -> list[int]:
    return [CHAR_TO_IDX[c] for c in text.upper() if c in CHAR_TO_IDX]


def decode_greedy(indices) -> str:
    """Collapse a CTC path: drop repeats, then drop blanks."""
    out: list[str] = []
    prev = -1
    for i in indices:
        i = int(i)
        if i != prev and i != BLANK:
            out.append(IDX_TO_CHAR.get(i, ""))
        prev = i
    return "".join(out)


def prepare(gray: np.ndarray) -> np.ndarray:
    """Normalise any plate crop into the model's input tensor layout."""
    g = gray if gray.ndim == 2 else cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    x = g.astype(np.float32) / 255.0
    return (x - 0.5) / 0.5


# --------------------------------------------------------------------------
# synthetic generation
# --------------------------------------------------------------------------

_MASK_POOL = [f.mask for f in BUILTIN_FORMATS if f.region != "generic"]


def random_plate_text(rng: random.Random) -> str:
    """Mostly real-format strings, sometimes free ones.

    Training only on valid formats would teach the model the grammar as well
    as the glyphs, and it would then quietly "correct" an unusual plate into a
    common shape. Mixing in free strings keeps it reading what is there.
    """
    if rng.random() < 0.65 and _MASK_POOL:
        mask = rng.choice(_MASK_POOL)
    else:
        mask = "".join(rng.choice("LD") for _ in range(rng.randint(4, 8)))
    out = []
    for m in mask[:MAX_LEN]:
        if m == "L":
            out.append(rng.choice(string.ascii_uppercase))
        elif m == "D":
            out.append(rng.choice(string.digits))
        else:
            out.append(rng.choice(ALPHABET))
    return "".join(out)


@dataclass
class SynthConfig:
    hard: float = 1.0          # global multiplier on degradation severity


class PlateSynth:
    """Draws plate crops that look like the pipeline's rectified output."""

    def __init__(self, fonts: list[str] | None = None, seed: int | None = None,
                 config: SynthConfig | None = None):
        self.fonts = fonts or find_fonts(limit=6)
        if not self.fonts:
            raise RuntimeError("no TrueType fonts available to synthesise plates")
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.cfg = config or SynthConfig()

    # -- drawing ----------------------------------------------------------
    def _render(self, text: str) -> np.ndarray:
        from PIL import Image, ImageDraw, ImageFont
        rng = self.rng

        h = rng.randint(56, 110)
        w = int(h * rng.uniform(2.2, 4.6))
        dark_on_light = rng.random() < 0.82
        if dark_on_light:
            bg = rng.randint(185, 255)
            fg = rng.randint(0, 70)
        else:
            bg = rng.randint(10, 70)
            fg = rng.randint(190, 255)

        img = Image.new("L", (w, h), bg)
        d = ImageDraw.Draw(img)

        # optional frame and jurisdiction banner - both are things the crop
        # often still contains after trimming
        if rng.random() < 0.55:
            d.rectangle([0, 0, w - 1, h - 1], outline=int(fg * 0.6 + bg * 0.4),
                        width=max(1, h // 40))
        band_top = 0
        if rng.random() < 0.30:
            bh = int(h * rng.uniform(0.16, 0.24))
            bfont = ImageFont.truetype(rng.choice(self.fonts), max(7, int(bh * 0.8)))
            label = rng.choice(["TEXAS", "CALIFORNIA", "OHIO", "NEW YORK", "GB", "D"])
            bb = d.textbbox((0, 0), label, font=bfont)
            d.text(((w - (bb[2] - bb[0])) // 2 - bb[0], 2), label, fill=fg, font=bfont)
            band_top = bh

        avail = h - band_top
        size = int(avail * rng.uniform(0.62, 0.85))
        font_path = rng.choice(self.fonts)
        for _ in range(20):
            font = ImageFont.truetype(font_path, max(8, size))
            bb = d.textbbox((0, 0), text, font=font)
            if (bb[2] - bb[0]) <= w * 0.93 and (bb[3] - bb[1]) <= avail * 0.92:
                break
            size = int(size * 0.9)
        font = ImageFont.truetype(font_path, max(8, size))

        # draw glyph by glyph so spacing can vary the way real plates do
        widths = [d.textlength(c, font=font) for c in text]
        gap = max(1.0, h * rng.uniform(0.01, 0.09))
        total = sum(widths) + gap * (len(text) - 1)
        x = (w - total) / 2.0
        bb = d.textbbox((0, 0), text, font=font)
        y = band_top + (avail - (bb[3] - bb[1])) / 2.0 - bb[1]
        for c, cw in zip(text, widths):
            d.text((x, y), c, fill=fg, font=font)
            x += cw + gap
        return np.array(img, np.uint8)

    # -- degradation ------------------------------------------------------
    def _degrade(self, img: np.ndarray) -> np.ndarray:
        rng, npr = self.rng, self.np_rng
        k = self.cfg.hard
        h, w = img.shape[:2]

        # Residual geometry the rectifier did not fully remove.
        #
        # These ranges were once widened much further (rotation +-15, shear
        # +-0.35, perspective 0.30, motion to 26px) on the theory that a
        # blurred image leaves the rectifier working from damaged edges, so
        # the model should train on more warped crops. Measured, that model
        # was worse on exactly the band it targeted - 0/6 against 2/6 - and
        # bluffed more often. Training destabilised too: loss climbed from
        # 0.14 to 0.48 after step 2000. Widening past this point makes the
        # task unlearnable rather than making the model tougher.
        if rng.random() < 0.75:
            ang = rng.uniform(-7, 7) * k
            m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
            img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if rng.random() < 0.55:
            sh = rng.uniform(-0.22, 0.22) * k
            m = np.array([[1, sh, -sh * h / 2], [0, 1, 0]], np.float64)
            img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if rng.random() < 0.55:
            d = rng.uniform(0.02, 0.16) * k
            src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
            dst = np.array([[w * rng.uniform(0, d), h * rng.uniform(0, d)],
                            [w * (1 - rng.uniform(0, d)), h * rng.uniform(0, d)],
                            [w * (1 - rng.uniform(0, d)), h * (1 - rng.uniform(0, d))],
                            [w * rng.uniform(0, d), h * (1 - rng.uniform(0, d))]], np.float32)
            img = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst),
                                      (w, h), borderMode=cv2.BORDER_REPLICATE)

        # crops that are a little too tight or too loose
        if rng.random() < 0.6:
            dx0 = int(w * rng.uniform(-0.05, 0.06))
            dx1 = int(w * rng.uniform(-0.05, 0.06))
            dy0 = int(h * rng.uniform(-0.10, 0.12))
            dy1 = int(h * rng.uniform(-0.10, 0.12))
            img = cv2.copyMakeBorder(img, max(0, dy0), max(0, dy1),
                                     max(0, dx0), max(0, dx1),
                                     cv2.BORDER_REPLICATE)
            y0, x0 = max(0, -dy0), max(0, -dx0)
            y1 = img.shape[0] - max(0, -dy1)
            x1 = img.shape[1] - max(0, -dx1)
            if y1 - y0 > 12 and x1 - x0 > 32:
                img = img[y0:y1, x0:x1]

        # illumination
        if rng.random() < 0.6:
            hh, ww = img.shape[:2]
            gy, gx = np.mgrid[0:hh, 0:ww].astype(np.float32)
            ramp = (gx / max(ww - 1, 1) * rng.uniform(-1, 1) +
                    gy / max(hh - 1, 1) * rng.uniform(-1, 1))
            img = clip8(img.astype(np.float32) * (1.0 + 0.35 * k * ramp))
        if rng.random() < 0.5:
            a = rng.uniform(0.55, 1.35)
            b = rng.uniform(-45, 45)
            img = clip8(img.astype(np.float32) * a + b)

        # optics
        if rng.random() < 0.55:
            from .restore import motion_psf
            img = clip8(cv2.filter2D(img.astype(np.float64), -1,
                                     motion_psf(rng.uniform(2, 16) * k,
                                                rng.uniform(0, 180))))
        if rng.random() < 0.6:
            img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.4, 2.6) * k)

        # sampling: shrink then stretch back, which is what a distant plate is
        if rng.random() < 0.7:
            f = rng.uniform(0.22, 0.85)
            hh, ww = img.shape[:2]
            small = cv2.resize(img, (max(16, int(ww * f)), max(8, int(hh * f))),
                               interpolation=cv2.INTER_AREA)
            img = cv2.resize(small, (ww, hh), interpolation=cv2.INTER_LINEAR)

        # screen patterning
        if rng.random() < 0.16:
            from .synth import screen_moire
            img = screen_moire(img, period=rng.uniform(2.4, 4.5),
                               angle=rng.uniform(0, 90),
                               strength=rng.uniform(0.08, 0.30))

        if rng.random() < 0.75:
            img = clip8(img.astype(np.float32) +
                        npr.normal(0, rng.uniform(2, 14) * k, img.shape))
        if rng.random() < 0.6:
            ok, buf = cv2.imencode(".jpg", img,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(25, 92)])
            if ok:
                img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        return img

    def sample(self) -> tuple[np.ndarray, str]:
        text = random_plate_text(self.rng)
        img = self._degrade(self._render(text))
        return img, text


# --------------------------------------------------------------------------
# real data
# --------------------------------------------------------------------------

_LABEL_RE = re.compile(r"^([A-Z0-9]{3,10})", re.I)


def load_real(directory) -> list[tuple[Path, str]]:
    """Load labelled photographs.

    Either put a `labels.csv` in the directory with `filename,text` rows, or
    name the files after the plate (`ABC1234.jpg`, `ABC1234_2.jpg`).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    pairs: list[tuple[Path, str]] = []
    csv_path = directory / "labels.csv"
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                if len(row) < 2 or row[0].strip().lower() in ("filename", "file"):
                    continue
                p = directory / row[0].strip()
                text = "".join(c for c in row[1].strip().upper() if c in ALPHABET)
                if p.is_file() and 3 <= len(text) <= MAX_LEN:
                    pairs.append((p, text))
        return pairs

    for p in sorted(directory.iterdir()):
        if p.suffix.lower() not in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            continue
        m = _LABEL_RE.match(p.stem)
        if not m:
            continue
        text = "".join(c for c in m.group(1).upper() if c in ALPHABET)
        if 3 <= len(text) <= MAX_LEN:
            pairs.append((p, text))
    return pairs


def load_real_image(path) -> np.ndarray:
    img = imread_any(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

"""The sequence recogniser: a small CRNN trained with CTC.

Why a sequence model rather than a better character classifier: the existing
template engine has to be handed one glyph at a time, so it inherits every
segmentation mistake. When blur merges two characters, or a frame edge splits
one, there is nothing the classifier can do about it. A CTC model reads the
whole strip and never commits to character boundaries at all - it emits a
per-column distribution and lets the alignment fall out. That is precisely the
failure mode the template pipeline could not fix.

It is small on purpose: it has to train on a CPU in minutes and run inside an
ensemble that already does a lot of work per image.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import BLANK, IMG_H, IMG_W, NUM_CLASSES, decode_greedy, prepare


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _build(torch, nn):
    class ConvBlock(nn.Module):
        def __init__(self, cin, cout, pool):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )
            self.pool = nn.MaxPool2d(pool) if pool else None

        def forward(self, x):
            x = self.body(x)
            return self.pool(x) if self.pool is not None else x

    class CRNN(nn.Module):
        """48x160 grayscale in, (T=40, classes) log-probabilities out."""

        def __init__(self, num_classes: int = NUM_CLASSES, hidden: int = 128):
            super().__init__()
            self.cnn = nn.Sequential(
                ConvBlock(1, 32, (2, 2)),      # 24 x 80
                ConvBlock(32, 64, (2, 2)),     # 12 x 40
                ConvBlock(64, 128, (2, 1)),    #  6 x 40
                ConvBlock(128, 256, (2, 1)),   #  3 x 40
                ConvBlock(256, 256, (3, 1)),   #  1 x 40
            )
            self.rnn = nn.GRU(256, hidden, num_layers=2, bidirectional=True,
                              batch_first=True, dropout=0.1)
            self.head = nn.Linear(hidden * 2, num_classes)

        def forward(self, x):
            f = self.cnn(x)                    # B, C, 1, T
            f = f.squeeze(2).permute(0, 2, 1)  # B, T, C
            f, _ = self.rnn(f)
            return self.head(f)                # B, T, classes

    return CRNN


def make_model(hidden: int = 128):
    import torch
    import torch.nn as nn
    return _build(torch, nn)(NUM_CLASSES, hidden)


def save(model, path, meta: dict | None = None) -> None:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "meta": meta or {},
                "img_h": IMG_H, "img_w": IMG_W,
                "num_classes": NUM_CLASSES}, path)


def load(path, hidden: int = 128):
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=False)
    model = make_model(hidden)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob.get("meta", {})


class CRNNOCR:
    """Engine adapter so the model can join the existing vote.

    It reads a whole plate crop, so unlike the template engine it is given the
    plate rather than a list of glyphs. Confidence comes from the CTC
    posteriors, which is a genuine per-character probability rather than a
    similarity score - but note it is the model's own confidence, and a model
    is perfectly capable of being confidently wrong about a plate that is not
    there. It joins the vote; it does not overrule the image-quality gate.
    """

    name = "crnn"

    def __init__(self, path, hidden: int = 128):
        import torch
        self._torch = torch
        self.model, self.meta = load(path, hidden)
        torch.set_num_threads(max(1, min(4, (torch.get_num_threads() or 1))))

    def read_image(self, plate_gray: np.ndarray) -> list[tuple[str, list[float]]]:
        torch = self._torch
        x = prepare(plate_gray)[None, None]
        with torch.no_grad():
            logits = self.model(torch.from_numpy(x))
            probs = torch.softmax(logits, dim=-1)[0].numpy()

        path = probs.argmax(axis=1)
        text = decode_greedy(path)
        if not text:
            return []

        # per-character confidence: the peak probability of each non-blank run
        confs: list[float] = []
        prev = -1
        run: list[float] = []
        for t, idx in enumerate(path):
            idx = int(idx)
            if idx != prev:
                if run:
                    confs.append(float(max(run)))
                run = []
            if idx != BLANK:
                run.append(float(probs[t, idx]))
            prev = idx
        if run:
            confs.append(float(max(run)))
        confs = confs[:len(text)] or [0.5] * len(text)
        while len(confs) < len(text):
            confs.append(confs[-1])
        return [(text, confs)]

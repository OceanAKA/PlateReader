"""Training the sequence recogniser.

Synthetic data is generated on the fly, so there is no dataset to download and
no epoch structure - training runs for a step budget. Real labelled crops, if
you have any, are mixed in at a fixed rate and also form the validation set,
because validating on synthetic data only tells you how well the model learned
the generator.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .dataset import (IMG_H, IMG_W, MAX_LEN, PlateSynth, SynthConfig, decode_greedy,
                      encode, load_real, load_real_image, prepare)
from .model import make_model, save


@dataclass
class TrainConfig:
    steps: int = 3000
    batch_size: int = 48
    lr: float = 2e-3
    hidden: int = 128
    real_dir: str | None = None
    real_ratio: float = 0.35        # share of each batch drawn from real data
    val_every: int = 250
    val_size: int = 256
    seed: int = 0
    out: str = "models/crnn.pt"
    workers: int = 0
    curriculum: bool = True         # start easy, get harder
    notes: dict = field(default_factory=dict)


def _augment_real(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Light augmentation for real crops - they are scarce, do not mangle them."""
    h, w = img.shape[:2]
    if rng.random() < 0.6:
        ang = rng.uniform(-4, 4)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.5:
        a, b = rng.uniform(0.75, 1.25), rng.uniform(-25, 25)
        img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
    if rng.random() < 0.4:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.2))
    return img


class Batcher:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.synth = PlateSynth(seed=cfg.seed, config=SynthConfig(hard=1.0))
        self.real: list[tuple[Path, str]] = []
        self.real_val: list[tuple[Path, str]] = []
        if cfg.real_dir:
            pairs = load_real(cfg.real_dir)
            self.rng.shuffle(pairs)
            cut = max(1, int(len(pairs) * 0.2)) if len(pairs) >= 5 else 0
            self.real_val = pairs[:cut]
            self.real = pairs[cut:]
        self._cache: dict[Path, np.ndarray] = {}

    def _real_sample(self) -> tuple[np.ndarray, str] | None:
        if not self.real:
            return None
        path, text = self.rng.choice(self.real)
        img = self._cache.get(path)
        if img is None:
            try:
                img = load_real_image(path)
            except Exception:
                return None
            self._cache[path] = img
        return _augment_real(img, self.rng), text

    def batch(self, n: int, hard: float = 1.0):
        self.synth.cfg.hard = hard
        imgs, texts = [], []
        while len(imgs) < n:
            use_real = self.real and self.rng.random() < self.cfg.real_ratio
            got = self._real_sample() if use_real else self.synth.sample()
            if got is None:
                got = self.synth.sample()
            img, text = got
            if not text:
                continue
            imgs.append(prepare(img))
            texts.append(text[:MAX_LEN])
        x = np.stack(imgs)[:, None]
        return x, texts

    def validation(self, n: int):
        """Prefer held-out real crops; fall back to fresh synthetic ones."""
        if self.real_val:
            out = []
            for path, text in self.real_val[:n]:
                try:
                    out.append((prepare(load_real_image(path)), text))
                except Exception:
                    continue
            if out:
                x = np.stack([a for a, _ in out])[:, None]
                return x, [t for _, t in out], "real"
        rng_state = self.synth.rng.getstate()
        self.synth.rng.seed(99991)
        imgs, texts = [], []
        for _ in range(n):
            img, text = self.synth.sample()
            imgs.append(prepare(img))
            texts.append(text)
        self.synth.rng.setstate(rng_state)
        return np.stack(imgs)[:, None], texts, "synthetic"


def _accuracy(model, torch, x, texts) -> tuple[float, float]:
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(x))
        paths = logits.argmax(dim=-1).cpu().numpy()
    model.train()
    exact = chars = total = 0
    for p, truth in zip(paths, texts):
        got = decode_greedy(p)
        exact += (got == truth)
        chars += sum(1 for a, b in zip(truth, got) if a == b)
        total += len(truth)
    return exact / max(1, len(texts)), chars / max(1, total)


def train(cfg: TrainConfig, log=print) -> dict:
    import torch
    import torch.nn as nn

    torch.manual_seed(cfg.seed)
    batcher = Batcher(cfg)
    model = make_model(cfg.hidden)
    model.train()

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.25)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    n_real = len(batcher.real)
    log(f"training {cfg.steps} steps, batch {cfg.batch_size}, "
        f"{n_real} real crop(s) mixed in at {cfg.real_ratio:.0%}"
        if n_real else
        f"training {cfg.steps} steps, batch {cfg.batch_size}, synthetic only")

    best = {"exact": -1.0, "step": 0}
    started = time.time()
    for step in range(1, cfg.steps + 1):
        # Curriculum: mild degradation first. Starting at full severity makes
        # CTC collapse to all-blank and never recover.
        hard = min(1.0, 0.35 + 0.65 * step / (0.6 * cfg.steps)) if cfg.curriculum else 1.0
        x, texts = batcher.batch(cfg.batch_size, hard)

        targets = [encode(t) for t in texts]
        flat = torch.tensor([i for t in targets for i in t], dtype=torch.long)
        tgt_len = torch.tensor([len(t) for t in targets], dtype=torch.long)

        logits = model(torch.from_numpy(x))              # B, T, C
        logp = logits.log_softmax(-1).permute(1, 0, 2)   # T, B, C
        inp_len = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long)

        loss = ctc(logp, flat, inp_len, tgt_len)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()

        if step % cfg.val_every == 0 or step == cfg.steps:
            vx, vtexts, kind = batcher.validation(cfg.val_size)
            ex, ch = _accuracy(model, torch, vx, vtexts)
            elapsed = time.time() - started
            log(f"  step {step:>5}/{cfg.steps}  loss {loss.item():6.3f}  "
                f"{kind} val: {ex:5.1%} exact  {ch:5.1%} chars  "
                f"({elapsed:.0f}s)")
            if ex >= best["exact"]:
                best = {"exact": ex, "chars": ch, "step": step, "val": kind}
                save(model, cfg.out, meta={
                    "steps": step, "val_exact": ex, "val_chars": ch,
                    "val_set": kind, "real_crops": n_real,
                    "batch": cfg.batch_size, "hidden": cfg.hidden,
                    "img": [IMG_H, IMG_W], **cfg.notes,
                })

    log(f"best {best['exact']:.1%} exact on the {best.get('val')} validation set "
        f"at step {best['step']}; saved to {cfg.out}")
    return best

"""Finding where the tool stops working, one degradation at a time.

"Make it read the hard ones" is only actionable if the hard band can be
located. This sweeps a single degradation axis while holding everything else
fixed, and reports the value at which accuracy falls through a threshold.

Two numbers matter per axis, and they are different questions:

  * the operating limit - where this implementation gives up;
  * the information limit - where the pixels stop carrying the answer at all.

When those coincide, the tool is as good as the image allows and further work
on it is wasted. When the operating limit comes first, the gap is headroom:
the answer is still in the image and the code is failing to extract it. Motion
blur is currently the widest such gap, which is why it is worth attacking and
why resolution is not.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from .formats import FormatSet
from .ocr import default_engines
from .pipeline import read_image
from .synth import make_sample, random_text

# Rules of thumb for unaided human reading of a still image, used only to say
# roughly where a person would give up. They are approximations, not claims
# about any particular person or display.
HUMAN_NOTES = {
    "scale": "a person needs roughly 15-20px of character height on a screen",
    "motion": "a person loses a plate once the smear passes ~1.5x stroke width",
    "blur": "a person copes to roughly 1x stroke width of defocus radius",
    "noise": "the eye averages noise well; people stay readable to high sigma",
    "quality": "blocking artefacts bite a person at about quality 20",
}


@dataclass
class AxisResult:
    axis: str
    values: list[float]
    exact: list[float]
    chars: list[float]
    limit: float | None                # last value still above the threshold
    note: str = ""
    refused: list[float] = field(default_factory=list)
    false_confident: list[float] = field(default_factory=list)


def _degrade_kwargs(axis: str, value: float) -> dict:
    if axis == "scale":
        return dict(scale=value)
    if axis == "motion":
        return dict(motion=value, motion_angle=8.0)
    if axis == "blur":
        return dict(blur=value)
    if axis == "noise":
        return dict(noise=value)
    if axis == "quality":
        return dict(quality=int(value))
    raise ValueError(f"unknown axis {axis!r}")


DEFAULT_SWEEPS: dict[str, list[float]] = {
    "scale": [1.0, 0.7, 0.5, 0.4, 0.32, 0.26, 0.22, 0.18, 0.15, 0.12],
    "motion": [0, 4, 8, 12, 16, 20, 25, 30, 36],
    "blur": [0, 0.8, 1.5, 2.2, 3.0, 3.8, 4.6, 5.5],
    "noise": [0, 5, 10, 16, 24, 34, 46, 60],
    # JPEG alone never broke it at 10, so the range runs lower: an axis
    # that bottoms out at 100% has not found a limit, it has run out of
    # sweep.
    "quality": [95, 80, 65, 50, 38, 28, 20, 14, 10, 7, 5, 3, 2],
}


def sweep_axis(axis: str, samples: int = 6, threshold: float = 0.5,
               engines=None, formats=None, aggressive: bool = True,
               base: dict | None = None, seed: int = 5,
               values: list[float] | None = None,
               progress=None) -> AxisResult:
    """Sweep one degradation and find where exact accuracy drops below threshold."""
    engines = engines if engines is not None else default_engines()
    formats = formats if formats is not None else FormatSet()
    values = values if values is not None else DEFAULT_SWEEPS[axis]
    base = base or {}

    exact_rates: list[float] = []
    char_rates: list[float] = []
    refused_rates: list[float] = []
    bluff_rates: list[float] = []
    limit: float | None = None

    for value in values:
        rng = random.Random(seed)
        hits = chars = total = refused = bluffs = 0
        for _ in range(samples):
            truth = random_text(rng=rng)
            kw = dict(base)
            kw.update(_degrade_kwargs(axis, value))
            img = make_sample(text=truth, seed=rng.randrange(1 << 30), **kw)
            results = read_image(img, aggressive=aggressive, engines=engines,
                                 max_candidates=3, formats=formats)
            got = results[0].text if results else ""
            if results and not results[0].quality.readable:
                refused += 1
            # A wrong answer the tool did NOT flag. Measured against
            # `trustworthy`, not the quality verdict alone: quality only asks
            # whether the pixels were sufficient, while `trustworthy` is what
            # actually drives the reassuring banner the caller sees, and it
            # additionally requires the processing chains to have agreed. An
            # earlier version of this metric checked the weaker condition and
            # therefore under-reported bluffing.
            if results and results[0].trustworthy and got != truth:
                bluffs += 1
            total += len(truth)
            chars += sum(1 for a, b in zip(truth, got) if a == b)
            hits += (got == truth)
        ex = hits / max(1, samples)
        exact_rates.append(ex)
        char_rates.append(chars / max(1, total))
        refused_rates.append(refused / max(1, samples))
        bluff_rates.append(bluffs / max(1, samples))
        if ex >= threshold:
            limit = value
        if progress:
            progress(axis, value, ex, chars / max(1, total),
                     refused / max(1, samples), bluffs / max(1, samples))

    return AxisResult(axis=axis, values=list(values), exact=exact_rates,
                      chars=char_rates, limit=limit,
                      note=HUMAN_NOTES.get(axis, ""), refused=refused_rates,
                      false_confident=bluff_rates)

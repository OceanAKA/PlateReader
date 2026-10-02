"""Cross-checking candidate readings against a real vehicle.

When the pipeline is torn between two readings, the strongest remaining
evidence is outside the plate: one of them belongs to a vehicle that exists,
and the other does not. If a lookup says GS58KTV is a silver Honda and the car
in the photograph is plainly silver, that settles it.

No lookup service is bundled, and this module deliberately does not embed one.
There is no free public plate-to-vehicle database:

  * United States - motor vehicle records are restricted by the Driver's
    Privacy Protection Act (18 U.S.C. 2721). Commercial resellers exist but
    require you to attest to a permissible use.
  * United Kingdom - the DVLA Vehicle Enquiry Service returns make, colour
    and tax/MOT status (no keeper details) via an API key you register for.
  * Elsewhere - varies, and is usually restricted.

So this is a plug: you supply a command that queries a source you are entitled
to query, and plateread uses its answers to break ties. The colour check below
needs no database at all and works offline.
"""
from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass, field

import cv2
import numpy as np

from .detect import order_quad
from .util import ALPHABET


@dataclass
class VehicleRecord:
    plate: str
    make: str | None = None
    model: str | None = None
    colour: str | None = None
    year: str | None = None
    raw: dict = field(default_factory=dict)

    def describe(self) -> str:
        bits = [b for b in (self.year, self.colour, self.make, self.model) if b]
        return " ".join(str(b) for b in bits) if bits else "registered vehicle"


class CommandLookup:
    """Query an external command, one plate at a time.

    The command is a template containing {plate}; it is run without a shell,
    and the plate is restricted to A-Z0-9 by the OCR alphabet, so there is
    nothing to escape. Anything JSON-shaped on stdout is accepted; the usual
    field names are picked out and the whole payload is kept.

        --verify-cmd "dvla-lookup {plate}"
        --verify-cmd "curl -s -H @hdr https://example/api/{plate}"
    """

    FIELD_ALIASES = {
        "make": ("make", "manufacturer", "brand", "vehicle_make"),
        "model": ("model", "vehicle_model", "series"),
        "colour": ("colour", "color", "vehicle_colour", "vehicle_color"),
        "year": ("year", "yearOfManufacture", "year_of_manufacture", "model_year"),
    }

    def __init__(self, template: str, timeout: float = 15.0):
        if "{plate}" not in template:
            raise ValueError("verify command must contain {plate}")
        self.template = template
        self.timeout = timeout

    def __call__(self, plate: str) -> VehicleRecord | None:
        if not plate or any(c not in ALPHABET for c in plate):
            return None
        argv = [part.replace("{plate}", plate) for part in shlex.split(self.template)]
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=self.timeout, shell=False)
        except (OSError, subprocess.SubprocessError):
            return None
        if proc.returncode != 0 or not proc.stdout.strip():
            return None
        try:
            data = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return None
        if isinstance(data, list):
            data = data[0] if data else {}
        if not isinstance(data, dict):
            return None
        # an explicit "not found" is a negative answer, not a malformed one
        if data.get("found") is False or data.get("error"):
            return None

        def pick(key):
            for alias in self.FIELD_ALIASES[key]:
                for k, v in data.items():
                    if k.lower() == alias.lower() and v not in (None, ""):
                        return str(v)
            return None

        return VehicleRecord(plate=plate, make=pick("make"), model=pick("model"),
                             colour=pick("colour"), year=pick("year"), raw=data)


# --------------------------------------------------------------------------
# offline cross-check: what colour is the car
# --------------------------------------------------------------------------

# Coarse buckets, because a registry says "SILVER" not "#B4B8BC".
_CHROMATIC = [
    ("red", 0), ("orange", 18), ("yellow", 30), ("green", 60),
    ("cyan", 90), ("blue", 112), ("purple", 140), ("red", 175),
]


def _name_hsv(h: float, s: float, v: float) -> str:
    if v < 55:
        return "black"
    if s < 40:
        if v > 190:
            return "white"
        return "silver" if v > 110 else "grey"
    best, bestd = "red", 999.0
    for name, hue in _CHROMATIC:
        d = min(abs(h - hue), 180 - abs(h - hue))
        if d < bestd:
            best, bestd = name, d
    if best == "orange" and v < 140:
        return "brown"
    return best


def dominant_colour(bgr: np.ndarray, quad: np.ndarray) -> str | None:
    """Body colour of the vehicle carrying this plate.

    Samples a band above the plate and to either side - bodywork, not the
    plate itself, not the road under it - and reports the majority bucket.
    """
    if bgr is None or bgr.ndim != 3:
        return None
    q = order_quad(quad)
    x0, y0 = q.min(axis=0)
    x1, y1 = q.max(axis=0)
    pw, ph = x1 - x0, y1 - y0
    if pw < 8 or ph < 5:
        return None

    h, w = bgr.shape[:2]
    # a band starting one plate-height above the plate, two plates wide
    bx0 = int(max(0, x0 - pw * 0.5))
    bx1 = int(min(w, x1 + pw * 0.5))
    by0 = int(max(0, y0 - ph * 3.0))
    by1 = int(max(0, y0 - ph * 0.35))
    if by1 - by0 < 4 or bx1 - bx0 < 8:
        return None

    patch = bgr[by0:by1, bx0:bx1]
    if patch.size < 48:
        return None
    hsv = cv2.cvtColor(cv2.GaussianBlur(patch, (5, 5), 0), cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()

    tally: dict[str, int] = {}
    for i in range(0, hh.size, max(1, hh.size // 4000)):
        name = _name_hsv(float(hh[i]), float(ss[i]), float(vv[i]))
        tally[name] = tally.get(name, 0) + 1
    if not tally:
        return None
    total = sum(tally.values())
    name, count = max(tally.items(), key=lambda kv: kv[1])
    return name if count / total >= 0.34 else None


def colours_agree(observed: str | None, recorded: str | None) -> bool | None:
    """None when it cannot be judged, else whether the two descriptions match."""
    if not observed or not recorded:
        return None
    rec = recorded.strip().lower()
    obs = observed.strip().lower()
    equivalents = {
        "silver": {"silver", "grey", "gray", "aluminium"},
        "grey": {"grey", "gray", "silver"},
        "white": {"white", "cream", "ivory", "pearl"},
        "black": {"black"},
        "blue": {"blue", "navy", "cyan", "turquoise"},
        "cyan": {"cyan", "blue", "turquoise"},
        "red": {"red", "maroon", "burgundy", "crimson"},
        "green": {"green", "olive"},
        "yellow": {"yellow", "gold"},
        "orange": {"orange", "bronze"},
        "brown": {"brown", "beige", "bronze", "tan"},
        "purple": {"purple", "violet", "mauve"},
    }
    allowed = equivalents.get(obs, {obs})
    return any(a in rec for a in allowed)


# --------------------------------------------------------------------------
# tie-breaking
# --------------------------------------------------------------------------

@dataclass
class Verification:
    plate: str
    prior: float                      # probability from the OCR vote
    record: VehicleRecord | None
    colour_match: bool | None

    @property
    def resolved(self) -> bool:
        return self.record is not None

    @property
    def score(self) -> float:
        s = self.prior
        if self.record is None:
            return s * 0.15               # exists nowhere: heavily discounted
        s *= 3.0
        if self.colour_match is True:
            s *= 2.0
        elif self.colour_match is False:
            s *= 0.35
        return s


def verify_candidates(candidates: list[tuple[str, float]], lookup,
                      observed_colour: str | None = None,
                      limit: int = 5) -> list[Verification]:
    """Query the top candidate readings and rank by what came back.

    Only the shortlist is queried - every call is a request against someone
    else's service, and the tail of the beam is noise anyway.
    """
    out: list[Verification] = []
    seen: set[str] = set()
    for text, prior in candidates[:limit]:
        if text in seen:
            continue
        seen.add(text)
        try:
            record = lookup(text)
        except Exception:
            record = None
        match = colours_agree(observed_colour, record.colour if record else None)
        out.append(Verification(text, float(prior), record, match))
    out.sort(key=lambda v: -v.score)
    return out

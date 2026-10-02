"""Real-world plate grammars, used to choose between competing readings.

A plate is not a free string. Every jurisdiction issues a small number of
fixed shapes - three letters then four digits, a letter then two digits then
three letters - and that structure resolves most of the ambiguity OCR cannot.
If position 5 must be a digit, then a glyph the classifier scored as "S" is an
S nowhere and a 5 everywhere.

Formats are written as masks rather than regexes because a mask says what each
*position* must be, which is what makes position-aware correction possible:

    L = letter        D = digit        A = either

IMPORTANT: this table is representative, not exhaustive or authoritative.
Jurisdictions change series, and every one of them issues specialist plates
(vanity, government, trade, diplomatic, motorcycle) that these masks do not
cover. Treat a format match as evidence, never as proof, and supply your own
table with --formats when you know the local series.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .util import DIGITS, LETTERS

# Glyph pairs that a degraded image genuinely confuses, split by direction.
DIGIT_TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A",
                   "5": "S", "6": "G", "7": "T", "8": "B"}
LETTER_TO_DIGIT = {"O": "0", "D": "0", "Q": "0", "I": "1", "J": "1", "L": "1",
                   "T": "7", "Z": "2", "A": "4", "S": "5", "G": "6", "B": "8"}


@dataclass(frozen=True)
class PlateFormat:
    region: str
    name: str
    mask: str
    example: str

    @property
    def length(self) -> int:
        return len(self.mask)

    def regex(self) -> str:
        parts = {"L": "[A-Z]", "D": "[0-9]", "A": "[A-Z0-9]"}
        return "".join(parts.get(c, re.escape(c)) for c in self.mask)

    def matches(self, text: str) -> bool:
        if len(text) != len(self.mask):
            return False
        for ch, m in zip(text, self.mask):
            if m == "L" and ch not in LETTERS:
                return False
            if m == "D" and ch not in DIGITS:
                return False
        return True


def _f(region, name, mask, example):
    return PlateFormat(region, name, mask, example)


# Common current passenger series. Deliberately conservative - where a
# jurisdiction runs several series, the widespread ones are listed.
BUILTIN_FORMATS: list[PlateFormat] = [
    # --- United Kingdom -------------------------------------------------
    _f("uk", "UK current (2001-)", "LLDDLLL", "AB12CDE"),
    _f("uk", "UK prefix (1983-2001)", "LDDDLLL", "A123BCD"),
    _f("uk", "UK suffix (1963-1983)", "LLLDDDL", "ABC123D"),

    # --- Europe ---------------------------------------------------------
    _f("fr", "France SIV (2009-)", "LLDDDLL", "AB123CD"),
    _f("es", "Spain (2000-)", "DDDDLLL", "1234BCD"),
    _f("it", "Italy (1994-)", "LLDDDLL", "AB123CD"),
    _f("nl", "Netherlands (sidecode 8)", "LDDDLL", "L123AB"),
    _f("de", "Germany (short district)", "LLDDDD", "BX1234"),
    _f("de", "Germany (long district)", "LLLLDDDD", "MSTA1234"),
    _f("pl", "Poland", "LLDDDLL", "WA123CD"),
    _f("be", "Belgium (2010-)", "DLLLDDD", "1ABC123"),

    # --- United States (representative state series) ---------------------
    _f("us-ca", "California passenger", "DLLLDDD", "1ABC234"),
    _f("us-ny", "New York (2001-)", "LLLDDDD", "ABC1234"),
    _f("us-tx", "Texas", "LLLDDDD", "ABC1234"),
    _f("us-fl", "Florida", "LLLDDL", "ABC12D"),
    _f("us-il", "Illinois passenger", "LLDDDDD", "AB12345"),
    _f("us-pa", "Pennsylvania", "LLLDDDD", "ABC1234"),
    _f("us-oh", "Ohio", "LLLDDDD", "ABC1234"),
    _f("us-ga", "Georgia", "LLLDDDD", "ABC1234"),
    _f("us-nj", "New Jersey", "LDDLLL", "A12BCD"),
    _f("us-mi", "Michigan", "DLLLDD", "1ABC23"),
    _f("us-wa", "Washington", "LLLDDDD", "ABC1234"),
    _f("us-va", "Virginia", "LLLDDDD", "ABC1234"),
    _f("us-nc", "North Carolina", "LLLDDDD", "ABC1234"),
    _f("us-az", "Arizona", "LLLDDDD", "ABC1234"),
    _f("us-ma", "Massachusetts", "DLLLDD", "1ABC23"),

    # --- Rest of world ---------------------------------------------------
    _f("in", "India (BH/state series)", "LLDDLLDDDD", "MH12AB1234"),
    _f("in", "India (single-letter series)", "LLDDLDDDD", "KA05A1234"),
    _f("au-nsw", "Australia NSW", "LLLDDLL", "ABC12DE"),
    _f("au-vic", "Australia Victoria", "LLLDDD", "ABC123"),
    _f("ca-on", "Ontario", "LLLLDDD", "ABCD123"),
    _f("br", "Brazil Mercosul", "LLLDLDD", "ABC1D23"),
    _f("za", "South Africa (GP)", "LLDDLLL", "AB12CDE"),
    _f("jp", "Japan (numeric block)", "DDDD", "1234"),

    # --- Generic fallbacks, only used when nothing specific fits ---------
    _f("generic", "Generic 6 alphanumeric", "AAAAAA", "AB12CD"),
    _f("generic", "Generic 7 alphanumeric", "AAAAAAA", "AB123CD"),
]


def load_formats(path: str | Path) -> list[PlateFormat]:
    """Load a user format table.

    JSON: [{"region": "us-ca", "name": "...", "mask": "DLLLDDD",
            "example": "1ABC234"}, ...]
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for row in data:
        mask = str(row["mask"]).upper()
        if not mask or any(c not in "LDA" for c in mask):
            raise ValueError(f"bad mask {mask!r}: use only L, D and A")
        out.append(PlateFormat(str(row.get("region", "custom")),
                               str(row.get("name", mask)),
                               mask,
                               str(row.get("example", ""))))
    if not out:
        raise ValueError("format file contained no formats")
    return out


def coerce_char(ch: str, cls: str) -> str | None:
    """Bend one character into the class a position demands, if plausible."""
    if cls == "A":
        return ch
    if cls == "L":
        if ch in LETTERS:
            return ch
        return DIGIT_TO_LETTER.get(ch)
    if cls == "D":
        if ch in DIGITS:
            return ch
        return LETTER_TO_DIGIT.get(ch)
    return None


class FormatSet:
    """A table of plate grammars, and the reranking that uses it."""

    SUBSTITUTION_PENALTY = 0.85

    def __init__(self, formats: list[PlateFormat] | None = None,
                 region: str | None = None):
        formats = list(formats if formats is not None else BUILTIN_FORMATS)
        if region:
            region = region.lower()
            picked = [f for f in formats
                      if f.region == region or f.region.startswith(region + "-")]
            if not picked:
                known = sorted({f.region for f in formats})
                raise ValueError(f"unknown region {region!r}; known: {', '.join(known)}")
            formats = picked
        self.formats = formats
        self.lengths = {f.length for f in formats}

    def match(self, text: str) -> PlateFormat | None:
        """First format this string already satisfies, specific ones first."""
        exact = [f for f in self.formats if f.region != "generic" and f.matches(text)]
        if exact:
            return exact[0]
        for f in self.formats:
            if f.matches(text):
                return f
        return None

    def coerce(self, text: str) -> tuple[str, PlateFormat] | None:
        """Nudge a whole string into the nearest format it can satisfy."""
        best: tuple[int, str, PlateFormat] | None = None
        for f in self.formats:
            if len(text) != f.length:
                continue
            out, changed = [], 0
            for ch, cls in zip(text, f.mask):
                got = coerce_char(ch, cls)
                if got is None:
                    break
                changed += (got != ch)
                out.append(got)
            else:
                if best is None or changed < best[0]:
                    best = (changed, "".join(out), f)
        return (best[1], best[2]) if best else None

    def constrain(self, per_pos: list[list[tuple[str, float]]]
                  ) -> tuple[str, float, float, PlateFormat] | None:
        """Re-read the per-position distributions under each grammar.

        This is stronger than fixing up the winning string: where the top
        choice is the wrong class for its position, the runner-up is often the
        right character outright, and this picks it up rather than blindly
        substituting the top choice into the required class.

        Returns (text, ranking score, mean per-position confidence, format).
        """
        best: tuple[str, float, float, PlateFormat] | None = None
        for f in self.formats:
            if len(per_pos) != f.length:
                continue
            chars: list[str] = []
            shares: list[float] = []
            score = 1.0
            for dist, cls in zip(per_pos, f.mask):
                pick = None
                for ch, p in dist:
                    got = coerce_char(ch, cls)
                    if got is None:
                        continue
                    s = p * (1.0 if got == ch else self.SUBSTITUTION_PENALTY)
                    if pick is None or s > pick[1]:
                        pick = (got, s)
                if pick is None:
                    break
                chars.append(pick[0])
                shares.append(pick[1])
                score *= max(pick[1], 1e-4)
            else:
                # prefer a specific jurisdiction over the catch-all masks
                weighted = score * (0.7 if f.region == "generic" else 1.0)
                if best is None or weighted > best[1]:
                    mean_share = sum(shares) / max(1, len(shares))
                    best = ("".join(chars), weighted, mean_share, f)
        return best

    def describe(self, text: str) -> str:
        f = self.match(text)
        return f"{f.name} ({f.example})" if f else "no known format"

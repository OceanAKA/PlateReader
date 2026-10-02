#!/usr/bin/env python3
"""Template for a --verify-cmd adapter.

plateread runs this once per candidate reading:

    plateread read photo.jpg --verify-cmd "python examples/lookup_stub.py {plate}"

Contract: take the plate as argv[1], print one JSON object on stdout. Any of
make / model / colour / year (and the common aliases) are picked up; print
{"found": false} when the plate resolves to nothing. Exit non-zero or print
nothing and the candidate is simply treated as unresolved.

To make it real, replace TABLE with a call to a source you are entitled to
query - your own fleet database, a parking system, or an official API you hold
a key for. Do not point it at a service you are not authorised to use.
"""
import json
import sys

TABLE = {
    "OSG8347": {"make": "Honda", "model": "Civic",
                "colour": "Silver", "yearOfManufacture": "2019"},
    "AB12CDE": {"make": "Ford", "model": "Focus", "colour": "Blue"},
}


def lookup(plate: str) -> dict | None:
    return TABLE.get(plate.upper())


def main() -> int:
    if len(sys.argv) < 2:
        print(json.dumps({"found": False}))
        return 0
    plate = sys.argv[1].upper()
    record = lookup(plate)
    print(json.dumps(dict(record, plate=plate) if record else {"found": False}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

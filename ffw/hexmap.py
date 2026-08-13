"""Hex geometry for the Fifth Frontier War stellar display.

Hexes use the Traveller four-digit convention ``CCRR`` -- two digits of column
followed by two digits of row.  Columns run spinward-to-trailing, rows run
coreward-to-rimward, and odd-numbered columns sit half a hex higher than
even-numbered ones (0201 lies between 0101 and 0102).
"""

from __future__ import annotations

import math
import re
from typing import Iterable, Iterator

HEX_RE = re.compile(r"^\d{4}$")


def parse(hex_id: str) -> tuple[int, int]:
    """``"2314" -> (23, 14)``."""
    if not HEX_RE.match(hex_id):
        raise ValueError("bad hex id %r" % (hex_id,))
    return int(hex_id[:2]), int(hex_id[2:])


def format_hex(col: int, row: int) -> str:
    return "%02d%02d" % (col, row)


def to_cube(hex_id: str) -> tuple[int, int, int]:
    """Offset ("even-q", even columns pushed down) to cube coordinates."""
    col, row = parse(hex_id)
    x = col
    z = row - (col + (col & 1)) // 2
    return x, -x - z, z


def distance(a: str, b: str) -> int:
    """Jump distance in parsecs between two hexes."""
    ax, ay, az = to_cube(a)
    bx, by, bz = to_cube(b)
    return max(abs(ax - bx), abs(ay - by), abs(az - bz))


def neighbours(hex_id: str) -> list[str]:
    col, row = parse(hex_id)
    up = col % 2 == 1          # odd columns sit half a hex higher
    if up:
        deltas = [(0, -1), (0, 1), (-1, -1), (-1, 0), (1, -1), (1, 0)]
    else:
        deltas = [(0, -1), (0, 1), (-1, 0), (-1, 1), (1, 0), (1, 1)]
    return [format_hex(col + dc, row + dr) for dc, dr in deltas]


def within(hex_id: str, radius: int, universe: Iterable[str]) -> list[str]:
    """Every hex of ``universe`` no further than ``radius`` from ``hex_id``."""
    return [h for h in universe if distance(hex_id, h) <= radius]


def pixel_centre(hex_id: str, size: float = 1.0) -> tuple[float, float]:
    """Cartesian centre used by the map renderer (y grows downwards)."""
    col, row = parse(hex_id)
    x = size * 1.5 * col
    y = size * math.sqrt(3) * (row + (0.0 if col % 2 else 0.5))
    return x, y


def ring(hex_id: str, radius: int) -> Iterator[str]:
    """All hexes exactly ``radius`` away (unbounded by the map)."""
    if radius == 0:
        yield hex_id
        return
    col, row = parse(hex_id)
    for dc in range(-radius, radius + 1):
        for dr in range(-radius * 2, radius * 2 + 1):
            cand = format_hex(col + dc, row + dr)
            if 0 <= col + dc and 0 <= row + dr and distance(hex_id, cand) == radius:
                yield cand

"""Combat results tables and modifier tables, transcribed from the Fifth
Frontier War chart sheet (page 22 of the scanned rules).

All tables use the game's "closest column without exceeding" convention: a
firing total picks the highest column whose header is <= the total.
"""

from __future__ import annotations

import bisect
from typing import Sequence

# --------------------------------------------------------------------------
# Space Combat Results Table (rule 4)
# columns are total attack factors; rows are the die roll 1-6; values are
# defence factors of battle damage inflicted ("-" is no effect -> 0)
# --------------------------------------------------------------------------
SPACE_COLUMNS = [1, 3, 6, 12, 18, 24, 30, 36, 42, 48]
SPACE_CRT = {
    1: [0, 0, 0, 1, 2, 3, 4, 5, 6, 7],
    2: [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    3: [0, 0, 1, 2, 4, 5, 6, 7, 8, 9],
    4: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    5: [0, 1, 2, 3, 5, 6, 7, 8, 9, 10],
    6: [1, 2, 3, 4, 6, 7, 8, 9, 10, 11],
}

# --------------------------------------------------------------------------
# Squadron Against SDB Combat Table
# columns are total bombardment factors, rows are the full SDB strength; the
# result is a percentage loss to the SDBs.  ``DESTROYED`` is the table's "d".
# --------------------------------------------------------------------------
DESTROYED = 100

SDB_ROWS = [1, 2, 5, 10, 12, 15, 20, 50, 100, 120, 150, 200, 500, 1000,
            1200, 1500, 2000, 5000]
_D = DESTROYED
_X = None  # table dash: no effect
SQUADRON_VS_SDB = {
    1:    [80, 90, _D, _D, _D, _D, _D, _D, _D, _D],
    2:    [70, 80, 90, _D, _D, _D, _D, _D, _D, _D],
    5:    [60, 70, 80, 90, _D, _D, _D, _D, _D, _D],
    10:   [50, 60, 70, 80, 90, _D, _D, _D, _D, _D],
    12:   [40, 50, 60, 70, 80, 90, _D, _D, _D, _D],
    15:   [30, 40, 50, 60, 70, 80, 90, _D, _D, _D],
    20:   [20, 30, 40, 50, 60, 70, 80, 90, _D, _D],
    50:   [10, 20, 30, 40, 50, 60, 70, 80, 90, _D],
    100:  [_X, 10, 20, 30, 40, 50, 60, 70, 80, 90],
    120:  [_X, _X, 10, 20, 30, 40, 50, 60, 70, 80],
    150:  [_X, _X, _X, 10, 20, 30, 40, 50, 60, 70],
    200:  [_X, _X, _X, _X, 10, 20, 30, 40, 50, 60],
    500:  [_X, _X, _X, _X, _X, 10, 20, 30, 40, 50],
    1000: [_X, _X, _X, _X, _X, _X, 10, 20, 30, 40],
    1200: [_X, _X, _X, _X, _X, _X, _X, 10, 20, 30],
    1500: [_X, _X, _X, _X, _X, _X, _X, _X, 10, 20],
    2000: [_X, _X, _X, _X, _X, _X, _X, _X, _X, 10],
    5000: [_X, _X, _X, _X, _X, _X, _X, _X, _X, _X],
}

# --------------------------------------------------------------------------
# SDB Against Squadron Combat Table
# columns are the SDBs' current strength, rows are the modified die roll
# (-2 .. 8); results are defence factors of battle damage.
# --------------------------------------------------------------------------
SDB_STRENGTH_COLUMNS = [10, 30, 60, 120, 180, 240, 300, 360, 420, 480]
SDB_VS_SQUADRON = {
    -2: [0, 0, 0, 0, 0, 0, 1, 2, 3, 4],
    -1: [0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
    0:  [0, 0, 0, 0, 1, 2, 3, 4, 5, 6],
    1:  [0, 0, 0, 1, 2, 3, 4, 5, 6, 7],
    2:  [0, 0, 1, 2, 3, 4, 5, 6, 7, 8],
    3:  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    4:  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    5:  [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    6:  [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    7:  [4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    8:  [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
}

# --------------------------------------------------------------------------
# Surface Bombing Table: columns are total bombardment factor, rows the
# tech-modified die roll; results are percentage losses.
# --------------------------------------------------------------------------
BOMBING_COLUMNS = SPACE_COLUMNS
SURFACE_BOMBING = {
    -2: [20, 30, 30, 40, 40, 50, 50, 50, 50, 50],
    -1: [20, 20, 30, 30, 40, 40, 50, 50, 50, 50],
    0:  [10, 20, 20, 30, 30, 40, 40, 50, 50, 50],
    1:  [10, 10, 20, 20, 30, 30, 40, 40, 50, 50],
    2:  [0, 10, 10, 20, 20, 30, 30, 40, 40, 50],
    3:  [0, 0, 10, 10, 20, 20, 30, 30, 40, 40],
    4:  [0, 0, 0, 10, 10, 20, 20, 30, 30, 40],
    5:  [0, 0, 0, 0, 10, 10, 20, 20, 30, 30],
    6:  [0, 0, 0, 0, 0, 10, 10, 20, 20, 30],
    7:  [0, 0, 0, 0, 0, 0, 10, 10, 20, 20],
    8:  [0, 0, 0, 0, 0, 0, 0, 10, 10, 20],
}

# --------------------------------------------------------------------------
# Troop Combat Results Table: columns are combat odds, rows are 2d6 modified
# by atmosphere; results are percentage losses to the fired-upon unit.
# --------------------------------------------------------------------------
ODDS_COLUMNS = ["1:100", "1:10", "1:5", "1:3", "1:2", "1:1.5", "1:1",
                "1.5:1", "2:1", "3:1", "5:1", "10:1", "100:1"]
ODDS_VALUES = [0.01, 0.1, 0.2, 1 / 3, 0.5, 1 / 1.5, 1.0,
               1.5, 2.0, 3.0, 5.0, 10.0, 100.0]
TROOP_CRT = {
    2:  [0, 10, 10, 20, 30, 40, 50, 60, 70, 90, _D, _D, _D],
    3:  [0, 0, 10, 10, 20, 30, 40, 50, 60, 80, 90, _D, _D],
    4:  [0, 0, 0, 10, 10, 20, 30, 40, 50, 70, 90, _D, _D],
    5:  [0, 0, 0, 0, 10, 10, 20, 30, 40, 60, 90, _D, _D],
    6:  [0, 0, 0, 0, 0, 10, 10, 20, 30, 50, 80, _D, _D],
    7:  [0, 0, 0, 0, 0, 0, 10, 10, 20, 40, 70, 90, _D],
    8:  [0, 0, 0, 0, 0, 0, 0, 10, 10, 30, 60, 80, _D],
    9:  [0, 0, 0, 0, 0, 0, 0, 0, 10, 20, 50, 70, _D],
    10: [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 40, 60, _D],
    11: [0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 30, 50, 90],
    12: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 20, 40, 80],
}

# --------------------------------------------------------------------------
# Tech level and atmosphere die roll modifiers
# --------------------------------------------------------------------------
def tech_modifier(tech_level: int) -> int:
    if tech_level >= 15:
        return 2
    if tech_level >= 13:
        return 0
    if tech_level >= 11:
        return -1
    if tech_level >= 9:
        return -2
    return -3


ATMOSPHERE_MODIFIER = {"breathable": 0, "tainted": -1, "vacuum": -2}

# Squadron quality column shifts (optional rule 9, Squadron Quality)
QUALITY_SHIFT = {
    ("imperial_standard", "zhodani_standard"): 1,
    ("imperial_standard", "zhodani_low"): 2,
    ("imperial_low", "zhodani_low"): 1,
    ("zhodani_standard", "imperial_low"): 1,
}

# Starport fuelling capacity (squadrons refuelled in zero time)
STARPORT_CAPACITY = {"A": 4, "B": 3, "C": 2, "D": 1, "E": 0, "X": 0}

# Required refuelling times: source -> streamlining class -> full phases needed
REFUEL_TIME = {
    "gas_giant": {"streamlined": 0, "partial": 0, "unstreamlined": 1},
    "ocean":     {"streamlined": 0, "partial": 1, "unstreamlined": 1},
    "base":      {"streamlined": 0, "partial": 0, "unstreamlined": 0},
}

# Fleet default planning factors (chart sheet, "Fleet Default Values")
FLEET_PLOTTING = {"imperial": 5, "zhodani": 4, "sword_worlds": 5, "vargr": 5}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def column_index(columns: Sequence[int], total: int) -> int:
    """Highest column index whose header does not exceed ``total``."""
    idx = bisect.bisect_right(columns, total) - 1
    return max(0, idx)


def split_attack(total: int, columns: Sequence[int] = SPACE_COLUMNS) -> list[int]:
    """Split a firing total into multiple attacks (rule 4, multiple attack)."""
    top = columns[-1]
    if total <= top:
        return [total]
    attacks = [top] * (total // top)
    remainder = total % top
    if remainder:
        attacks.append(remainder)
    return attacks


def space_result(die: int, total: int, column_shift: int = 0) -> int:
    """Battle damage from one space-combat attack of ``total`` factors."""
    die = min(6, max(1, die))
    damage = 0
    parts = split_attack(total)
    for i, part in enumerate(parts):
        idx = column_index(SPACE_COLUMNS, part)
        if i == 0:                      # quality shift applies once per round
            idx += column_shift
        while idx >= len(SPACE_COLUMNS):
            damage += SPACE_CRT[die][len(SPACE_COLUMNS) - 1]
            idx -= len(SPACE_COLUMNS)
        damage += SPACE_CRT[die][max(0, idx)]
    return damage


def squadron_vs_sdb_result(die: int, bombardment: int, sdb_full: int,
                           sdb_tech: int, passive: bool) -> int:
    """Percentage loss inflicted on an SDB unit."""
    modified = die - sdb_tech // 2 - (3 if passive else 0)
    row = min((r for r in SDB_ROWS if r >= sdb_full), default=SDB_ROWS[-1])
    losses = 0
    for i, part in enumerate(split_attack(bombardment)):
        idx = column_index(SPACE_COLUMNS, part) + modified
        idx = max(0, min(len(SPACE_COLUMNS) - 1, idx))
        value = SQUADRON_VS_SDB[row][idx]   # None is the table's dash
        losses += value or 0
    return min(DESTROYED, losses)


def sdb_vs_squadron_result(die: int, sdb_current: int, sdb_tech: int) -> int:
    """Battle damage inflicted on squadrons by an SDB unit."""
    row = max(-2, min(8, die + tech_modifier(sdb_tech)))
    damage = 0
    for part in split_attack(sdb_current, SDB_STRENGTH_COLUMNS):
        idx = column_index(SDB_STRENGTH_COLUMNS, part)
        damage += SDB_VS_SQUADRON[row][idx]
    return damage


def surface_bombing_result(die: int, bombardment: int, target_tech: int) -> int:
    row = max(-2, min(8, die + tech_modifier(target_tech)))
    losses = 0
    for part in split_attack(bombardment):
        idx = column_index(BOMBING_COLUMNS, part)
        losses += SURFACE_BOMBING[row][idx]
    return min(DESTROYED, losses)


def odds_index(firing: float, defending: float) -> int:
    """Combat odds column, always rounded down in favour of the defender."""
    if defending <= 0:
        return len(ODDS_COLUMNS) - 1
    ratio = firing / defending
    idx = 0
    for i, value in enumerate(ODDS_VALUES):
        if ratio + 1e-9 >= value:
            idx = i
    return idx


def troop_result(dice: int, firing: float, defending: float,
                 tech_shift: int, atmosphere: str) -> int:
    """Percentage loss inflicted on a troop, defence or SDB target."""
    idx = odds_index(firing, defending) + tech_shift
    idx = max(0, min(len(ODDS_COLUMNS) - 1, idx))
    roll = dice + ATMOSPHERE_MODIFIER.get(atmosphere, 0)
    roll = max(2, min(12, roll))
    return TROOP_CRT[roll][idx]

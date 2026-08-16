"""The combat tables, transcribed from the chart, and the rules for reading them.

Four tables and three modifier lists, all off the *Invasion: Earth* combat
tables sheet.  They are kept as literal data rather than as formulas because
they are not formulas -- the space combat columns step 1, 3, 6, 12, 18, 24 and
then by sixes, which is a designer's judgement about diminishing returns and
not a curve anyone would derive.

Every table is read the same way, and the rule is stated once here rather than
at each call site:

    A player uses the column that most closely approximates without exceeding
    his total attack factor.

So 29 attack factors fire on the 24 column, and a strength below the first
column has no column at all.  The chart's own closing note governs modified
rolls: "Whenever a die roll is modified above the highest number on a table or
below the lowest number on the table, the roll is treated as being the highest
or lowest number, respectively."
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# space combat
# --------------------------------------------------------------------------
#: Columns of the space combat table: attack -- squadrons firing on squadrons.
SPACE_ATTACK_COLUMNS = (1, 3, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72,
                        78, 84, 90)

#: ``SPACE_ATTACK[die]`` is the battle damage by column.  ``None`` is the
#: chart's dash: no effect.  Damage is "the (minimum) number of defense factors
#: that must be eliminated".
SPACE_ATTACK: dict[int, tuple] = {
    1: (None, None, None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14),
    2: (None, None, None, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    3: (None, None, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
    4: (None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),
    5: (None, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17),
    6: (1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18),
}

#: Columns of the space combat table: bombardment -- every attack by a squadron
#: on a SDB wing, and every attack by a SDB wing or PD unit on a squadron.
#: Note the leading 0 column: a bombardment factor of zero still rolls.
SPACE_BOMBARD_COLUMNS = (0, 1, 3, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66,
                         72, 78, 84)

SPACE_BOMBARD: dict[int, tuple] = {
    1: (None, None, None, None, None, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6),
    2: (None, None, None, None, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7),
    3: (None, None, None, None, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7),
    4: (None, None, None, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8),
    5: (None, None, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8),
    6: (None, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9),
}

# --------------------------------------------------------------------------
# surface bombardment
# --------------------------------------------------------------------------
#: Columns of the surface bombardment table -- naval units and PD units firing
#: on surface units.  Results are percentage losses.
SURFACE_BOMBARD_COLUMNS = (1, 3, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72)

#: Rolls run from -2, because the tech level and assault-from-space modifiers
#: are both negative and stack.
SURFACE_BOMBARD: dict[int, tuple] = {
    -2: (20, 30, 30, 40, 40, 50, 50, 50, 50, 50, 50, 50, 50, 50),
    -1: (20, 20, 30, 30, 40, 40, 50, 50, 50, 50, 50, 50, 50, 50),
    0: (10, 20, 20, 30, 30, 40, 40, 50, 50, 50, 50, 50, 50, 50),
    1: (10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 50, 50, 50, 50),
    2: (None, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 50, 50, 50),
    3: (None, None, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 50, 50),
    4: (None, None, None, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50, 50),
    5: (None, None, None, None, 10, 10, 20, 20, 30, 30, 40, 40, 50, 50),
    6: (None, None, None, None, None, 10, 10, 20, 20, 30, 30, 40, 40, 50),
}

# --------------------------------------------------------------------------
# troop combat
# --------------------------------------------------------------------------
#: Combat odds, as ratios, lowest first.  The chart writes them 1:100 through
#: 100:1; they are kept as floats so an odds ratio can be looked up by
#: comparison, and as strings so a log can print what the chart prints.
TROOP_ODDS = ((1, 100), (1, 10), (1, 5), (1, 3), (1, 2), (2, 3), (1, 1),
              (3, 2), (2, 1), (3, 1), (5, 1), (10, 1), (100, 1))

#: The chart writes the 2:3 column as "1:1 1/2" and the 3:2 column as "1 1/2:1".
TROOP_ODDS_LABELS = ("1:100", "1:10", "1:5", "1:3", "1:2", "1:1½", "1:1",
                     "1½:1", "2:1", "3:1", "5:1", "10:1", "100:1")

#: ``TROOP_COMBAT[dice]`` by odds column.  ``None`` is a dash, ``"d"`` is the
#: chart's d: "the attacked unit has suffered 100% losses and is destroyed".
TROOP_COMBAT: dict[int, tuple] = {
    2: (None, 10, 10, 20, 30, 40, 50, 60, 70, 90, "d", "d", "d"),
    3: (None, None, 10, 10, 20, 30, 40, 50, 60, 80, 90, "d", "d"),
    4: (None, None, None, 10, 10, 20, 30, 40, 50, 70, 90, "d", "d"),
    5: (None, None, None, None, 10, 10, 20, 30, 40, 60, 90, "d", "d"),
    6: (None, None, None, None, None, 10, 10, 20, 30, 50, 80, "d", "d"),
    7: (None, None, None, None, None, None, 10, 10, 20, 40, 70, 90, "d"),
    8: (None, None, None, None, None, None, None, 10, 10, 30, 60, 80, "d"),
    9: (None, None, None, None, None, None, None, None, 10, 20, 50, 70, "d"),
    10: (None, None, None, None, None, None, None, None, None, 10, 40, 60, "d"),
    11: (None, None, None, None, None, None, None, None, None, 10, 30, 50, 90),
    12: (None, None, None, None, None, None, None, None, None, None, 10, 40, 80),
}

# --------------------------------------------------------------------------
# modifiers
# --------------------------------------------------------------------------
#: Tech level modifiers, for the *bombarded* unit on the surface bombardment
#: table.  A high-tech unit is harder to hurt from orbit.
TECH_MODIFIER = {14: 0, 13: 0, 12: -1, 11: -1}

#: Assault from space modifiers, applied on top of the tech modifier when a
#: unit is bombarded while landing or evacuating.  Jump troops and marines are
#: equipped for it; everyone else is caught in the open.
ASSAULT_MODIFIER = {"jump": 0, "marine": 0, "other": -3}

#: Rule 5's weather: "whenever surface combat occurs in a hex that is in winter
#: or has winter climate, each attack on the troop combat table for that combat
#: is modified by -1".
WINTER_MODIFIER = -1

#: Rule 5's guerrillas: an attack on a guerrilla in hiding is modified by +3.
GUERRILLA_HIDING_MODIFIER = 3

#: Attacking a SDB wing that is still hiding in the oceans shifts the
#: bombardment column three to the left.
HIDDEN_SDB_COLUMN_SHIFT = -3


# --------------------------------------------------------------------------
def column_index(columns, strength: float) -> int:
    """The column to use for a strength: the largest that does not exceed it.

    Returns ``-1`` when the strength is below the first column, which on the
    space attack table means the attack cannot be made at all.  The bombardment
    table starts at 0, so it always has a column.
    """
    index = -1
    for i, threshold in enumerate(columns):
        if strength >= threshold:
            index = i
        else:
            break
    return index


def clamp_roll(table: dict, roll: int) -> int:
    """Hold a modified roll inside the table, as the chart's own note requires."""
    lo, hi = min(table), max(table)
    return max(lo, min(hi, roll))


def space_attack(strength: float, roll: int) -> int:
    """Battle damage from a squadron group firing on a squadron group."""
    col = column_index(SPACE_ATTACK_COLUMNS, strength)
    if col < 0:
        return 0
    result = SPACE_ATTACK[clamp_roll(SPACE_ATTACK, roll)][col]
    return 0 if result is None else result


def space_bombard(strength: float, roll: int, column_shift: int = 0) -> int:
    """Battle damage from a bombardment attack, with an optional column shift.

    The shift is how an attack on a SDB wing still hiding in the oceans is
    resolved: three columns left, and "any attack shifted below the 0 column is
    resolved using the 0 column".
    """
    col = column_index(SPACE_BOMBARD_COLUMNS, strength)
    col = max(0, col + column_shift)
    col = min(col, len(SPACE_BOMBARD_COLUMNS) - 1)
    result = SPACE_BOMBARD[clamp_roll(SPACE_BOMBARD, roll)][col]
    return 0 if result is None else result


def surface_bombard(strength: float, roll: int) -> int:
    """Percentage loss to a surface unit from bombardment."""
    col = column_index(SURFACE_BOMBARD_COLUMNS, strength)
    if col < 0:
        return 0
    result = SURFACE_BOMBARD[clamp_roll(SURFACE_BOMBARD, roll)][col]
    return 0 if result is None else result


def odds_index(firing: float, defending: float) -> int:
    """The odds column for an attack, rounded down in the defender's favour.

    The rulebook's worked examples: "13 factors firing on a unit with a current
    strength of 5 is 13:5, which rounds down to 2:1.  25 factors firing on a
    unit with a current strength of 2 is 25:2, which rounds down to 10:1", and
    "treat odds less than 1:100 as 1:100".
    """
    if defending <= 0:
        return len(TROOP_ODDS) - 1
    best = 0
    for i, (num, den) in enumerate(TROOP_ODDS):
        if firing * den >= defending * num:
            best = i
    return best


def troop_combat(firing: float, defending: float, roll: int,
                 column_shift: int = 0):
    """Percentage loss to the attacked unit, or ``"d"`` for destroyed.

    ``column_shift`` carries the tech level difference: "if the tech level
    difference is positive, the odds are shifted the indicated number of
    columns to the right ... any attack shifted above 100:1 or below 1:100 is
    treated as 100:1 or 1:100".
    """
    col = odds_index(firing, defending) + column_shift
    col = max(0, min(col, len(TROOP_ODDS) - 1))
    result = TROOP_COMBAT[clamp_roll(TROOP_COMBAT, roll)][col]
    return 0 if result is None else result


def current_strength(full: float, losses: int) -> float:
    """A unit's strength after percentage losses.

    The percentage loss table on the chart is just this arithmetic tabulated
    for the five printed unit sizes, so the arithmetic is what is kept.  Losses
    "are always based on the full (printed) strength of the unit", which is why
    they are additive and why this reads the full strength rather than the
    current one.
    """
    return max(0.0, full * (100 - min(100, losses)) / 100.0)

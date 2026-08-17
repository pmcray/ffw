"""The state of a game of *Invasion: Earth*, and fast copying of it.

The shape follows ``ffw.state`` deliberately: frozen counter classes shared
between copies, mutable per-counter records holding position and damage, and a
``clone`` that copies only what changes so a search agent can roll the game
forward without paying for a deep copy.  The two games are different enough in
their rules that sharing an engine would be a fiction, but an agent that has
learned to read one board should be able to read the other, and that starts
with the states looking alike.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from . import hexmap, oob, terra

IMPERIAL = oob.IMPERIAL
SOLOMANI = oob.SOLOMANI

#: The four space boxes, in the order the rules resolve combat in them:
#: "deep space, far orbit, and close orbit".  ``OUT_SYSTEM`` is the Imperial
#: marshalling ground, reachable only by jump.
OUT_SYSTEM = "out_system"
DEEP_SPACE = "deep_space"
FAR_ORBIT = "far_orbit"
CLOSE_ORBIT = "close_orbit"

SPACE_BOXES = (OUT_SYSTEM, DEEP_SPACE, FAR_ORBIT, CLOSE_ORBIT)

#: In-system movement is a line: "a unit moving between the close orbit and
#: deep space boxes must first enter the far orbit box as part of its move".
IN_SYSTEM = (DEEP_SPACE, FAR_ORBIT, CLOSE_ORBIT)

#: Luna sits in the far orbit box and can be used as an advance base.
LUNA = "luna"

#: Missions a naval unit in close orbit may be given.
BOMBARDMENT = "bombardment"
OVERWATCH = "overwatch"

#: Six turns to a quarter, four quarters to a year; "the game begins with the
#: April 1 turn", which is turn 1 of the second quarter.
TURNS_PER_QUARTER = 6
FIRST_TURN_MONTH = 4

#: Rule 8: "this occurs when there are fewer than 10 urban hexes remaining to
#: the Solomani player".
VICTORY_URBAN_THRESHOLD = 10

#: The level of victory table, worst-to-best for the Imperium so an index into
#: it is an ordering.
VICTORY_ORDER = [
    "solomani major victory", "solomani marginal victory", "draw",
    "imperial marginal victory", "imperial major victory",
    "imperial decisive victory",
]


def enemy_of(side: str) -> str:
    return SOLOMANI if side == IMPERIAL else IMPERIAL


# --------------------------------------------------------------------------
@dataclass
class NavalUnit:
    """A squadron or SDB wing in play.

    ``location`` is a space box name, or a hex id when a bombarding unit is
    placed "on the map in the hex containing the surface unit it is to
    bombard", or a hex id when a SDB wing is hiding in the oceans.

    Naval units "do not take percentage losses.  A naval unit is totally
    eliminated upon receiving battle damage", so there is no damage field --
    only whether it is still here.
    """
    uid: int
    cls: oob.NavalClass
    navy: str
    location: str | int
    hidden: bool = False
    mission: str | None = None
    cargo: list[int] = field(default_factory=list)     # surface unit uids
    #: whether the unit has jumped this turn, for "may make only one jump"
    jumped: bool = False
    #: whether the unit has left close orbit at any point this turn, which
    #: halves its bombardment factor
    manoeuvred: bool = False
    #: a dummy SDB wing -- rule 5's mock turtles
    dummy: bool = False

    @property
    def side(self) -> str:
        return self.navy

    @property
    def is_sdb(self) -> bool:
        return self.cls.kind == "SDB"

    def bombard_factor(self, halved: bool = False) -> float:
        """Bombardment strength, halved where the rules halve it.

        "When halving factors, retain all fractions (thus, half of 7 is 3 1/2)."
        """
        value = float(self.cls.bombard)
        if self.manoeuvred:
            value /= 2.0
        if halved:
            value /= 2.0
        return value

    def copy(self) -> "NavalUnit":
        new = NavalUnit.__new__(NavalUnit)
        new.__dict__.update(self.__dict__)
        new.cargo = list(self.cargo)
        return new


@dataclass
class SurfaceUnit:
    """A troop unit, PD unit or guerrilla in play.

    ``losses`` is a percentage of the *printed* strength, because "these losses
    are always based on the full (printed) strength of the unit.  Thus,
    multiple percentage losses are additive."
    """
    uid: int
    cls: oob.SurfaceClass
    navy: str
    location: str | int          # hex id, a space box while carried, or LUNA
    losses: int = 0
    carrier: int | None = None   # naval unit uid while being transported
    hiding: bool = False         # guerrillas only
    landed_turn: int = -1        # "may not move ... on the turn it is unloaded"

    @property
    def side(self) -> str:
        return self.navy

    @property
    def current(self) -> float:
        """Strength after losses -- what the unit actually fires with."""
        return max(0.0, self.cls.factor * (100 - min(100, self.losses)) / 100.0)

    @property
    def dead(self) -> bool:
        return self.losses >= 100

    def combat_strength(self, supplied: bool = True) -> float:
        """Current strength as it counts in surface combat.

        Armour doubles it, elite doubles it again -- "an elite armor unit would
        have its current strength doubled twice" -- and a mercenary past half
        strength fires at half.  Being out of supply does not reduce what a
        unit fires with; it stops it firing at all, which is the caller's
        business, and halves it when fired *upon*.
        """
        value = self.current
        if self.cls.armor:
            value *= 2
        if self.cls.elite:
            value *= 2
        if self.cls.mercenary and self.losses > 50:
            value /= 2.0
        return value

    def defence_strength(self, supplied: bool = True) -> float:
        """Strength as the target of an attack.

        Armour and elite double here too -- the rule says "during surface
        combat resolution" without distinguishing attack from defence -- and
        rule 5's supply paragraph halves a unit "when fired upon (attacked)".
        """
        value = self.current
        if self.cls.armor:
            value *= 2
        if self.cls.elite:
            value *= 2
        if not supplied:
            value /= 2.0
        return max(value, 0.0)

    def copy(self) -> "SurfaceUnit":
        new = SurfaceUnit.__new__(SurfaceUnit)
        new.__dict__.update(self.__dict__)
        return new


@dataclass
class Base:
    """An Imperial base: the only Imperial source of supply.

    "A base has no combat factor and is automatically destroyed when there are
    enemy surface units but no friendly surface units in the base's hex."  It
    is bombarded as a tech level 14 unit.
    """
    uid: int
    location: str | int
    losses: int = 0
    carrier: int | None = None

    navy: str = IMPERIAL

    @property
    def side(self) -> str:
        return IMPERIAL

    @property
    def dead(self) -> bool:
        return self.losses >= 100

    def copy(self) -> "Base":
        new = Base.__new__(Base)
        new.__dict__.update(self.__dict__)
        return new


# --------------------------------------------------------------------------
class Geometry:
    """Cached adjacency and range bands, shared between clones.

    The grid never changes during a game, so every clone can point at the same
    tables.  ``ffw`` learned this the expensive way: the profiler put a million
    calls to a distance function at the top of the list.
    """

    def __init__(self, terrain: dict):
        self.terrain = terrain
        self.cells = tuple(sorted(terrain))
        self.adjacency = {c: hexmap.neighbours(c) for c in self.cells}
        self._bands: dict[tuple[int, int], tuple[int, ...]] = {}
        self.land = frozenset(c for c in self.cells if terrain[c] in terra.LAND)
        self.deep_sea = frozenset(c for c in self.cells
                                  if terrain[c] in terra.DEEP_SEA)
        self.urban = frozenset(c for c in self.cells
                               if terrain[c] == terra.URBAN)
        self.starports = frozenset(c for c in self.cells
                                   if terrain[c] == terra.STARPORT)
        self.lat_lon = {c: hexmap.lat_lon(c) for c in self.cells}

    def adjacent(self, cell) -> tuple[int, ...]:
        return self.adjacency.get(cell, ())

    def within(self, origin: int, radius: int) -> tuple[int, ...]:
        key = (origin, radius)
        band = self._bands.get(key)
        if band is None:
            band = hexmap.within(origin, radius)
            self._bands[key] = band
        return band

    def distance(self, a, b) -> int:
        return hexmap.distance(a, b)


def _copy_units(units: dict) -> dict:
    return {uid: unit.copy() for uid, unit in units.items()}


# --------------------------------------------------------------------------
@dataclass
class GameState:
    turn: int = 1
    naval: dict[int, NavalUnit] = field(default_factory=dict)
    surface: dict[int, SurfaceUnit] = field(default_factory=dict)
    bases: dict[int, Base] = field(default_factory=dict)
    pools: dict = field(default_factory=dict)
    #: Solomani replacement points, accumulated one per ungarrisoned urban hex
    #: per turn; Imperial replacement points, bought a wave at a time.
    replacement_points: dict = field(default_factory=lambda: {IMPERIAL: 0,
                                                              SOLOMANI: 0})
    #: "The number of waves taken should also be recorded ... for use when
    #: determining victory": each one costs a victory point.
    waves_taken: int = 0
    #: Which hexes the Imperial player currently garrisons.
    garrisoned: set = field(default_factory=set)
    game_over: bool = False
    result: str | None = None
    abandoned: bool = False
    log: list = field(default_factory=list)
    options: dict = field(default_factory=dict)
    _next_uid: int = 1

    geometry: Geometry | None = None
    starport_names: dict = field(default_factory=dict)

    # -- identity ------------------------------------------------------
    def new_uid(self) -> int:
        uid = self._next_uid
        self._next_uid += 1
        return uid

    def note(self, text: str) -> None:
        self.log.append("T%02d %s" % (self.turn, text))

    # -- the calendar --------------------------------------------------
    @property
    def quarter(self) -> int:
        """Which quarter of the year the current turn falls in, from 1.

        The game starts on the April 1 turn, which is the first turn of the
        second quarter, so turn 1 is quarter 1 of the game rather than of the
        calendar.  Victory is checked "during the special turn at the end of
        each quarter year", so what matters is how many complete quarters have
        been played.
        """
        return (self.turn - 1) // TURNS_PER_QUARTER + 1

    @property
    def quarters_elapsed(self) -> int:
        return (self.turn - 1) // TURNS_PER_QUARTER

    def is_quarter_end(self) -> bool:
        return self.turn % TURNS_PER_QUARTER == 0

    def month(self) -> int:
        """Calendar month of the current turn; two turns to a month."""
        return (FIRST_TURN_MONTH - 1 + (self.turn - 1) // 2) % 12 + 1

    def in_winter(self, cell) -> bool:
        """Whether a hex is in winter, by season and by climate.

        "The northern hemisphere is in winter from December through March.  All
        non-clear, non-urban hexes entirely north of the 30 degree north line
        are affected by winter."  Plus the climate rule: tundra, ice sheet and
        sea ice hexes are in winter whatever the season.
        """
        if not isinstance(cell, int):
            return False
        terrain = self.geometry.terrain.get(cell)
        if terrain in terra.ALWAYS_WINTER:
            return True
        if terrain in (terra.CLEAR, terra.URBAN, terra.STARPORT):
            return False
        lat, _lon = self.geometry.lat_lon[cell]
        month = self.month()
        if lat > 30 and (month >= 12 or month <= 3):
            return True
        if lat < -30 and 6 <= month <= 9:
            return True
        return False

    # -- lookups -------------------------------------------------------
    def naval_at(self, where, side: str | None = None,
                 include_hidden: bool = False) -> list:
        out = []
        for unit in self.naval.values():
            if unit.location != where:
                continue
            if unit.hidden and not include_hidden:
                continue
            if side is not None and unit.side != side:
                continue
            out.append(unit)
        return out

    def surface_at(self, cell, side: str | None = None) -> list:
        return [u for u in self.surface.values()
                if u.location == cell and u.carrier is None and not u.dead
                and (side is None or u.side == side)]

    def bases_at(self, cell) -> list:
        return [b for b in self.bases.values()
                if b.location == cell and b.carrier is None and not b.dead]

    def strength_at(self, cell, side: str) -> float:
        return sum(u.current for u in self.surface_at(cell, side))

    def zone_of_control(self, side: str) -> set:
        """Every hex in the zone of control of a unit of ``side``.

        "Corps-sized, army-sized, and all planetary defense surface units have
        zones of control.  A unit's zone of control exists in the hex it
        occupies and extends into the six hexes surrounding it."
        """
        zoc = set()
        for unit in self.surface.values():
            if unit.side != side or unit.carrier is not None or unit.dead:
                continue
            if not (unit.cls.planetary_defense
                    or unit.cls.size in ("corps", "army")):
                continue
            cell = unit.location
            if not isinstance(cell, int):
                continue
            zoc.add(cell)
            zoc.update(self.geometry.adjacent(cell))
        return zoc

    # -- the scoreboard ------------------------------------------------
    def solomani_urban(self) -> list:
        """Urban hexes still the Solomani player's: ungarrisoned by the Imperium.

        Starports count for supply and SDB construction but not as urban hexes
        -- the rulebook counts the two separately, and puts the victory
        condition on urban hexes alone.
        """
        return [c for c in self.geometry.urban if c not in self.garrisoned]

    def objective_met(self) -> bool:
        """Whether the Imperium has achieved what it came for.

        "This occurs when there are fewer than 10 urban hexes remaining to the
        Solomani player."  Everything in rule 8 hangs off this sentence.
        """
        return len(self.solomani_urban()) < VICTORY_URBAN_THRESHOLD

    def victory_points(self) -> int:
        """Rule 8's victory point total, from the Imperial player's side.

        "When this occurs, the play of the game ceases, and the Imperial player
        is awarded 10 victory points", modified -1 per quarter the game lasted,
        -1 per replacement wave taken, +1 if all Solomani non-guerrilla surface
        units are eliminated, and +1 if all Solomani naval units are.

        *When this occurs* is a condition and not a formality.  The ten points
        are the reward for reducing the Solomani below ten cities; an invasion
        that runs out of turns having taken none of them has not earned them,
        and awarding them anyway makes the clock the only term in the score --
        every game ends in the same level of victory no matter how it was
        played, which is exactly what this file used to do.
        """
        points = 10 if self.objective_met() else 0
        points -= max(1, self.quarters_elapsed)
        points -= self.waves_taken
        if not any(u.side == SOLOMANI and not u.cls.guerrilla and not u.dead
                   for u in self.surface.values()):
            points += 1
        if not any(u.side == SOLOMANI for u in self.naval.values()):
            points += 1
        return points

    def victory_level(self, points: int | None = None) -> str:
        """The level of victory table."""
        total = self.victory_points() if points is None else points
        if total >= 7:
            return "imperial decisive victory"
        if total >= 4:
            return "imperial major victory"
        if total >= 1:
            return "imperial marginal victory"
        if total >= -1:
            return "draw"
        if total >= -3:
            return "solomani marginal victory"
        return "solomani major victory"

    def margin(self) -> float:
        """A continuous score, Imperial-positive, for comparing play.

        The victory point total is a small integer and the level table has six
        steps, so neither resolves the difference between two doctrines that
        both fail to end the war.  What separates them is how far the invasion
        got, so the margin is how many urban hexes the Imperium has taken,
        carrying the victory point total as a much heavier term for the games
        that actually finish.

        This is the same distinction ``ffw`` had to learn: the level is what
        the rules settle on, the margin is what compares play.
        """
        taken = len(self.geometry.urban) - len(self.solomani_urban())
        return taken + 10.0 * self.victory_points() if self.game_over else taken

    # -- copying -------------------------------------------------------
    def clone(self, keep_log: bool = False) -> "GameState":
        """A copy cheap enough for a search agent to make thousands of.

        Counter classes and the geometry are frozen and shared; everything that
        a turn can change is copied.
        """
        new = GameState.__new__(GameState)
        new.turn = self.turn
        new.naval = _copy_units(self.naval)
        new.surface = _copy_units(self.surface)
        new.bases = _copy_units(self.bases)
        new.pools = {k: (list(v) if isinstance(v, list) else v)
                     for k, v in self.pools.items()}
        new.replacement_points = dict(self.replacement_points)
        new.waves_taken = self.waves_taken
        new.garrisoned = set(self.garrisoned)
        new.game_over = self.game_over
        new.result = self.result
        new.abandoned = self.abandoned
        new.log = list(self.log) if keep_log else []
        new.options = self.options
        new._next_uid = self._next_uid
        new.geometry = self.geometry
        new.starport_names = self.starport_names
        return new

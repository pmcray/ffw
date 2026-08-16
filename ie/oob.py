"""The counters, transcribed from the counter inventory and order of battle.

Naval units are rated attack-bombardment-defense; surface units carry a combat
factor and a tech level.  Both are frozen records shared between every copy of
a game state -- a counter's printed values never change, only where it is and
how much of it is left.

The order of battle chart is the authority on how many of each there are, and
the counter inventory is the authority on what each one is rated.  Where the
two can be checked against each other they agree, and ``tests`` checks them.
"""

from __future__ import annotations

from dataclasses import dataclass

IMPERIAL = "imperial"
SOLOMANI = "solomani"


# --------------------------------------------------------------------------
@dataclass(frozen=True)
class NavalClass:
    """A squadron or SDB wing, rated attack-bombardment-defense.

    ``kind`` is the counter's ID code: BR battle, CR cruiser, SR scout, AR
    assault/transport, SDB system defense boat wing.  ``jump`` records whether
    the unit can make an interstellar jump, which in this game only Imperial
    squadrons can -- the Solomani squadrons are "non-starships left at Terra
    for repair", and SDB wings are "inherently incapable".
    """
    name: str
    kind: str
    attack: int
    bombard: int
    defense: int
    jump: bool = False
    colonial: bool = False

    @property
    def capacity(self) -> int:
        """Transport capability, in combat factors, from the unit ID chart."""
        return TRANSPORT_CAPACITY.get(self.kind, 0)


@dataclass(frozen=True)
class SurfaceClass:
    """A troop unit or PD unit.

    ``factor`` is the printed combat factor and ``tech`` the tech level, which
    together are what the counter shows: ``1C-14`` is 100 factors at TL 14.
    ``bombard`` is non-zero only for PD units, which "have instrinsic troop
    garrisons" as well as guns.
    """
    name: str
    size: str                      # regiment, brigade, division, corps, army
    factor: int
    tech: int
    armor: bool = False
    elite: bool = False
    jump: bool = False
    marine: bool = False
    commando: bool = False
    guerrilla: bool = False
    mercenary: bool = False
    planetary_defense: bool = False
    bombard: int = 0
    mobile: bool = True

    @property
    def assault_class(self) -> str:
        """Which row of the assault from space modifiers table applies."""
        if self.jump or self.marine:
            return "jump" if self.jump else "marine"
        return "other"


#: Transport capability table, from the unit identification chart.  "The table
#: specifies the maximum number of combat factors a given naval unit may
#: transport at one time.  For example, a battle squadron may carry no more
#: than one 20-factor surface unit."
TRANSPORT_CAPACITY = {"AR": 600, "BR": 20, "CR": 5, "SR": 0, "SDB": 0}

#: A base "may be transported by a naval unit, requiring 100 factors of
#: transport capability".
BASE_TRANSPORT_COST = 100

#: "A player may have no more than 1000 combat factors of surface units in a
#: single hex", counted on current strengths.
STACKING_LIMIT = 1000

#: Grav vehicles: "a movement allowance of 10 movement points per friendly
#: movement phase ... the cost to enter a hex is 1 movement point".
MOVEMENT_POINTS = 10

#: "this supply line may be up to 5 hexes in length".
SUPPLY_RANGE = 5

#: A PD unit or surfaced SDB wing may fire "only if those units are within
#: three hexes".
PD_RANGE = 3


def _n(name, kind, a, b, d, jump=False, colonial=False):
    return NavalClass(name, kind, a, b, d, jump, colonial)


# --------------------------------------------------------------------------
# Imperial naval units -- all jump-capable; only the Imperium can leave Sol
# --------------------------------------------------------------------------
IMPERIAL_NAVAL = [
    (_n("BR 2-3-5", "BR", 2, 3, 5, jump=True), 1),
    (_n("BR 3-3-6", "BR", 3, 3, 6, jump=True), 2),
    (_n("BR 4-4-4", "BR", 4, 4, 4, jump=True), 6),
    (_n("BR 3-4-6", "BR", 3, 4, 6, jump=True), 1),
    (_n("BR 4-4-6", "BR", 4, 4, 6, jump=True), 2),
    (_n("BR 5-4-8", "BR", 5, 4, 8, jump=True), 2),
    (_n("CR 1-1-3", "CR", 1, 1, 3, jump=True), 1),
    (_n("CR 3-2-4", "CR", 3, 2, 4, jump=True), 2),
    (_n("CR 2-2-6", "CR", 2, 2, 6, jump=True), 3),
    (_n("CR 3-2-8", "CR", 3, 2, 8, jump=True), 1),
    (_n("AR 0-0-4", "AR", 0, 0, 4, jump=True), 4),      # transport squadrons
    (_n("SR 0-3-4", "SR", 0, 3, 4, jump=True), 2),
    (_n("SR 0-6-6", "SR", 0, 6, 6, jump=True), 1),
    (_n("BR 2-2-2 colonial", "BR", 2, 2, 2, jump=True, colonial=True), 4),
    (_n("BR 2-0-3 colonial", "BR", 2, 0, 3, jump=True, colonial=True), 1),
    (_n("BR 3-1-4 colonial", "BR", 3, 1, 4, jump=True, colonial=True), 1),
    (_n("CR 1-2-5 colonial", "CR", 1, 2, 5, jump=True, colonial=True), 4),
    (_n("CR 0-2-6 colonial", "CR", 0, 2, 6, jump=True, colonial=True), 2),
    (_n("CR 2-2-6 colonial", "CR", 2, 2, 6, jump=True, colonial=True), 2),
]

# --------------------------------------------------------------------------
# Solomani naval units -- no jump drive at all
# --------------------------------------------------------------------------
SOLOMANI_NAVAL = [
    (_n("BR 6-2-6", "BR", 6, 2, 6), 2),
    (_n("BR 3-0-4", "BR", 3, 0, 4), 2),
    (_n("BR 3-1-5", "BR", 3, 1, 5), 1),
    (_n("BR 1-3-2", "BR", 1, 3, 2), 1),
    (_n("BR 1-2-5", "BR", 1, 2, 5), 1),
    (_n("BR 2-2-4", "BR", 2, 2, 4), 1),
]

#: The SDB wings, thirty-four of them, from the counter inventory.
SOLOMANI_SDB = [
    (_n("SDB 0-3-5", "SDB", 0, 3, 5), 6),
    (_n("SDB 0-3-4", "SDB", 0, 3, 4), 2),
    (_n("SDB 0-2-5", "SDB", 0, 2, 5), 6),
    (_n("SDB 0-2-4", "SDB", 0, 2, 4), 9),
    (_n("SDB 0-2-3", "SDB", 0, 2, 3), 2),
    (_n("SDB 0-2-2", "SDB", 0, 2, 2), 3),
    (_n("SDB 0-1-3", "SDB", 0, 1, 3), 6),
]

#: "The Solomani player may use two blank counters as dummy SDB wings."
DUMMY_SDB = 2


def _s(name, size, factor, tech, **kw):
    return SurfaceClass(name, size, factor, tech, **kw)


# --------------------------------------------------------------------------
# Imperial surface units
# --------------------------------------------------------------------------
IMPERIAL_TROOPS = [
    (_s("lift infantry army", "army", 500, 14), 2),
    (_s("grav tank corps", "corps", 100, 14, armor=True), 1),
    (_s("grav tank corps", "corps", 100, 13, armor=True), 1),
    (_s("armored lift infantry corps", "corps", 100, 14, armor=True), 3),
    (_s("armored lift infantry corps", "corps", 100, 13, armor=True), 1),
    (_s("lift infantry corps", "corps", 100, 14), 3),
    (_s("lift infantry corps", "corps", 100, 13), 2),
    (_s("lift infantry corps", "corps", 100, 12), 3),
    (_s("armored grav cavalry corps", "corps", 100, 13, armor=True), 1),
    (_s("grav cavalry corps", "corps", 100, 13), 1),
    (_s("grav jump division", "division", 20, 14, jump=True), 1),
    (_s("grav jump division", "division", 20, 14, jump=True, elite=True), 1),
    (_s("grav tank division", "division", 20, 14, armor=True), 1),
    (_s("grav tank division", "division", 20, 14, armor=True, elite=True), 1),
    (_s("armored grav cavalry division", "division", 20, 14, armor=True), 1),
    (_s("lift infantry division", "division", 20, 14), 1),
    (_s("lift infantry division", "division", 20, 13), 2),
    (_s("grav jump brigade", "brigade", 10, 14, jump=True), 4),
    (_s("grav jump brigade", "brigade", 10, 13, jump=True), 2),
]

IMPERIAL_COLONIAL = [
    (_s("colonial armored lift infantry corps", "corps", 100, 14, armor=True), 1),
    (_s("colonial lift infantry corps", "corps", 100, 12), 5),
    (_s("colonial lift infantry corps", "corps", 100, 11), 1),
    (_s("colonial armored grav cavalry corps", "corps", 100, 11, armor=True), 1),
    (_s("colonial grav cavalry corps", "corps", 100, 11), 1),
    (_s("colonial grav tank corps", "corps", 100, 13, armor=True), 1),
    (_s("colonial armored lift infantry division", "division", 20, 12, armor=True), 1),
    (_s("colonial grav tank division", "division", 20, 11, armor=True, elite=True), 1),
    (_s("colonial lift infantry division", "division", 20, 12), 2),
    (_s("colonial lift infantry brigade", "brigade", 10, 14), 1),
    (_s("colonial lift infantry brigade", "brigade", 10, 13), 1),
]

IMPERIAL_MARINES = [
    (_s("star marine division", "division", 20, 14, marine=True), 1),
    (_s("star marine regiment", "regiment", 5, 14, marine=True), 1),
    (_s("star marine regiment", "regiment", 5, 14, marine=True, elite=True), 2),
    (_s("star marine regiment", "regiment", 5, 13, marine=True), 1),
    (_s("star marine regiment", "regiment", 5, 12, marine=True), 1),
]

IMPERIAL_MERCENARIES = [
    (_s("mercenary lift infantry division", "division", 20, 14, mercenary=True), 1),
    (_s("mercenary armored lift infantry division", "division", 20, 12,
        armor=True, mercenary=True), 1),
    (_s("mercenary lift infantry brigade", "brigade", 10, 13, mercenary=True), 1),
    (_s("mercenary lift infantry brigade", "brigade", 10, 12, mercenary=True), 1),
    (_s("mercenary armored grav cavalry brigade", "brigade", 10, 13,
        armor=True, mercenary=True), 1),
    (_s("mercenary grav jump regiment", "regiment", 5, 14,
        jump=True, elite=True, mercenary=True), 1),
]

#: "The Imperial player has a number of bases available, which are used as
#: logistical and adminstrative headquarters."  They have no combat factor and
#: are the only Imperial source of supply.
IMPERIAL_BASES = 10

# --------------------------------------------------------------------------
# Solomani surface units
# --------------------------------------------------------------------------
SOLOMANI_TROOPS = [
    # The four field armies are named for the continents they hold -- NA, SA,
    # AF, AS -- and two of them are a tech level behind the other two.
    (_s("lift infantry army", "army", 500, 14), 2),      # SA, AS
    (_s("lift infantry army", "army", 500, 13), 2),      # NA, AF
    (_s("grav tank corps", "corps", 100, 14, armor=True), 1),
    (_s("lift infantry corps", "corps", 100, 14), 1),
    (_s("armored grav cavalry corps", "corps", 100, 14, armor=True), 1),
    (_s("armored lift infantry corps", "corps", 100, 14, armor=True), 1),
    (_s("lift infantry corps", "corps", 100, 13), 1),
    (_s("armored lift infantry corps", "corps", 100, 13, armor=True), 1),
    (_s("armored lift infantry corps", "corps", 100, 13, armor=True), 1),
    (_s("grav cavalry corps", "corps", 100, 13), 1),
    (_s("lift infantry corps", "corps", 100, 12), 3),
    (_s("lift infantry corps", "corps", 100, 11), 1),
    (_s("grav jump division", "division", 20, 14, jump=True), 2),
    (_s("grav tank division", "division", 20, 14, armor=True), 1),
    (_s("armored lift cavalry division", "division", 20, 14, armor=True), 1),
    (_s("armored lift infantry division", "division", 20, 13, armor=True), 2),
    (_s("lift infantry division", "division", 20, 13), 4),
    (_s("lift infantry division", "division", 20, 12), 2),
    (_s("lift cavalry division", "division", 20, 12), 2),
    (_s("lift infantry division", "division", 20, 11), 2),
    (_s("elite grav tank regiment", "regiment", 10, 14, armor=True, elite=True), 2),
    (_s("elite armored grav cavalry regiment", "regiment", 10, 14,
        armor=True, elite=True), 1),
    (_s("armored lift infantry regiment", "regiment", 10, 14, armor=True), 3),
    (_s("elite grav commando regiment", "regiment", 10, 14,
        commando=True, elite=True), 2),
]

#: PD units: "collections of energy weapons and missiles capable of engaging
#: naval units bombarding the surface".  Static, so ``mobile=False``; the
#: bombardment factors are the ones printed on the counters.
SOLOMANI_PD = [
    (_s("planetary defense corps", "corps", 100, 14,
        planetary_defense=True, bombard=9, mobile=False), 3),
    (_s("planetary defense division", "division", 20, 14,
        planetary_defense=True, bombard=6, mobile=False), 7),
    (_s("planetary defense division", "division", 20, 13,
        planetary_defense=True, bombard=5, mobile=False), 6),
    (_s("planetary defense division", "division", 20, 12,
        planetary_defense=True, bombard=4, mobile=False), 4),
    (_s("planetary defense regiment", "regiment", 10, 14,
        planetary_defense=True, bombard=3, mobile=False), 4),
]

#: "The Solomani player has eight guerrilla units available to enter play",
#: four at the end of each of the first two quarters.
SOLOMANI_GUERRILLAS = [
    (_s("guerrilla corps", "corps", 100, 13, guerrilla=True), 8),
]
GUERRILLA_WAVE = 4

#: Rule 6: two transport squadrons must be withdrawn on turn 2.
WITHDRAWAL_TURN = 2
WITHDRAWAL_COUNT = 2
WITHDRAWAL_KIND = "AR"


def expand(spec) -> list:
    """``[(cls, count), ...] -> [cls, cls, ...]``, as in the FFW order of battle."""
    out = []
    for item, count in spec:
        out.extend([item] * count)
    return out


def total(spec) -> int:
    return sum(count for _cls, count in spec)


def factors(spec) -> int:
    return sum(cls.factor * count for cls, count in spec)

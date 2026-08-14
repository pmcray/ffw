"""Game state for Fifth Frontier War."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable

from . import hexmap
from .oob import SquadronClass, TroopClass
from .tables import FLEET_PLOTTING, STARPORT_CAPACITY

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

#: the two players.  The Zhodani player also runs the Sword Worlds and Vargr.
IMPERIAL, ZHODANI = "imperial", "zhodani"
NAVIES = ("imperial", "zhodani", "sword_worlds", "vargr")

#: which navies each player commands
NAVIES_OF = {IMPERIAL: ("imperial",), ZHODANI: ("zhodani", "sword_worlds", "vargr")}


def side_of(navy: str) -> str:
    return IMPERIAL if navy == "imperial" else ZHODANI


# --------------------------------------------------------------------------
@dataclass
class World:
    name: str
    hex: str
    starport: str
    tech_level: int
    sdb: int
    defense_battalions: int
    water: bool
    atmosphere: str
    naval_base: bool
    scout_base: bool
    military_base: bool
    gas_giant: bool
    zone: str | None
    owner: str                      # original allegiance
    subsector_capital: bool

    # mutable wartime state
    control: str | None = None      # imperial / zhodani / None
    sdb_losses: int = 0
    defense_losses: int = 0
    zone_lifted: bool = False       # red zone opened by Zhodani entry

    @property
    def base(self) -> bool:
        return self.naval_base or self.scout_base or self.military_base

    @property
    def sdb_current(self) -> int:
        if self.sdb_losses >= 100:
            return 0
        return self.sdb * (100 - self.sdb_losses) // 100

    @property
    def defense_current(self) -> int:
        if self.defense_losses >= 100:
            return 0
        return self.defense_battalions * (100 - self.defense_losses) // 100

    @property
    def victory_points(self) -> float:
        """VPs for a player controlling this world from outside its border."""
        if self.owner == "neutral":
            return self.tech_level / 2.0
        return self.tech_level * (2 if self.subsector_capital else 1)

    @property
    def starport_capacity(self) -> int:
        return STARPORT_CAPACITY.get(self.starport, 0)

    @property
    def original_control(self) -> str | None:
        """Which player held this world at the start of the game."""
        return {"imperial": IMPERIAL, "zhodani": ZHODANI,
                "sword_worlds": ZHODANI, "vargr": ZHODANI}.get(self.owner)

    def garrison_required(self, side: str | None = None) -> int:
        """Troop factors needed to hold the world (rule 5, Control).

        The rule reads "to maintain control, the world must be garrisoned ...
        if the garrison is under strength, control of the world reverts to its
        original owner", so it binds on a world a player has *taken*.  A world
        cannot revert to the side that already owns it, and its own defence
        battalions are its garrison, so passing ``side`` returns zero for that
        side's own territory.  Called without ``side`` it returns the strength a
        conqueror would need, which is what the map display wants.
        """
        if side is not None and self.original_control == side:
            return 0
        if self.defense_battalions == 0:
            return 1
        return max(1, self.defense_battalions // 100)


@dataclass
class WorldMap:
    worlds: dict[str, World]
    xboat_routes: list[tuple[str, str]]
    entry_hexes: dict[str, str]

    @classmethod
    def load(cls, path: str | None = None) -> "WorldMap":
        path = path or os.path.join(DATA, "worlds.json")
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        worlds = {}
        for w in raw["worlds"]:
            world = World(
                name=w["name"], hex=w["hex"], starport=w["starport"],
                tech_level=w["tech_level"], sdb=w["sdb"],
                defense_battalions=w["defense_battalions"], water=w["water"],
                atmosphere=w["atmosphere"], naval_base=w["naval_base"],
                scout_base=w["scout_base"], military_base=w["military_base"],
                gas_giant=w["gas_giant"], zone=w["zone"], owner=w["owner"],
                subsector_capital=w["subsector_capital"],
            )
            world.control = {"imperial": IMPERIAL,
                             "zhodani": ZHODANI,
                             "sword_worlds": ZHODANI,
                             "vargr": ZHODANI}.get(w["owner"])
            worlds[world.hex] = world
        routes = [tuple(r) for r in raw["xboat_routes"]]
        return cls(worlds, routes, raw["entry_hexes"])

    def __iter__(self) -> Iterable[World]:
        return iter(self.worlds.values())

    def get(self, hex_id: str) -> World | None:
        return self.worlds.get(hex_id)

    def bounds(self) -> tuple[int, int, int, int]:
        cols = [hexmap.parse(h)[0] for h in self.worlds]
        rows = [hexmap.parse(h)[1] for h in self.worlds]
        return min(cols), max(cols), min(rows), max(rows)

    def xboat_neighbours(self, hex_id: str) -> list[str]:
        out = []
        for a, b in self.xboat_routes:
            if a == hex_id:
                out.append(b)
            elif b == hex_id:
                out.append(a)
        return out

    def xboat_reach(self, start: str, jump: int = 4) -> set[str]:
        """Worlds reachable along the xboat network within one jump-4 move."""
        seen, frontier = {start}, [start]
        out = set()
        while frontier:
            here = frontier.pop()
            for nxt in self.xboat_neighbours(here):
                if nxt in seen:
                    continue
                seen.add(nxt)
                if hexmap.distance(start, nxt) <= jump:
                    out.add(nxt)
                    frontier.append(nxt)
        return out


# --------------------------------------------------------------------------
@dataclass
class Squadron:
    uid: int
    cls: SquadronClass
    navy: str
    location: str                 # hex id, or a reinforcement box name
    reduced: bool = False
    fleet: int | None = None
    refuelled: bool = True
    refuel_progress: int = 0      # movement phases spent refuelling
    troops: list[int] = field(default_factory=list)   # troop uids aboard
    colonial: bool = False
    black_globe: bool = False

    @property
    def side(self) -> str:
        return side_of(self.navy)

    @property
    def attack(self) -> int:
        return self.cls.reduced[0] if self.reduced else self.cls.attack

    @property
    def bombard(self) -> int:
        return self.cls.reduced[1] if self.reduced else self.cls.bombard

    @property
    def defense(self) -> int:
        return self.cls.reduced[2] if self.reduced else self.cls.defense

    @property
    def jump(self) -> int:
        return self.cls.jump

    @property
    def capacity(self) -> int:
        return self.cls.capacity(self.reduced)

    def damage_value(self) -> int:
        """Defence factors removed by reducing (or destroying) this squadron."""
        if self.reduced or self.cls.reduced is None:
            return self.defense
        return self.cls.defense - self.cls.reduced[2]


@dataclass
class TroopUnit:
    uid: int
    cls: TroopClass
    navy: str
    location: str                 # hex id of the world it stands on
    losses: int = 0               # percentage
    aboard: int | None = None     # squadron uid when embarked
    guerrilla: bool = False
    overt: bool = False

    @property
    def side(self) -> str:
        return side_of(self.navy)

    @property
    def full_factor(self) -> int:
        return self.cls.factor

    @property
    def current(self) -> int:
        if self.losses >= 100:
            return 0
        return max(0, self.cls.factor * (100 - self.losses) // 100)

    def combat_strength(self, world: World | None = None) -> float:
        """Current strength after doubling for armour and elite status."""
        value = float(self.current)
        if self.cls.armor:
            value *= 2
        if self.cls.elite:
            value *= 2
        if self.cls.mercenary and self.losses > 50:
            value = float(int(value / 2))
        return value

    def tech_level(self, world: World | None = None) -> int:
        if self.guerrilla and world is not None:
            return world.tech_level
        return self.cls.tech_level


@dataclass
class Admiral:
    uid: int
    navy: str
    precedence: int
    planning: int
    tactical: int
    name: str
    location: str
    fleet: int | None = None
    aboard: bool = False
    warrant: bool = False

    @property
    def side(self) -> str:
        return side_of(self.navy)

    @property
    def effective_precedence(self) -> int:
        return 0 if self.warrant else self.precedence


@dataclass
class Fleet:
    uid: int
    name: str
    navy: str
    location: str
    squadrons: list[int] = field(default_factory=list)
    admiral: int | None = None
    plot: dict[int, tuple[str, str]] = field(default_factory=dict)  # turn -> (kind, hex)
    active: bool = False

    @property
    def side(self) -> str:
        return side_of(self.navy)


# --------------------------------------------------------------------------
@dataclass
class GameState:
    world_map: WorldMap
    turn: int = 1
    squadrons: dict[int, Squadron] = field(default_factory=dict)
    troops: dict[int, TroopUnit] = field(default_factory=dict)
    admirals: dict[int, Admiral] = field(default_factory=dict)
    fleets: dict[int, Fleet] = field(default_factory=dict)
    pools: dict = field(default_factory=dict)         # off-map counters by group
    replacement_points: dict = field(default_factory=lambda:
                                     {IMPERIAL: {"squadron": 0, "troop": 0},
                                      ZHODANI: {"squadron": 0, "troop": 0}})
    secret_base: str | None = None
    secret_base_revealed: bool = False
    game_over: bool = False
    result: str | None = None
    log: list[str] = field(default_factory=list)
    _next_uid: int = 1
    options: dict = field(default_factory=lambda: {
        "black_globes": True, "jump_troops": True,
        "squadron_quality": True, "surprise_attack": False})

    # -- identity helpers --------------------------------------------------
    def new_uid(self) -> int:
        uid = self._next_uid
        self._next_uid += 1
        return uid

    # -- queries -----------------------------------------------------------
    def squadrons_at(self, hex_id: str, side: str | None = None) -> list[Squadron]:
        return [s for s in self.squadrons.values()
                if s.location == hex_id and (side is None or s.side == side)]

    def troops_at(self, hex_id: str, side: str | None = None,
                  landed_only: bool = True) -> list[TroopUnit]:
        out = []
        for t in self.troops.values():
            if t.location != hex_id or t.losses >= 100:
                continue
            if landed_only and t.aboard is not None:
                continue
            if side is not None and t.side != side:
                continue
            out.append(t)
        return out

    def fleets_at(self, hex_id: str, side: str | None = None) -> list[Fleet]:
        return [f for f in self.fleets.values()
                if f.active and f.location == hex_id
                and (side is None or f.side == side)]

    def admirals_at(self, hex_id: str, side: str | None = None) -> list[Admiral]:
        return [a for a in self.admirals.values()
                if a.location == hex_id and (side is None or a.side == side)]

    def senior_admiral(self, hex_id: str, side: str) -> Admiral | None:
        present = self.admirals_at(hex_id, side)
        if not present:
            return None
        own = self.squadrons_at(hex_id, side)
        def eligible(a: Admiral) -> bool:
            if not own:
                return True
            same = sum(1 for s in own if s.navy == a.navy)
            return same >= len(own) - same
        pool = [a for a in present if eligible(a)] or present
        return min(pool, key=lambda a: a.effective_precedence)

    def tactical_factor(self, hex_id: str, side: str) -> int:
        admiral = self.senior_admiral(hex_id, side)
        if admiral is None:
            return 0
        # a flag officer on the surface loses one for communication lag
        return admiral.tactical - (0 if admiral.aboard else 1)

    # -- control -----------------------------------------------------------
    def enemy_of(self, side: str) -> str:
        return ZHODANI if side == IMPERIAL else IMPERIAL

    def controls(self, hex_id: str, side: str) -> bool:
        world = self.world_map.get(hex_id)
        return world is not None and world.control == side

    def garrison_strength(self, hex_id: str, side: str) -> int:
        return sum(t.current for t in self.troops_at(hex_id, side)
                   if not (t.guerrilla and not t.overt))

    def update_control(self, hex_id: str) -> None:
        """Re-derive control of a world after combat (rule 5)."""
        world = self.world_map.get(hex_id)
        if world is None:
            return
        original = world.original_control
        for side in (IMPERIAL, ZHODANI):
            if world.control == side:
                continue
            enemy = self.enemy_of(side)
            if self.squadrons_at(hex_id, enemy):
                continue
            if world.control == enemy and world.sdb_current > 0:
                continue
            if self.troops_at(hex_id, enemy):
                continue
            if world.control == enemy and world.defense_current > 0:
                continue
            if self.garrison_strength(hex_id, side) >= world.garrison_required(side):
                world.control = side
            elif (world.atmosphere == "vacuum" and world.defense_current == 0
                  and any(s.attack > 0 for s in self.squadrons_at(hex_id, side))):
                # rule 5: a surrendered airless world may be garrisoned by any
                # squadron with an attack factor greater than zero
                world.control = side
        # a world whose garrison lapses reverts to its original owner
        if world.control is not None and world.control != original:
            if self.garrison_strength(hex_id, world.control) < world.garrison_required(world.control):
                if not self._surrendered_garrison(world):
                    world.control = original

    def _surrendered_garrison(self, world: World) -> bool:
        """A vacuum world that has surrendered may be held by a warship."""
        if world.atmosphere != "vacuum" or world.control is None:
            return False
        return any(s.attack > 0 for s in self.squadrons_at(world.hex, world.control))

    # -- victory -----------------------------------------------------------
    def victory_points(self) -> dict[str, float]:
        points = {IMPERIAL: 0.0, ZHODANI: 0.0}
        for world in self.world_map:
            if world.control is None:
                continue
            origin = world.owner
            holder = world.control
            if origin == "neutral":
                points[holder] += world.tech_level / 2.0
            elif origin == "imperial" and holder == ZHODANI:
                points[ZHODANI] += world.victory_points
            elif origin != "imperial" and holder == IMPERIAL:
                points[IMPERIAL] += world.victory_points
        # Rule 8: Esalin's co-dominium status makes it an enemy world for both
        # players, so it is worth its full tech level rather than the half a
        # neutral world earns.  The loop above has already paid the half.
        esalin = next((w for w in self.world_map if w.name == "Esalin"), None)
        if esalin is not None and esalin.control is not None:
            points[esalin.control] += esalin.tech_level / 2.0
        vargr = [w for w in self.world_map if w.owner == "vargr"]
        if vargr and all(w.control == IMPERIAL for w in vargr):
            points[IMPERIAL] += 15
        sw = [w for w in self.world_map if w.owner == "sword_worlds"]
        if sw and all(w.control == IMPERIAL for w in sw):
            points[IMPERIAL] += 15
        return points

    def victory_margin(self) -> float:
        pts = self.victory_points()
        return pts[ZHODANI] - pts[IMPERIAL]

    def victory_level(self, margin: float | None = None) -> str:
        m = self.victory_margin() if margin is None else margin
        if m <= -301:
            return "imperial automatic victory"
        if m <= -201:
            return "imperial decisive victory"
        if m <= -101:
            return "imperial major victory"
        if m <= -51:
            return "imperial marginal victory"
        if m <= 50:
            return "stalemate"
        if m <= 100:
            return "zhodani marginal victory"
        if m <= 200:
            return "zhodani major victory"
        if m <= 300:
            return "zhodani decisive victory"
        return "zhodani automatic victory"

    def note(self, message: str) -> None:
        self.log.append("T%02d %s" % (self.turn, message))

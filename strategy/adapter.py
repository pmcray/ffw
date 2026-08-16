"""Descriptions of each game that training code can drive without knowing which.

A ``GameAdapter`` answers, for its own game, the questions
``strategy.features`` asks: what counts as an objective, how much force a side
has, where that force is, and how the scoreboard reads.  Nothing else is
shared, and nothing in ``ffw`` or ``ie`` imports this -- the dependency points
one way, so either game can be worked on without touching the other.

The two adapters are deliberately short.  If one of them starts growing game
logic, that is a sign the shared layer is reaching too far into a rulebook it
does not own.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GameAdapter:
    """The interface.  See ``FFW`` and ``IE`` for the two implementations."""

    name: str
    sides: tuple
    scheduled_length: int
    map_diameter: float

    def new_game(self, seed=None):                     # pragma: no cover
        raise NotImplementedError

    def play(self, state, agents, max_turns, rng=None):  # pragma: no cover
        raise NotImplementedError

    def enemy_of(self, side: str) -> str:
        a, b = self.sides
        return b if side == a else a

    # -- the questions strategy.features asks --------------------------
    def objectives(self, state) -> list:               # pragma: no cover
        raise NotImplementedError

    def objectives_held(self, state, side) -> list:    # pragma: no cover
        raise NotImplementedError

    def initial_objective_share(self, side) -> float:  # pragma: no cover
        raise NotImplementedError

    def force_strength(self, state, side) -> float:    # pragma: no cover
        raise NotImplementedError

    def initial_force(self, side) -> float:            # pragma: no cover
        raise NotImplementedError

    def force_posture(self, state, side):              # pragma: no cover
        """``(forward, reserve, largest_stack)`` in strength."""
        raise NotImplementedError

    def reach(self, state, side):                      # pragma: no cover
        """``(mean distance to objectives, mean distance to the enemy)``."""
        raise NotImplementedError

    def supply_health(self, state, side) -> float:     # pragma: no cover
        raise NotImplementedError

    def normalised_margin(self, state, side) -> float:  # pragma: no cover
        """The victory margin scaled to roughly [-1, 1], ``side``-positive."""
        raise NotImplementedError

    def normalised_level(self, state, side) -> float:  # pragma: no cover
        """Position on the victory table scaled to [-1, 1], ``side``-positive."""
        raise NotImplementedError


# ==========================================================================
class _FFW(GameAdapter):
    """*Fifth Frontier War*: two navies contesting a subsector of worlds."""

    def __init__(self):
        from ffw import hexmap                          # noqa: F401
        from ffw.state import IMPERIAL, ZHODANI
        super().__init__(name="ffw", sides=(IMPERIAL, ZHODANI),
                         scheduled_length=60, map_diameter=32.0)
        self._initial = None

    def new_game(self, seed=None):
        from ffw.engine import new_game
        return new_game(seed=seed)

    def play(self, state, agents, max_turns, rng=None):
        from ffw.engine import play
        imperial, zhodani = self.sides
        return play(state, agents[imperial], agents[zhodani],
                    max_turns=max_turns)

    # -- territory -----------------------------------------------------
    def objectives(self, state) -> list:
        """Every world worth victory points, which here is every world."""
        return [w.hex for w in state.world_map]

    def objectives_held(self, state, side) -> list:
        return [w.hex for w in state.world_map if w.control == side]

    def initial_objective_share(self, side) -> float:
        # roughly half the subsector each at the start; measured once and
        # cached so the feature is a change from the opening, not from zero
        if self._initial is None:
            state = self.new_game(seed=0)
            total = max(1, len(self.objectives(state)))
            self._initial = {s: len(self.objectives_held(state, s)) / total
                             for s in self.sides}
        return self._initial[side]

    # -- force ---------------------------------------------------------
    def force_strength(self, state, side) -> float:
        naval = sum(sq.attack + sq.defense for sq in state.squadrons.values()
                    if sq.side == side)
        troops = sum(t.current for t in state.troops.values()
                     if t.side == side and not t.guerrilla)
        return naval + troops / 10.0

    def initial_force(self, side) -> float:
        if not hasattr(self, "_force0"):
            state = self.new_game(seed=0)
            self._force0 = {s: self.force_strength(state, s)
                            for s in self.sides}
        return self._force0[side]

    def force_posture(self, state, side):
        from ffw.engine import REINFORCEMENT_BOXES
        forward = reserve = 0.0
        by_hex: dict = {}
        for sq in state.squadrons.values():
            if sq.side != side:
                continue
            weight = sq.attack + sq.defense
            if sq.location in REINFORCEMENT_BOXES or not sq.location:
                reserve += weight
            else:
                forward += weight
                by_hex[sq.location] = by_hex.get(sq.location, 0.0) + weight
        largest = max(by_hex.values(), default=0.0)
        return forward, reserve, largest

    def reach(self, state, side):
        from ffw import hexmap
        enemy = self.enemy_of(side)
        mine = [sq.location for sq in state.squadrons.values()
                if sq.side == side and sq.location in state.world_map.worlds]
        theirs = [sq.location for sq in state.squadrons.values()
                  if sq.side == enemy and sq.location in state.world_map.worlds]
        targets = [w.hex for w in state.world_map if w.control == enemy]
        if not mine:
            return self.map_diameter, self.map_diameter
        to_objectives = (min(hexmap.distance(mine[0], t) for t in targets)
                         if targets else self.map_diameter)
        to_enemy = (min(hexmap.distance(mine[0], t) for t in theirs)
                    if theirs else self.map_diameter)
        return to_objectives, to_enemy

    def supply_health(self, state, side) -> float:
        """Fuel, which is this game's supply: what fraction can refuel here."""
        from ffw.engine import can_refuel_at
        squadrons = [sq for sq in state.squadrons.values()
                     if sq.side == side and sq.location in state.world_map.worlds]
        if not squadrons:
            return 1.0
        ok = sum(1 for sq in squadrons
                 if can_refuel_at(state, side, sq.location, [sq.cls.refuel]))
        return ok / len(squadrons)

    # -- the scoreboard ------------------------------------------------
    def normalised_margin(self, state, side) -> float:
        from ffw.state import ZHODANI
        margin = state.victory_margin() / 300.0
        return margin if side == ZHODANI else -margin

    def normalised_level(self, state, side) -> float:
        from ffw.state import ZHODANI
        from ffw.training import VICTORY_ORDER, level_index
        middle = (len(VICTORY_ORDER) - 1) / 2.0
        result = state.result or state.victory_level()
        value = (level_index(result) - middle) / middle
        return value if side == ZHODANI else -value


# ==========================================================================
class _IE(GameAdapter):
    """*Invasion: Earth*: one fleet assaulting one planet."""

    def __init__(self):
        from ie.state import IMPERIAL, SOLOMANI
        super().__init__(name="ie", sides=(IMPERIAL, SOLOMANI),
                         scheduled_length=24, map_diameter=21.0)
        self._initial = None

    def new_game(self, seed=None):
        from ie.engine import new_game
        return new_game(seed=seed)

    def play(self, state, agents, max_turns, rng=None):
        from ie.engine import play
        imperial, solomani = self.sides
        return play(state, agents[imperial], agents[solomani],
                    max_turns=max_turns, rng=rng)

    # -- territory -----------------------------------------------------
    def objectives(self, state) -> list:
        """The sixty-one urban hexes: the only thing either side scores on."""
        return sorted(state.geometry.urban)

    def objectives_held(self, state, side) -> list:
        from ie.state import IMPERIAL
        if side == IMPERIAL:
            return sorted(state.geometry.urban & state.garrisoned)
        return state.solomani_urban()

    def initial_objective_share(self, side) -> float:
        from ie.state import IMPERIAL
        return 0.0 if side == IMPERIAL else 1.0

    # -- force ---------------------------------------------------------
    def force_strength(self, state, side) -> float:
        naval = sum(u.cls.attack + u.cls.defense for u in state.naval.values()
                    if u.side == side)
        troops = sum(u.current for u in state.surface.values()
                     if u.side == side and not u.dead and not u.cls.guerrilla)
        return naval + troops / 10.0

    def initial_force(self, side) -> float:
        if not hasattr(self, "_force0"):
            state = self.new_game(seed=0)
            self._force0 = {s: self.force_strength(state, s)
                            for s in self.sides}
        return self._force0[side]

    def force_posture(self, state, side):
        """Forward is on the surface; reserve is in orbit or out-system."""
        forward = reserve = 0.0
        by_hex: dict = {}
        for unit in state.surface.values():
            if unit.side != side or unit.dead:
                continue
            weight = unit.current / 10.0
            if unit.carrier is not None or not isinstance(unit.location, int):
                reserve += weight
            else:
                forward += weight
                by_hex[unit.location] = by_hex.get(unit.location, 0.0) + weight
        for unit in state.naval.values():
            if unit.side != side:
                continue
            weight = unit.cls.attack + unit.cls.defense
            if isinstance(unit.location, int):
                forward += weight
            else:
                reserve += weight
        largest = max(by_hex.values(), default=0.0)
        return forward, reserve, largest

    def reach(self, state, side):
        geo = state.geometry
        enemy = self.enemy_of(side)
        mine = [u.location for u in state.surface.values()
                if u.side == side and not u.dead and isinstance(u.location, int)]
        theirs = [u.location for u in state.surface.values()
                  if u.side == enemy and not u.dead
                  and isinstance(u.location, int)]
        if not mine:
            return self.map_diameter, self.map_diameter
        anchor = mine[0]
        targets = self.objectives_held(state, enemy)
        to_objectives = (min(geo.distance(anchor, t) for t in targets)
                         if targets else self.map_diameter)
        to_enemy = (min(geo.distance(anchor, t) for t in theirs)
                    if theirs else self.map_diameter)
        return to_objectives, to_enemy

    def supply_health(self, state, side) -> float:
        """Rule 5's supply trace: what fraction of the army is in supply."""
        from ie.engine import Engine
        from ie.agents import Agent
        ashore = [u for u in state.surface.values()
                  if u.side == side and not u.dead
                  and isinstance(u.location, int)]
        if not ashore:
            return 1.0
        engine = Engine(state, Agent(self.sides[0]), Agent(self.sides[1]))
        return sum(1 for u in ashore if engine._in_supply(u)) / len(ashore)

    # -- the scoreboard ------------------------------------------------
    def normalised_margin(self, state, side) -> float:
        from ie.state import IMPERIAL
        total = max(1, len(state.geometry.urban))
        taken = total - len(state.solomani_urban())
        value = 2.0 * taken / total - 1.0
        return value if side == IMPERIAL else -value

    def normalised_level(self, state, side) -> float:
        from ie.state import IMPERIAL, VICTORY_ORDER
        middle = (len(VICTORY_ORDER) - 1) / 2.0
        result = state.result or state.victory_level()
        value = (VICTORY_ORDER.index(result) - middle) / middle
        return value if side == IMPERIAL else -value


FFW = _FFW
IE = _IE

_CACHE: dict = {}


def adapters() -> dict:
    """One instance of each adapter, built lazily and reused.

    They cache opening measurements -- initial force, initial territory share
    -- which cost a game setup each, so they are worth keeping.
    """
    if not _CACHE:
        _CACHE["ffw"] = _FFW()
        _CACHE["ie"] = _IE()
    return dict(_CACHE)


def adapter_for(name: str) -> GameAdapter:
    return adapters()[name]

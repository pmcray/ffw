"""Every decision *Invasion: Earth* gives a player, each with a workable default.

The engine never decides anything a player would decide.  It asks, through the
methods below, and each one has a default that plays legally and passively, so
a subclass can override only the decision it cares about and inherit the rest.

That is the same contract ``ffw.agents.base`` offers, and deliberately so: an
agent written against one game's decision list should look recognisable against
the other's.  The decisions differ -- there is no jump plotting here and no
orbital bombardment there -- but the shape is the same, and the shared
``strategy`` package depends on it.
"""

from __future__ import annotations

import random

from ..state import (CLOSE_ORBIT, DEEP_SPACE, FAR_ORBIT, IMPERIAL, LUNA,
                     OUT_SYSTEM, SOLOMANI, GameState)


class Agent:
    """The interface.  Subclass and override what you want to change."""

    name = "agent"

    def __init__(self, side: str, seed: int | None = None,
                 label: str | None = None):
        self.side = side
        self.rng = random.Random(seed)
        self.label = label or self.name

    def begin_game(self, state: GameState, side: str, engine) -> None:
        """Called once, after setup, before the first turn."""

    def begin_turn(self, state: GameState, side: str, engine) -> None:
        """Called at the top of each turn."""

    # ---- 1. initial segment ------------------------------------------
    def consolidate(self, state: GameState, side: str, engine) -> list:
        """Which damaged troop units to merge.

        Returns ``[(donor_uid, recipient_uid), ...]``.  Consolidation removes
        the donor and gives its current strength to the recipient, which "may
        never be raised in strength above their printed strengths".  The
        default consolidates nothing, because merging is only ever right when
        the player knows what the spare factors are for.
        """
        return []

    def emergency_replacements(self, state: GameState, side: str, engine) -> int:
        """How many emergency replacement waves (Imperial) or points (Solomani).

        Emergency replacements are deliberately bad value -- the Solomani pay
        two points per factor instead of one, and an Imperial emergency wave
        buys 50 points where a regular one buys 100 -- and every Imperial wave
        costs a victory point at the end.  The default takes none.
        """
        return 0

    # ---- 2. space segment --------------------------------------------
    def naval_moves(self, state: GameState, side: str, engine) -> dict:
        """Where each naval unit goes this phase.

        Returns ``{uid: destination}``, where a destination is a space box.  A
        unit omitted stays put.  Illegal moves are dropped by the engine rather
        than raising, because an agent should not have to re-derive the
        movement rules to be safe.
        """
        return {}

    def take_another_naval_phase(self, state: GameState, side: str,
                                 engine) -> bool:
        """Whether to take another naval phase after a battle was fought.

        "The player may continue to take a new naval phase after each naval
        phase in which a space battle was fought."  The default declines; a
        player who keeps going is committing to another round of losses.
        """
        return False

    def hide_in_deep_space(self, state: GameState, side: str, engine) -> set:
        """Which naval units in deep space go into (or stay in) hiding."""
        return set()

    def disengage(self, state: GameState, side: str, box: str, engine) -> bool:
        """Whether to break off a space battle at the end of a round.

        The Solomani player is offered this first each round; if he declines,
        the Imperial player is offered it.  Disengaging from close orbit costs
        a free round of fire from anything in far orbit.
        """
        return False

    # ---- 3. space-surface segment ------------------------------------
    def missions(self, state: GameState, side: str, engine) -> dict:
        """Missions for naval units in close orbit.

        Returns ``{uid: (mission, target)}`` where mission is ``BOMBARDMENT``
        with a hex as target, or ``OVERWATCH`` with ``None``.  A unit omitted
        stays unassigned.
        """
        return {}

    def sdb_surface(self, state: GameState, side: str, engine) -> set:
        """Which hidden SDB wings come out of hiding this segment."""
        return set()

    def sdb_relocate(self, state: GameState, side: str, engine) -> dict:
        """Where SDB wings that stayed hidden move, one hex at most."""
        return {}

    def overwatch_targets(self, state: GameState, side: str, surfaced: list,
                          engine) -> list:
        """The order in which surfaced SDB wings are attacked.

        "Each SDB wing that comes out of hiding is attacked separately; the
        order in which these attacks are resolved is chosen by the Imperial
        player."
        """
        return list(surfaced)

    def defence_fire(self, state: GameState, side: str, engine) -> dict:
        """What each PD unit and surfaced SDB wing shoots at.

        Returns ``{uid: ("naval", box_or_hex) | ("surface", target_uid)}``.
        "Each unit capable of firing during this phase may fire once", so a PD
        unit chooses between the fleet overhead and the troops in front of it.
        """
        return {}

    def landings(self, state: GameState, side: str, engine) -> list:
        """Loading and unloading between close orbit and the surface.

        Returns a list of ``("load", unit_uid, carrier_uid)`` and
        ``("unload", unit_uid, cell)`` instructions.  Bases use the same verbs.
        """
        return []

    # ---- 4. surface segment ------------------------------------------
    def surface_moves(self, state: GameState, side: str, engine) -> dict:
        """Where each mobile surface unit goes.  ``{uid: destination_cell}``.

        The engine pathfinds within the ten movement points, charging the
        zone-of-control and occupied-hex surcharges, and refuses a move it
        cannot pay for.
        """
        return {}

    def allocate_fire(self, state: GameState, side: str, cell, engine) -> list:
        """How to divide fire in a surface combat.

        Returns ``[(firer_uid, target_uid, factors), ...]``.  "A firing unit
        may split its combat factor to fire at several units", each attack is
        resolved separately, and "a unit may be fired upon only once; total all
        factors firing upon a unit".  The default concentrates everything on
        the strongest enemy unit present, which is a real decision and not a
        good one.
        """
        firers = state.surface_at(cell, side)
        targets = state.surface_at(cell, state.enemy_of(side)
                                   if hasattr(state, "enemy_of") else
                                   (SOLOMANI if side == IMPERIAL else IMPERIAL))
        if not firers or not targets:
            return []
        target = max(targets, key=lambda u: u.current)
        return [(u.uid, target.uid, u.combat_strength()) for u in firers]

    def guerrilla_hiding(self, state: GameState, side: str, engine) -> set:
        """Which guerrilla units are hiding this combat phase.

        Hiding costs the unit its fire and buys a +3 on every attack against
        it, which on the troop combat table is most of a column.
        """
        return set()

    # ---- 5. the quarterly special turn -------------------------------
    def abandon_invasion(self, state: GameState, engine) -> bool:
        """Rule 8: the Imperial player may call the whole thing off.

        "The game ends upon this decision, and the Solomani player wins a major
        victory."  It is the worst result on the table, so the default is never
        -- but it is a real option, and an agent that is losing badly and still
        buying replacement waves is choosing something worse than a draw.
        """
        return False

    def spend_replacements(self, state: GameState, side: str, engine) -> None:
        """Spend accumulated replacement points at the quarter's end."""

    def replacement_waves(self, state: GameState, engine) -> int:
        """How many replacement waves the Imperial player takes this quarter.

        Each is 100 replacement points, one base, or three naval units -- and
        each costs a victory point at the end of the game.  The default takes
        none, which is the correct default precisely because the cost is real.
        """
        return 0

    def place_guerrillas(self, state: GameState, count: int, engine) -> list:
        """Where the Solomani player puts newly available guerrilla units.

        "No more than one guerrilla may be placed in a single hex during this
        initial placement, but the Solomani player may place the guerrilla
        units in any hexes on the map (including those garrisoned by Imperial
        units)."
        """
        return []

    def build_sdb(self, state: GameState, engine) -> list:
        """At which ungarrisoned starports the Solomani lay down SDB wings."""
        return []

    def deploy_sdb(self, state: GameState, ready: list, engine) -> dict:
        """Where a finished SDB wing goes: a deep sea hex, or close orbit."""
        return {}


class RandomAgent(Agent):
    """Legal moves chosen at random.  The floor every other agent is measured against."""

    name = "random"

    def naval_moves(self, state, side, engine):
        boxes = [DEEP_SPACE, FAR_ORBIT, CLOSE_ORBIT]
        moves = {}
        for unit in state.naval.values():
            if unit.side != side or unit.hidden or unit.is_sdb:
                continue
            if not isinstance(unit.location, str):
                continue
            if self.rng.random() < 0.5:
                moves[unit.uid] = self.rng.choice(
                    boxes + ([OUT_SYSTEM] if unit.cls.jump else []))
        return moves

    def surface_moves(self, state, side, engine):
        moves = {}
        for unit in state.surface.values():
            if (unit.side != side or unit.carrier is not None or unit.dead
                    or not unit.cls.mobile or not isinstance(unit.location, int)):
                continue
            if self.rng.random() < 0.5:
                options = [c for c in state.geometry.adjacent(unit.location)
                           if c in state.geometry.land]
                if options:
                    moves[unit.uid] = self.rng.choice(options)
        return moves

    def landings(self, state, side, engine):
        if side != IMPERIAL:
            return []
        out = []
        beaches = [c for c in state.geometry.land]
        for unit in state.surface.values():
            if unit.side != side or unit.carrier is None:
                continue
            carrier = state.naval.get(unit.carrier)
            if carrier is None or carrier.location != CLOSE_ORBIT:
                continue
            if self.rng.random() < 0.4 and beaches:
                out.append(("unload", unit.uid, self.rng.choice(beaches)))
        return out

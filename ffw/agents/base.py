"""The agent interface.

An agent is asked for a decision at each point in the sequence of play where
the rules give a player a choice.  Every hook has a workable default, so a new
agent only has to override the decisions it wants to make differently.
"""

from __future__ import annotations

import random

from .. import hexmap
from ..state import GameState, Squadron, TroopUnit, IMPERIAL, ZHODANI


class Agent:
    """Base class: a legal but unambitious player."""

    name = "agent"

    def __init__(self, side: str, seed: int | None = None):
        self.side = side
        self.rng = random.Random(seed)

    def __repr__(self) -> str:
        return "<%s %s>" % (type(self).__name__, self.side)

    # -- lifecycle ---------------------------------------------------------
    def begin_game(self, state: GameState, side: str, engine) -> None:
        pass

    # -- phase 1: replacements --------------------------------------------
    def spend_replacements(self, state: GameState, side: str, engine) -> None:
        """Rebuild reduced squadrons and refill damaged troops."""
        points = state.replacement_points[side]
        # one RP rebuilds a reduced squadron at a friendly base at home
        for sq in state.squadrons.values():
            if points["squadron"] <= 0:
                break
            if sq.side != side or not sq.reduced:
                continue
            world = state.world_map.get(sq.location)
            if world is None:
                if sq.location in ("imperial_reinforcements", "imperial_rimward",
                                   "zhodani_reinforcements", "zhodani_additional",
                                   "sword_worlds"):
                    sq.reduced = False
                    points["squadron"] -= 1
                continue
            if not world.base or world.control != side:
                continue
            # rule 6: rebuilding happens at a friendly base inside the borders
            # of the squadron's own state
            home = {"imperial": ("imperial",),
                    "zhodani": ("zhodani",),
                    "sword_worlds": ("sword_worlds",),
                    "vargr": ("vargr",)}[sq.navy]
            if world.owner not in home:
                continue
            if sq.cls.kind == "scout" and not world.naval_base and not world.scout_base:
                continue
            sq.reduced = False
            points["squadron"] -= 1
        # troop RPs top up units standing on a safe friendly world
        for troop in state.troops.values():
            if points["troop"] <= 0:
                break
            if troop.side != side or troop.losses == 0 or troop.guerrilla:
                continue
            if troop.cls.elite or troop.aboard is not None:
                continue
            world = state.world_map.get(troop.location)
            if world is None or world.control != side:
                continue
            if state.troops_at(troop.location, state.enemy_of(side)):
                continue
            cost_per = 2 if troop.cls.armor else 1
            missing = troop.full_factor - troop.current
            affordable = points["troop"] // cost_per
            healed = min(missing, affordable)
            if healed <= 0:
                continue
            points["troop"] -= healed * cost_per
            troop.losses = max(0, troop.losses - 100 * healed // troop.full_factor)

    # -- phase 2: movement -------------------------------------------------
    def move_admirals(self, state: GameState, side: str, engine) -> None:
        """Xboat movement.  By default admirals stay where they are."""
        return None

    def move_scouts(self, state: GameState, side: str,
                    scouts: list[Squadron], engine) -> dict[int, str]:
        return {}

    def move_well_led_fleets(self, state: GameState, side: str,
                             fleets, engine) -> dict[int, str]:
        return {}

    def plot_fleet(self, state: GameState, side: str, fleet, turn: int,
                   engine) -> tuple[str, str]:
        return ("hold", fleet.location)

    # -- phase 3: combat ---------------------------------------------------
    def choose_loss(self, state: GameState, side: str,
                    targets: list[Squadron], remaining: int, engine) -> Squadron:
        """Which squadron absorbs battle damage.  Sacrifice the least useful."""
        def cost(sq: Squadron):
            return (sq.attack + sq.bombard, -len(sq.troops), sq.defense)
        empty = [sq for sq in targets if not sq.troops]
        return min(empty or targets, key=cost)

    def wants_to_disengage(self, state: GameState, side: str, hex_id: str,
                           engine) -> bool:
        own = state.squadrons_at(hex_id, side)
        foe = state.squadrons_at(hex_id, state.enemy_of(side))
        if not own or not foe:
            return False
        mine = sum(sq.attack for sq in own)
        theirs = sum(sq.attack for sq in foe)
        return theirs > mine * 2

    def choose_retreat(self, state: GameState, side: str, hex_id: str,
                       jump: int, engine) -> str | None:
        options = [h for h in state.world_map.worlds
                   if h != hex_id and hexmap.distance(hex_id, h) <= jump
                   and not state.squadrons_at(h, state.enemy_of(side))]
        if not options:
            return None
        friendly = [h for h in options if state.controls(h, side)]
        return (friendly or options)[0]

    def attack_sdbs(self, state: GameState, side: str, hex_id: str,
                    engine) -> bool:
        world = state.world_map.get(hex_id)
        if world is None:
            return False
        bombard = sum(sq.bombard for sq in state.squadrons_at(hex_id, side))
        return bombard > 0

    def sdbs_passive(self, state: GameState, side: str, hex_id: str,
                     engine) -> bool:
        return False

    def break_off_sdb_action(self, state: GameState, side: str, hex_id: str,
                             engine) -> bool:
        return False

    def bombing_targets(self, state: GameState, side: str, hex_id: str,
                        bombard: int, engine) -> list[tuple[object, int]]:
        """Split the bombardment total between surface targets."""
        world = state.world_map.get(hex_id)
        enemy = state.enemy_of(side)
        targets: list[tuple[object, int]] = []
        troops = [t for t in state.troops_at(hex_id, enemy)
                  if not (t.guerrilla and not t.overt)]
        if world is not None and world.defense_current > 0 \
                and world.control == enemy:
            troops = troops + ["defense"]
        if not troops:
            return []
        share = max(1, bombard // len(troops))
        for target in troops:
            targets.append((target, share))
        return targets

    def load_troops(self, state: GameState, side: str, engine) -> None:
        """Embark troops standing with a friendly fleet that is going somewhere."""
        enemy = state.enemy_of(side)
        for fleet in state.fleets.values():
            if not fleet.active or fleet.side != side:
                continue
            plot = fleet.plot.get(state.turn)
            going = plot is not None and plot[0] == "jump"
            if not going:
                continue
            hex_id = fleet.location
            if state.squadrons_at(hex_id, enemy):
                continue
            world = state.world_map.get(hex_id)
            spare = self._spare_troops(state, side, hex_id, world)
            if not spare:
                continue
            for uid in fleet.squadrons:
                sq = state.squadrons.get(uid)
                if sq is None or sq.cls.kind in ("scout", "tanker"):
                    continue
                free = sq.capacity - sum(state.troops[t].current
                                         for t in sq.troops if t in state.troops)
                while spare and free > 0:
                    troop = spare[0]
                    if troop.current > free or troop.current <= 0:
                        break
                    spare.pop(0)
                    troop.aboard = sq.uid
                    sq.troops.append(troop.uid)
                    free -= troop.current

    def _spare_troops(self, state, side, hex_id, world):
        """Troops that may leave a world without losing control of it."""
        here = [t for t in state.troops_at(hex_id, side)
                if t.aboard is None and not t.guerrilla and t.current > 0]
        here.sort(key=lambda t: -t.current)
        if world is None or world.control != side:
            return here
        need = world.garrison_required()
        keep = []
        total = 0
        for troop in reversed(here):
            if total < need:
                keep.append(troop)
                total += troop.current
        return [t for t in here if t not in keep]

    def transfer_troops(self, state: GameState, side: str, hex_id: str,
                        blocked: bool, engine) -> None:
        """Land embarked troops when the system is worth holding."""
        world = state.world_map.get(hex_id)
        if world is None:
            return
        if blocked and not any(
                state.troops[u].cls.jump_troops
                for sq in state.squadrons_at(hex_id, side) for u in sq.troops
                if u in state.troops):
            return
        if world.zone == "red" and side == IMPERIAL and not world.zone_lifted:
            if not any(a.warrant for a in state.admirals_at(hex_id, side)):
                return
        enemy = state.enemy_of(side)
        contested = (world.control != side
                     or state.troops_at(hex_id, enemy)
                     or state.garrison_strength(hex_id, side)
                     < world.garrison_required())
        if not contested:
            return
        for sq in state.squadrons_at(hex_id, side):
            for uid in list(sq.troops):
                troop = state.troops.get(uid)
                if troop is None:
                    sq.troops.remove(uid)
                    continue
                troop.aboard = None
                troop.location = hex_id
                sq.troops.remove(uid)

    def surface_fire(self, state: GameState, side: str, hex_id: str,
                     shooters: list[TroopUnit], engine):
        """Return ``[(target, factors, tech_level), ...]``."""
        world = state.world_map.get(hex_id)
        enemy = state.enemy_of(side)
        targets: list[object] = [t for t in state.troops_at(hex_id, enemy)
                                 if not (t.guerrilla and not t.overt)]
        if world is not None and world.control == enemy and world.defense_current > 0:
            targets.append("defense")
        if not targets or not shooters:
            return []
        total = sum(t.combat_strength(world) for t in shooters)
        tech = min((t.tech_level(world) for t in shooters), default=0)
        if total <= 0:
            return []
        # concentrate on the weakest enemy unit first
        def strength(t):
            return (t.combat_strength(world) if isinstance(t, TroopUnit)
                    else float(world.defense_current))
        targets.sort(key=strength)
        allocation = []
        left = total
        for target in targets:
            if left <= 0:
                break
            need = min(left, max(1.0, strength(target) * 3))
            allocation.append((target, need, tech))
            left -= need
        if left > 0 and allocation:
            target, need, t = allocation[0]
            allocation[0] = (target, need + left, t)
        return allocation

    # -- phase 4: reorganisation ------------------------------------------
    def fleet_adjustment(self, state: GameState, side: str, engine) -> None:
        from ..engine import reorganise_fleets
        reorganise_fleets(state, side)

    def set_guerrilla_modes(self, state: GameState, engine) -> None:
        pass

    def declare_armistice(self, state: GameState, engine) -> bool:
        return False


class RandomAgent(Agent):
    """Moves fleets to random reachable systems.  A sparring partner."""

    name = "random"

    def plot_fleet(self, state, side, fleet, turn, engine):
        from ..engine import fleet_jump
        jump = fleet_jump(state, fleet)
        if jump <= 0 or not fleet.location:
            return ("hold", fleet.location)
        options = self._reachable_hexes(state, fleet.location, jump)
        if not options or self.rng.random() < 0.4:
            return ("hold", fleet.location)
        return ("jump", self.rng.choice(options))

    def move_scouts(self, state, side, scouts, engine):
        moves = {}
        for sq in scouts:
            options = self._reachable_hexes(state, sq.location, sq.jump)
            if options:
                moves[sq.uid] = self.rng.choice(options)
        return moves

    def move_well_led_fleets(self, state, side, fleets, engine):
        from ..engine import fleet_jump
        moves = {}
        for fleet in fleets:
            jump = fleet_jump(state, fleet)
            options = self._reachable_hexes(state, fleet.location, jump)
            if options and self.rng.random() < 0.6:
                moves[fleet.uid] = self.rng.choice(options)
        return moves

    def _reachable_hexes(self, state, origin, jump):
        """Systems within jump range, handling the off-map reinforcement boxes."""
        if origin not in state.world_map.worlds:
            entry = state.world_map.entry_hexes.get(origin)
            return [entry] if entry else []
        return [h for h in state.world_map.worlds
                if 0 < hexmap.distance(origin, h) <= jump]

"""Doctrine-driven agents whose behaviour is controlled by a weight vector.

Every operational decision (where to send a fleet, which system to raid, when
to break off) is scored by a linear combination of hand-built features.  The
weights are the trainable parameters: ``ffw.training`` optimises them by
self-play, so the same class covers both the hand-tuned "doctrine" opponent and
the machine-trained one.
"""

from __future__ import annotations

import math

from .. import hexmap
from ..engine import fleet_jump, fleet_plotting_factor
from ..state import GameState, IMPERIAL, ZHODANI
from .base import Agent

#: names of the tunable weights, in vector order
WEIGHTS = [
    "vp_value",            # victory points on offer at the destination
    "capital",             # extra pull of a subsector capital
    "undefended",          # target has weak SDBs and defence battalions
    "enemy_fleet",         # enemy squadrons already in the hex
    "own_support",         # friendly strength within one jump
    "distance",            # cost of a longer jump
    "refuel",              # fuel available at the destination
    "base",                # destination has a friendly base
    "invasion",            # troops embarked and the world is invadeable
    "garrison_gap",        # own world whose garrison is short
    "threat_cover",        # own world under threat within one jump
    "concentration",       # prefer moving where friends are heading
    "home_pull",           # stay inside friendly space
    "risk_tolerance",      # willingness to fight outnumbered
    "guerrilla_overt",     # Zhodani: how eagerly guerrillas rise
    "armistice",           # Zhodani: how readily to call it a day
    "frontier_pull",       # advance towards the enemy's own space
    "takeable",            # the landing force can actually beat the garrison
]

#: a competent hand-written doctrine
DEFAULT_WEIGHTS = {
    "vp_value": 1.00, "capital": 0.60, "undefended": 0.80,
    "enemy_fleet": -0.70, "own_support": 0.45, "distance": -0.55,
    "refuel": 0.90, "base": 0.35, "invasion": 1.30,
    "garrison_gap": 0.50, "threat_cover": 0.65, "concentration": 0.40,
    "home_pull": 0.25, "risk_tolerance": 0.50,
    "guerrilla_overt": 0.35, "armistice": -0.20,
    "frontier_pull": 0.85, "takeable": 1.60,
}


def weight_vector(weights: dict | None = None) -> list[float]:
    weights = weights or DEFAULT_WEIGHTS
    return [float(weights.get(name, 0.0)) for name in WEIGHTS]


def weight_dict(vector) -> dict:
    return {name: float(value) for name, value in zip(WEIGHTS, vector)}


class HeuristicAgent(Agent):
    """Scores every reachable system and sends fleets to the best one."""

    name = "heuristic"

    def __init__(self, side: str, weights=None, seed: int | None = None,
                 label: str | None = None):
        super().__init__(side, seed)
        if weights is None:
            self.w = dict(DEFAULT_WEIGHTS)
        elif isinstance(weights, dict):
            self.w = dict(DEFAULT_WEIGHTS)
            self.w.update(weights)
        else:
            self.w = weight_dict(weights)
        if label:
            self.name = label
        self._intent: dict[str, int] = {}
        self._front_cache = None
        self._intent_turn = -1

    # ------------------------------------------------------------------
    def begin_game(self, state, side, engine):
        self.side = side
        self._intent = {}
        self._front_cache = None
        self._intent_turn = -1

    # -- scoring -------------------------------------------------------
    def score_destination(self, state: GameState, side: str, origin: str,
                          dest: str, carrying: int = 0) -> float:
        world = state.world_map.get(dest)
        if world is None:
            return self._score_box(state, side, origin, dest, carrying)
        w = self.w
        enemy = state.enemy_of(side)
        score = 0.0

        if world.control != side:
            value = world.victory_points
            if world.control is None:
                value = world.tech_level / 2.0
            score += w["vp_value"] * value / 10.0
            if world.subsector_capital:
                score += w["capital"] * 2.0
            weak = 1.0 / (1.0 + world.sdb_current / 50.0 +
                          world.defense_current / 200.0)
            score += w["undefended"] * weak
            if carrying > 0:
                # a landing only pays if it can beat the garrison; the huge
                # high-population worlds are worth exactly their tech level and
                # cost an army, so they are best left behind the front
                opposition = (world.defense_current
                              + state.garrison_strength(dest, enemy))
                if carrying >= max(world.garrison_required(side), opposition * 0.75):
                    score += w["takeable"] * min(2.0, carrying /
                                                 max(1.0, opposition + 1.0))
                score += w["invasion"] * min(1.0, carrying / 300.0)
        else:
            gap = world.garrison_required(side) - state.garrison_strength(dest, side)
            if gap > 0:
                score += w["garrison_gap"] * min(2.0, gap / 20.0)
            score += w["home_pull"] * 0.5
            # an empty fleet should call at a world with troops waiting to embark
            if carrying == 0:
                waiting = sum(t.current for t in state.troops_at(dest, side)
                              if t.aboard is None and not t.guerrilla)
                spare = max(0, waiting - world.garrison_required(side))
                if spare > 0:
                    score += w["invasion"] * 0.6 * min(1.5, spare / 200.0)

        foe = state.squadrons_at(dest, enemy)
        if foe:
            strength = sum(sq.attack for sq in foe)
            score += w["enemy_fleet"] * strength / 20.0
            score += w["risk_tolerance"] * min(1.5, strength / 25.0)

        support = 0
        for other in state.world_map.worlds:
            if hexmap.distance(dest, other) <= 1:
                support += sum(sq.attack for sq in state.squadrons_at(other, side))
        score += w["own_support"] * support / 30.0

        threatened = 0
        for other in state.world_map.worlds:
            near = state.world_map.get(other)
            if near is None or near.control != side:
                continue
            if hexmap.distance(dest, other) <= 1 and \
                    state.squadrons_at(other, enemy):
                threatened += 1
        score += w["threat_cover"] * min(2.0, threatened / 2.0)

        if world.gas_giant or (world.control == side and (world.water or world.base)):
            score += w["refuel"]
        if world.base and world.control == side:
            score += w["base"]
        score += w["concentration"] * self._intent.get(dest, 0) * 0.5
        score += w["distance"] * hexmap.distance(origin, dest) / 3.0
        # a gradient towards the enemy's own space, so fleets advance instead
        # of shuffling between two friendly systems on the border
        near = self._distance_to_front(state, side, dest)
        score += w["frontier_pull"] * (1.0 - min(1.0, near / 8.0))
        return score

    def _score_box(self, state: GameState, side: str, origin: str,
                   dest: str, carrying: int) -> float:
        """Worth of a run back to a reinforcement box to pick up new squadrons.

        Squadrons arriving in a box cannot move without a fleet marker, and
        markers are finite, so a fleet that has nothing better to do should go
        and collect them.
        """
        from ..engine import REINFORCEMENT_BOXES
        if REINFORCEMENT_BOXES.get(dest) != side:
            return -1e9
        waiting = sum(1 for sq in state.squadrons.values()
                      if sq.location == dest and sq.fleet is None
                      and sq.cls.kind != "scout")
        if waiting == 0 or carrying > 0:
            return -1e9
        entry = state.world_map.entry_hexes.get(dest)
        far = hexmap.distance(origin, entry) if entry else 6
        return (self.w["own_support"] * min(3.0, waiting / 3.0)
                + self.w["distance"] * far / 3.0
                + self.w["refuel"] * 0.5)

    def _distance_to_front(self, state: GameState, side: str, dest: str) -> int:
        cache = self._front_cache
        if cache is None or cache[0] is not state.turn:
            enemy = state.enemy_of(side)
            targets = [w.hex for w in state.world_map
                       if w.control == enemy and w.owner != "neutral"]
            self._front_cache = cache = (state.turn, targets)
        targets = cache[1]
        if not targets:
            return 0
        return min(hexmap.distance(dest, h) for h in targets)

    # -- movement ------------------------------------------------------
    def plot_fleet(self, state, side, fleet, turn, engine):
        if self._intent_turn != state.turn:      # intent is a per-turn signal
            self._intent, self._intent_turn = {}, state.turn
        jump = fleet_jump(state, fleet)
        if jump <= 0 or not fleet.location:
            return ("hold", fleet.location)
        # a fleet that must refuel cannot be plotted to jump twice running
        previous = fleet.plot.get(turn - 1)
        if previous is not None and previous[0] == "jump":
            if self._needs_a_turn_to_refuel(state, fleet):
                return ("hold", previous[1])
        origin = fleet.plot.get(turn - 1, ("hold", fleet.location))[1]
        dest = self._best_destination(state, side, fleet, origin, jump)
        if dest is None or dest == origin:
            return ("hold", origin)
        self._intent[dest] = self._intent.get(dest, 0) + 1
        return ("jump", dest)

    def reachable(self, state, side, fleet, origin, jump) -> list[str]:
        """Destinations within jump range where the fleet will find fuel.

        A fleet that jumps somewhere it cannot refuel is stranded for the rest
        of the war, so no destination without a fuel source is ever considered
        unless the fleet is carrying its own tanker.
        """
        from ..engine import can_refuel_at
        classes = [state.squadrons[u].cls.refuel for u in fleet.squadrons
                   if u in state.squadrons]
        has_tanker = any(state.squadrons[u].cls.kind == "tanker"
                         for u in fleet.squadrons if u in state.squadrons)
        # a fleet plots several turns ahead, so "carrying troops" has to mean
        # troops already aboard *or* troops it can pick up before it leaves
        troops = any(state.squadrons[u].troops for u in fleet.squadrons
                     if u in state.squadrons)
        if not troops and origin in state.world_map.worlds:
            troops = self._loadable(state, side, origin) > 0
        out = []
        for h in state.world_map.worlds:
            if not 0 < hexmap.distance(origin, h) <= jump:
                continue
            if has_tanker or can_refuel_at(state, side, h, classes):
                out.append(h)
            elif troops and can_refuel_at(state, side, h, classes,
                                          expect_control=True):
                out.append(h)          # an invasion that expects to take the world
        # a friendly reinforcement box is always a legal, always-fuelled
        # destination, one jump further than its entry hex
        from ..engine import REINFORCEMENT_BOXES
        for box, owner in REINFORCEMENT_BOXES.items():
            if owner != side:
                continue
            entry = state.world_map.entry_hexes.get(box)
            if entry and (origin == entry
                          or hexmap.distance(origin, entry) + 1 <= jump):
                out.append(box)
        return out

    def _needs_a_turn_to_refuel(self, state, fleet) -> bool:
        return any(state.squadrons[u].cls.refuel != "streamlined"
                   for u in fleet.squadrons if u in state.squadrons)

    def _best_destination(self, state, side, fleet, origin, jump):
        if origin not in state.world_map.worlds:
            entry = state.world_map.entry_hexes.get(origin)
            return entry
        carrying = sum(state.troops[t].current
                       for u in fleet.squadrons if u in state.squadrons
                       for t in state.squadrons[u].troops if t in state.troops)
        if carrying == 0:
            carrying = self._loadable(state, side, origin)
        options = self.reachable(state, side, fleet, origin, jump)
        if not options:
            return None
        scored = [(self.score_destination(state, side, origin, h, carrying), h)
                  for h in options]
        scored.sort(reverse=True)
        top = scored[:4]
        weights = [math.exp(2.0 * (s - top[0][0])) for s, _ in top]
        total = sum(weights)
        pick = self.rng.random() * total
        for (score, hex_id), weight in zip(top, weights):
            pick -= weight
            if pick <= 0:
                return hex_id
        return top[0][1]

    def _loadable(self, state, side, hex_id) -> int:
        world = state.world_map.get(hex_id)
        spare = self._spare_troops(state, side, hex_id, world)
        return sum(t.current for t in spare)

    def move_well_led_fleets(self, state, side, fleets, engine):
        moves = {}
        for fleet in fleets:
            jump = fleet_jump(state, fleet)
            dest = self._best_destination(state, side, fleet, fleet.location, jump)
            if dest and dest != fleet.location:
                moves[fleet.uid] = dest
        return moves

    def move_scouts(self, state, side, scouts, engine):
        """Scouts screen forward: sit on enemy approach routes and refuel points."""
        moves = {}
        enemy = state.enemy_of(side)
        for sq in scouts:
            if sq.location not in state.world_map.worlds:
                entry = state.world_map.entry_hexes.get(sq.location)
                if entry:
                    moves[sq.uid] = entry
                continue
            options = [h for h in state.world_map.worlds
                       if 0 < hexmap.distance(sq.location, h) <= sq.jump]
            if not options:
                continue
            def scout_score(h):
                world = state.world_map.get(h)
                value = 0.0
                if state.squadrons_at(h, enemy):
                    value += 2.0
                if world.control == enemy:
                    value += 1.0
                if world.gas_giant:
                    value += 0.5
                if state.squadrons_at(h, side):
                    value -= 1.0
                return value - 0.2 * hexmap.distance(sq.location, h)
            moves[sq.uid] = max(options, key=scout_score)
        return moves

    def move_admirals(self, state, side, engine):
        """Send flag officers by xboat to the largest fleet that lacks one."""
        enemy = state.enemy_of(side)
        needy = sorted((f for f in state.fleets.values()
                        if f.active and f.side == side and f.admiral is None),
                       key=lambda f: -len(f.squadrons))
        for fleet in needy:
            if fleet.location not in state.world_map.worlds:
                continue
            if state.squadrons_at(fleet.location, enemy):
                continue
            free = [a for a in state.admirals.values()
                    if a.side == side and a.fleet is None
                    and a.location in state.world_map.worlds
                    and not state.squadrons_at(a.location, enemy)]
            best = None
            for adm in free:
                if fleet.location in state.world_map.xboat_reach(adm.location):
                    if best is None or adm.planning < best.planning:
                        best = adm
            if best is not None:
                best.location = fleet.location
                best.aboard = False

    # -- combat --------------------------------------------------------
    def wants_to_disengage(self, state, side, hex_id, engine):
        own = state.squadrons_at(hex_id, side)
        foe = state.squadrons_at(hex_id, state.enemy_of(side))
        if not own or not foe:
            return False
        mine = sum(sq.attack for sq in own) + 1
        theirs = sum(sq.attack for sq in foe) + 1
        tolerance = 1.4 + self.w["risk_tolerance"]
        if any(sq.troops for sq in own) and theirs > mine:
            return True
        return theirs > mine * tolerance

    def choose_retreat(self, state, side, hex_id, jump, engine):
        enemy = state.enemy_of(side)
        options = [h for h in state.world_map.worlds
                   if h != hex_id and hexmap.distance(hex_id, h) <= jump
                   and not state.squadrons_at(h, enemy)]
        if not options:
            return None
        def safety(h):
            world = state.world_map.get(h)
            value = 0.0
            if world.control == side:
                value += 2.0
            if world.base and world.control == side:
                value += 1.5
            if world.gas_giant:
                value += 0.5
            value += 0.3 * sum(sq.attack for sq in state.squadrons_at(h, side)) / 10.0
            return value - 0.1 * hexmap.distance(hex_id, h)
        return max(options, key=safety)

    def sdbs_passive(self, state, side, hex_id, engine):
        """Hide the boats when the raider is too strong to hurt."""
        world = state.world_map.get(hex_id)
        if world is None:
            return False
        enemy = state.enemy_of(side)
        bombard = sum(sq.bombard for sq in state.squadrons_at(hex_id, enemy))
        return bombard > world.sdb_current / 2.0

    def bombing_targets(self, state, side, hex_id, bombard, engine):
        world = state.world_map.get(hex_id)
        enemy = state.enemy_of(side)
        troops = [t for t in state.troops_at(hex_id, enemy)
                  if not (t.guerrilla and not t.overt)]
        targets: list = list(troops)
        if world is not None and world.control == enemy and world.defense_current > 0:
            targets.append("defense")
        if not targets:
            return []
        # soften the defence battalions first: they are what stops an invasion
        def priority(t):
            if t == "defense":
                return -world.defense_current
            return -t.current
        targets.sort(key=priority)
        take = targets[:2]
        share = bombard // len(take)
        return [(t, share) for t in take]

    # -- reorganisation ------------------------------------------------
    def set_guerrilla_modes(self, state, engine):
        """Rise where the rising can actually take a world, hide otherwise."""
        for unit in state.troops.values():
            if not unit.guerrilla:
                continue
            world = state.world_map.get(unit.location)
            if world is None:
                continue
            imperial = state.garrison_strength(unit.location, IMPERIAL)
            defence = world.defense_current
            worth = unit.current * (1.0 + self.w["guerrilla_overt"])
            unit.overt = worth > (imperial + defence) * 0.6 and unit.losses < 60

    def declare_armistice(self, state, engine):
        if self.side != ZHODANI:
            return False
        margin = state.victory_margin()
        # call it off when ahead and the tide is not going to turn
        threshold = 60 + 120 * self.w["armistice"]
        return state.turn >= 40 and margin > threshold


class ScriptedAgent(HeuristicAgent):
    """The stock doctrine, with an explicit opening plan.

    The Zhodani open with the historical drive on Jewell, Efate and Regina; the
    Imperium covers the Regina-Rhylanor axis and trades space for time.
    """

    name = "scripted"

    OPENING = {
        ZHODANI: ["1510", "2109", "2314", "1018", "1520"],
        IMPERIAL: ["2314", "2109", "3120", "2123", "1510"],
    }

    def score_destination(self, state, side, origin, dest, carrying=0):
        score = super().score_destination(state, side, origin, dest, carrying)
        if state.turn <= 12:
            plan = self.OPENING.get(side, [])
            if dest in plan:
                score += 2.5 - 0.3 * plan.index(dest)
        return score

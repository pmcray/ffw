"""A search agent: sample candidate plans and roll them out.

Fifth Frontier War is simultaneous and hidden-information, so a classical
minimax tree is not available.  What *is* available is a fast forward model:
the engine can be deep-copied and run with cheap stand-in agents.  The
LookaheadAgent samples several candidate destinations for each fleet, plays a
short rollout for each, and keeps the plan with the best average outcome.
"""

from __future__ import annotations

import copy
import random

from .. import hexmap
from ..engine import Engine, fleet_jump
from ..state import GameState, IMPERIAL, ZHODANI
from .heuristic import HeuristicAgent


class LookaheadAgent(HeuristicAgent):
    """Monte-Carlo plan evaluation on top of the heuristic doctrine."""

    name = "lookahead"

    def __init__(self, side: str, weights=None, seed: int | None = None,
                 candidates: int = 3, rollouts: int = 2, horizon: int = 3,
                 label: str | None = None):
        super().__init__(side, weights, seed, label)
        self.candidates = candidates
        self.rollouts = rollouts
        self.horizon = horizon

    def _best_destination(self, state, side, fleet, origin, jump):
        if origin not in state.world_map.worlds:
            return super()._best_destination(state, side, fleet, origin, jump)
        options = [h for h in state.world_map.worlds
                   if 0 < hexmap.distance(origin, h) <= jump]
        if not options:
            return None
        carrying = sum(state.troops[t].current
                       for u in fleet.squadrons if u in state.squadrons
                       for t in state.squadrons[u].troops if t in state.troops)
        scored = sorted(
            ((self.score_destination(state, side, origin, h, carrying), h)
             for h in options), reverse=True)
        shortlist = [h for _, h in scored[:self.candidates]]
        if len(shortlist) <= 1 or state.turn > 45:
            return shortlist[0] if shortlist else None

        best, best_value = shortlist[0], -1e18
        for dest in shortlist:
            total = 0.0
            for r in range(self.rollouts):
                total += self._rollout(state, side, fleet, dest,
                                       seed=self.rng.randrange(1 << 30))
            average = total / max(1, self.rollouts)
            if average > best_value:
                best, best_value = dest, average
        return best

    def _rollout(self, state: GameState, side: str, fleet, dest: str,
                 seed: int) -> float:
        """Play a few turns from a copy of the position and score the result."""
        try:
            sim = copy.deepcopy(state)
        except Exception:
            return 0.0
        sim.log = []
        sim_fleet = sim.fleets.get(fleet.uid)
        if sim_fleet is None:
            return 0.0
        sim_fleet.plot[sim.turn + 1] = ("jump", dest)
        friend = HeuristicAgent(side, self.w, seed=seed)
        foe = HeuristicAgent(sim.enemy_of(side), seed=seed + 1)
        agents = {side: friend, sim.enemy_of(side): foe}
        engine = Engine(sim, agents[IMPERIAL], agents[ZHODANI],
                        rng=random.Random(seed))
        for _ in range(self.horizon):
            if sim.game_over:
                break
            try:
                engine.play_turn()
            except Exception:
                break
        margin = sim.victory_margin()
        return margin if side == ZHODANI else -margin

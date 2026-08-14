"""A search agent: sample candidate plans and roll them out.

Fifth Frontier War is simultaneous and hidden-information, so a classical
minimax tree is not available.  What *is* available is a fast forward model:
the engine can be copied and run with cheap stand-in agents.  The
LookaheadAgent samples several candidate destinations for a fleet, plays a
short rollout for each, and keeps the plan with the best outcome.

This is the one agent in the collection that does not decide by the doctrine's
scoring core.  The heuristic, scripted, neural and trained agents are four
tunings of one linear evaluation; this one uses that evaluation only to
*propose* a shortlist and then decides by simulation, so a tournament against
it compares two ways of thinking rather than two weight vectors.

Three things make it affordable enough to enter a tournament:

* ``GameState.clone`` instead of ``copy.deepcopy`` -- roughly twenty times
  cheaper, because the frozen counter classes and the map geometry are shared
  rather than walked.
* **Common random numbers.**  Every candidate for a decision is rolled out on
  the *same* seed, so the rollouts differ only in the destination being tested.
  Sharing the randomness removes most of the variance that otherwise forces
  several rollouts per candidate, which is why one rollout each is the default
  -- the same paired-comparison trick ``play_paired`` uses on whole games.
* A per-turn **budget**.  Searching every fleet's plot costs far more than it
  is worth; the strongest few fleets get the search and the rest fall back to
  the doctrine.
"""

from __future__ import annotations

import random

from ..engine import Engine
from ..state import GameState, IMPERIAL, ZHODANI
from .heuristic import HeuristicAgent


class LookaheadAgent(HeuristicAgent):
    """Monte-Carlo plan evaluation on top of the heuristic doctrine."""

    name = "lookahead"

    def __init__(self, side: str, weights=None, seed: int | None = None,
                 candidates: int = 3, rollouts: int = 1, horizon: int = 2,
                 budget: int = 6, evaluator=None, label: str | None = None,
                 pickets: bool = True):
        super().__init__(side, weights, seed, label, pickets)
        self.candidates = candidates
        self.rollouts = rollouts
        self.horizon = horizon
        self.budget = budget
        #: optional value network used to score the leaf position.  Two turns
        #: of war move the scoreboard by almost nothing, so reading the raw
        #: margin at the leaf mostly measures the combat dice; a network
        #: trained on margin *deltas* is asked the question the leaf poses --
        #: which of these positions is about to improve.
        self.evaluator = evaluator
        self._searched = (-1, frozenset())
        self.rollouts_played = 0

    def begin_game(self, state, side, engine):
        super().begin_game(state, side, engine)
        self._searched = (-1, frozenset())
        self.rollouts_played = 0

    # -- budget --------------------------------------------------------
    def _importance(self, state: GameState, fleet) -> float:
        """How much this fleet's next move is likely to matter."""
        power = 0.0
        for uid in fleet.squadrons:
            sq = state.squadrons.get(uid)
            if sq is None:
                continue
            power += sq.attack
            power += 2.0 * sum(state.troops[t].current for t in sq.troops
                               if t in state.troops)
        return power

    def _worth_searching(self, state: GameState, side: str, fleet) -> bool:
        turn, chosen = self._searched
        if turn != state.turn:
            active = [f for f in state.fleets.values()
                      if f.active and f.side == side]
            active.sort(key=lambda f: (-self._importance(state, f), f.uid))
            chosen = frozenset(f.uid for f in active[:self.budget])
            self._searched = (state.turn, chosen)
        return fleet.uid in chosen

    # -- decision ------------------------------------------------------
    def _best_destination(self, state, side, fleet, origin, jump):
        if origin not in state.world_map.worlds:
            return super()._best_destination(state, side, fleet, origin, jump)
        options = state.world_map.within(origin, jump)
        if not options:
            return None
        carrying = sum(state.troops[t].current
                       for u in fleet.squadrons if u in state.squadrons
                       for t in state.squadrons[u].troops if t in state.troops)
        index = self.squadron_index(state, side)
        scored = sorted(
            ((self.score_destination(state, side, origin, h, carrying, index), h)
             for h in options), reverse=True)
        shortlist = [h for _, h in scored[:self.candidates]]
        if len(shortlist) <= 1 or state.turn > 45:
            return shortlist[0] if shortlist else None
        if not self._worth_searching(state, side, fleet):
            return super()._best_destination(state, side, fleet, origin, jump)

        # common random numbers: one draw per decision, shared by every
        # candidate, so the comparison isolates the destination
        seeds = [self.rng.randrange(1 << 30) for _ in range(self.rollouts)]
        best, best_value = shortlist[0], -1e18
        for dest in shortlist:
            total = 0.0
            for seed in seeds:
                total += self._rollout(state, side, fleet, dest, seed=seed)
            average = total / max(1, self.rollouts)
            if average > best_value:
                best, best_value = dest, average
        return best

    def _rollout(self, state: GameState, side: str, fleet, dest: str,
                 seed: int) -> float:
        """Play a few turns from a copy of the position and score the result."""
        try:
            sim = state.clone()
        except Exception:
            return 0.0
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
            self.rollouts_played += 1
        return self._leaf_value(sim, side)

    def _leaf_value(self, sim: GameState, side: str) -> float:
        if self.evaluator is None:
            margin = sim.victory_margin()
            return margin if side == ZHODANI else -margin
        from .. import features
        value = self.evaluator(
            features.perspective(features.extract(sim), ZHODANI))
        return float(value) if side == ZHODANI else -float(value)

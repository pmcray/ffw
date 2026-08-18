"""Optimising a doctrine's weights, for whichever game it belongs to.

``ffw/training.py`` has a cross-entropy search that works and is two thousand
lines deep in *Fifth Frontier War* -- fleet plots, armistice turns, value
networks.  Copying it into ``ie/`` would have given the second game a trainer
and the project two of them to keep in step.  What actually differs between the
games is smaller than that: the weight dictionary, the class that turns one into
an agent, and how a finished game is scored.  All three are already behind
``GameAdapter``, so the search lives here and is pointed at a game by name.

Two things are deliberately not shared with ``ffw.training``:

**Pairing.** *Fifth Frontier War* pairs games by swapping seats, because both
seats are playable.  *Invasion: Earth* has no such symmetry -- the Imperium
always attacks and the Solomani always defend -- so a comparison here pairs two
weight vectors on the *same* seat against the same opponent on the same seeds.
That is still common random numbers, and it still answers "did this change the
play"; it just cannot also answer "did this player draw the better side".

**The objective.** Territory alone is a poor training signal in a game where
most candidates take no territory at all: the search spends its generations
looking at a flat field of zeros.  ``objective`` therefore offers territory plus
attrition -- how much of the enemy is gone, scaled by what they started with --
which is dense enough to climb and expressible for either game through the
adapter.  Verification uses territory alone, because territory is what the
rules score.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor

from .adapter import adapter_for

#: What a candidate is scored on while it is being searched for.
OBJECTIVES = ("margin", "margin+attrition")

#: How much of the training score attrition may account for.  Small: it is
#: there to give the search a gradient where territory gives none, not to
#: redefine what winning means.
ATTRITION_WEIGHT = 0.35


def tunables(game: str, side: str) -> list:
    """The weight names this side's half of the doctrine actually reads.

    A doctrine that plays both sides from one vector has dimensions in it that
    the side being trained never looks at, and a search that varies them anyway
    pays for the extra dimensions in samples it does not have.  A game that has
    not said which are which gets all of them, which is what the search used to
    do everywhere.
    """
    defaults, _agent_class = doctrine_for(game)
    if game == "ie":
        from ie.agents.heuristic import SIDE_WEIGHTS
        return [k for k in sorted(defaults) if k in SIDE_WEIGHTS.get(side, ())]
    return sorted(defaults)


def doctrine_for(game: str):
    """``(default weights, agent class)`` for a game, imported on demand."""
    if game == "ie":
        from ie.agents.heuristic import HeuristicAgent, WEIGHTS
        return dict(WEIGHTS), HeuristicAgent
    if game == "ffw":
        # ffw.agents.WEIGHTS is the list of names; the values live in
        # DEFAULT_WEIGHTS.  ie names the dictionary WEIGHTS.  The two games
        # disagreeing about this is exactly the kind of thing an adapter is for.
        from ffw.agents import DEFAULT_WEIGHTS, HeuristicAgent
        return dict(DEFAULT_WEIGHTS), HeuristicAgent
    raise ValueError("no doctrine for %r" % game)


def objective(adapter, state, side: str, shape: str = "margin") -> float:
    """Score a finished game from ``side``'s point of view.

    ``margin`` is the adapter's own normalised margin, which is the share of
    the objectives held and is what the rules care about.  ``margin+attrition``
    adds how much of the enemy's starting force is gone, which is what a
    doctrine that has not yet learned to take an objective can still improve.
    """
    score = adapter.normalised_margin(state, side)
    if shape == "margin":
        return score
    if shape != "margin+attrition":
        raise ValueError("unknown objective %r" % shape)
    enemy = adapter.enemy_of(side)
    start = adapter.initial_force(enemy)
    if start <= 0:                                     # pragma: no cover
        return score
    left = adapter.force_strength(state, enemy)
    return score + ATTRITION_WEIGHT * (1.0 - max(0.0, left) / start)


# --------------------------------------------------------------------------
def play_one(job) -> float:
    """One game, at module level so a process pool can pickle it."""
    game, side, weights, opponent_weights, seed, max_turns, shape = job
    adapter = adapter_for(game)
    _defaults, agent_class = doctrine_for(game)
    enemy = adapter.enemy_of(side)
    agents = {
        side: agent_class(side, weights, seed=seed, label="candidate"),
        enemy: agent_class(enemy, opponent_weights, seed=seed + 1,
                           label="opponent"),
    }
    state = adapter.new_game(seed=seed)
    adapter.play(state, agents, max_turns=max_turns)
    return objective(adapter, state, side, shape)


def _map(jobs, workers: int):
    if workers <= 1 or len(jobs) < 2:
        return [play_one(job) for job in jobs]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(play_one, jobs))


def default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 1)))


def score_weights(game: str, side: str, weights: dict,
                  opponent_weights: dict | None = None, games: int = 4,
                  seed: int = 0, max_turns: int = 24,
                  shape: str = "margin+attrition", workers: int = 1) -> float:
    """Mean objective over a fixed block of seeds.

    The block is fixed on purpose.  Every candidate in every generation is
    judged on the same games, because otherwise the score of a generation says
    which seeds it drew rather than how it played -- the lesson ``ffw`` records
    as common random numbers.
    """
    jobs = [(game, side, weights, opponent_weights, seed * 1000 + i,
             max_turns, shape) for i in range(games)]
    scores = _map(jobs, workers)
    return sum(scores) / len(scores)


def score_many(game: str, side: str, candidates: list,
               opponent_weights: dict | None = None, games: int = 4,
               seed: int = 0, max_turns: int = 24,
               shape: str = "margin+attrition", workers: int = 1) -> list:
    """Score a whole population in one batch.

    A generation is ten candidates of three games each; run one candidate at a
    time and a four-core machine spends most of its life with three cores idle.
    Batching them is the difference between a training run over lunch and one
    over an afternoon.
    """
    jobs = [(game, side, weights, opponent_weights, seed * 1000 + i,
             max_turns, shape)
            for weights in candidates for i in range(games)]
    scores = _map(jobs, workers)
    return [sum(scores[n * games:(n + 1) * games]) / games
            for n in range(len(candidates))]


def paired_advantage(game: str, side: str, weights: dict,
                     against: dict | None = None, opponent_weights=None,
                     games: int = 12, seed: int = 0, max_turns: int = 24,
                     shape: str = "margin", workers: int = 1) -> dict:
    """Is ``weights`` better than ``against`` on the same seat and seeds?

    Returns the mean difference, its standard error and a verdict, using the
    same arithmetic and the same two-standard-error rule as ``ffw.training``'s
    ``summarise`` -- imported rather than re-derived, because a project should
    have one place that decides whether a difference is real.
    """
    from ffw.training import summarise
    defaults, _agent_class = doctrine_for(game)
    against = defaults if against is None else against
    seeds = [seed * 1000 + i for i in range(games)]
    jobs = ([(game, side, weights, opponent_weights, s, max_turns, shape)
             for s in seeds]
            + [(game, side, against, opponent_weights, s, max_turns, shape)
               for s in seeds])
    scores = _map(jobs, workers)
    mine, theirs = scores[:games], scores[games:]
    return summarise([a - b for a, b in zip(mine, theirs)])


# --------------------------------------------------------------------------
class TrainingLog:
    """One row a generation, the same shape ``ffw.training.TrainingLog`` has."""

    def __init__(self):
        self.rows: list[dict] = []

    def add(self, **row) -> None:
        self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]

    def curve(self, key: str = "incumbent"):
        return [row[key] for row in self.rows]


def train(game: str = "ie", side: str | None = None, generations: int = 6,
          population: int = 10, elite: int = 3, games: int = 3,
          sigma: float = 0.35, seed: int = 0, base_weights: dict | None = None,
          opponent_weights: dict | None = None, max_turns: int = 24,
          shape: str = "margin+attrition", workers: int | None = None,
          progress=None):
    """Cross-entropy optimisation of one side's weights.  ``(best, log)``.

    The loop is the one ``ffw`` arrived at after several that did not work:
    sample a population around the current centre, keep the elite, and move the
    centre **only if the proposal is better on the same games**.  Plain
    cross-entropy moves every generation and walks downhill happily whenever the
    elite sample was lucky, which on a game this noisy is most generations.  The
    spread is floored by a separately decayed exploration term for the same
    reason: the standard deviation of three elite samples collapses within a
    couple of generations and the search stops looking.
    """
    import numpy as np

    adapter = adapter_for(game)
    side = side or adapter.sides[0]
    defaults, _agent_class = doctrine_for(game)
    names = tunables(game, side)
    fixed = {k: v for k, v in defaults.items() if k not in names}
    workers = default_workers() if workers is None else workers

    def to_vector(weights):
        return np.asarray([float(weights[k]) for k in names], dtype=np.float64)

    def to_weights(vector):
        out = dict(fixed)
        out.update({k: float(v) for k, v in zip(names, vector)})
        return out

    rng = np.random.default_rng(seed)
    mean = to_vector(base_weights or defaults)
    spread = np.full(mean.shape, sigma)
    log = TrainingLog()

    def evaluate_all(vectors) -> list:
        return score_many(game, side, [to_weights(v) for v in vectors],
                          opponent_weights, games=games, seed=seed,
                          max_turns=max_turns, shape=shape, workers=workers)

    def evaluate(vector) -> float:
        return evaluate_all([vector])[0]

    incumbent = evaluate(mean)
    best_weights, best_score = to_weights(mean), incumbent

    for generation in range(generations):
        candidates = rng.normal(mean, spread, (population, len(mean)))
        scores = np.array(evaluate_all(candidates))
        order = np.argsort(scores)[::-1]
        keep = candidates[order[:elite]]
        proposal = keep.mean(axis=0)
        proposal_score = evaluate(proposal)
        if proposal_score >= incumbent:
            mean, incumbent = proposal, proposal_score
        explore = sigma * (0.75 ** (generation + 1))
        spread = np.maximum(keep.std(axis=0), explore)
        top = float(scores[order[0]])
        if top > best_score:
            best_score, best_weights = top, to_weights(candidates[order[0]])
        log.add(generation=generation, mean=float(scores.mean()), best=top,
                incumbent=float(incumbent), weights=to_weights(mean))
        if progress is not None:
            progress(generation, float(scores.mean()), top, float(incumbent))
    if incumbent > best_score:
        best_weights, best_score = to_weights(mean), incumbent
    return best_weights, log

"""Training and evaluation harness.

Two learning methods are provided, because the two things worth learning in
Fifth Frontier War are different in kind:

``train_weights``
    Cross-entropy method over the doctrine weight vector.  A population of
    candidate doctrines plays against a fixed opponent; the best fraction is
    kept and the sampling distribution is refitted to it.  This is what tunes
    *where fleets go*.

``train_value_network``
    Self-play regression.  Games are played, every position is recorded with
    the final victory margin as its label, and a small MLP is fitted to predict
    it.  The resulting network drives ``NeuralAgent``, which uses it to decide
    *how hard to press* -- when to gamble and when to consolidate.

Both are deliberately cheap: a whole game runs in a few seconds, so a useful
training run finishes inside a notebook cell.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import features
from .agents import (HeuristicAgent, NeuralAgent, ScriptedAgent, ValueNetwork,
                     WEIGHTS, weight_dict, weight_vector)
from .engine import new_game, play
from .state import IMPERIAL, ZHODANI

VICTORY_ORDER = [
    "imperial automatic victory", "imperial decisive victory",
    "imperial major victory", "imperial marginal victory", "stalemate",
    "zhodani marginal victory", "zhodani major victory",
    "zhodani decisive victory", "zhodani automatic victory",
]


# --------------------------------------------------------------------------
def play_match(imperial_agent, zhodani_agent, seed: int = 0,
               max_turns: int = 55, record: bool = False):
    """Play one game.  Returns ``(margin, result, positions)``."""
    state = new_game(seed=seed)
    positions: list[np.ndarray] = []
    on_turn = (lambda s: positions.append(features.extract(s))) if record else None
    result = play(state, imperial_agent, zhodani_agent,
                  max_turns=max_turns, on_turn=on_turn)
    return state.victory_margin(), result, positions


def score_for(side: str, margin: float) -> float:
    """Signed score: positive is good for ``side``."""
    return margin if side == ZHODANI else -margin


# --------------------------------------------------------------------------
@dataclass
class TrainingLog:
    generations: list[dict] = field(default_factory=list)

    def add(self, **kw) -> None:
        self.generations.append(kw)

    def best(self) -> dict | None:
        return max(self.generations, key=lambda g: g["best_score"]) \
            if self.generations else None

    def to_arrays(self):
        gen = [g["generation"] for g in self.generations]
        mean = [g["mean_score"] for g in self.generations]
        best = [g["best_score"] for g in self.generations]
        return gen, mean, best


def train_weights(side: str = ZHODANI, generations: int = 6,
                  population: int = 12, elite: int = 4, games: int = 2,
                  sigma: float = 0.45, opponent=None, seed: int = 0,
                  base_weights: dict | None = None, max_turns: int = 45,
                  progress=None):
    """Cross-entropy optimisation of the doctrine weights for one side.

    Returns ``(best_weights, log)``.
    """
    rng = np.random.default_rng(seed)
    mean = np.asarray(weight_vector(base_weights), dtype=np.float64)
    spread = np.full(mean.shape, sigma)
    log = TrainingLog()
    best_overall, best_score = weight_dict(mean), -1e18

    for generation in range(generations):
        candidates = rng.normal(mean, spread, (population, len(mean)))
        scores = np.zeros(population)
        for i, vector in enumerate(candidates):
            total = 0.0
            for g in range(games):
                match_seed = seed * 1000 + generation * 97 + g
                learner = HeuristicAgent(side, weight_dict(vector),
                                         seed=match_seed, label="candidate")
                other = (opponent(state_side(side), match_seed) if opponent
                         else ScriptedAgent(state_side(side), seed=match_seed))
                if side == ZHODANI:
                    margin, _, _ = play_match(other, learner, seed=match_seed,
                                              max_turns=max_turns)
                else:
                    margin, _, _ = play_match(learner, other, seed=match_seed,
                                              max_turns=max_turns)
                total += score_for(side, margin)
            scores[i] = total / games
        order = np.argsort(scores)[::-1]
        keep = candidates[order[:elite]]
        mean = keep.mean(axis=0)
        spread = np.maximum(keep.std(axis=0), 0.05)
        top = float(scores[order[0]])
        if top > best_score:
            best_score, best_overall = top, weight_dict(candidates[order[0]])
        log.add(generation=generation, mean_score=float(scores.mean()),
                best_score=top, weights=weight_dict(mean))
        if progress is not None:
            progress(generation, float(scores.mean()), top)
    return best_overall, log


def state_side(side: str) -> str:
    return IMPERIAL if side == ZHODANI else ZHODANI


# --------------------------------------------------------------------------
def train_value_network(games: int = 12, epochs: int = 60, hidden: int = 32,
                        seed: int = 0, network: ValueNetwork | None = None,
                        max_turns: int = 45, agents=None, progress=None,
                        side: str = ZHODANI, weights: dict | None = None):
    """Self-play regression: predict the final margin from a position.

    The network is side-agnostic -- ``features.perspective`` flips the vector
    and ``NeuralAgent`` negates the score for the Imperium -- but which side
    the learning agent plays still shapes the positions it sees, so ``side``
    puts the ``NeuralAgent`` on the side being trained.
    """
    net = network or ValueNetwork(hidden=hidden, seed=seed)
    xs: list[np.ndarray] = []
    ys: list[float] = []
    for g in range(games):
        match_seed = seed * 7919 + g
        if agents is None:
            learner = NeuralAgent(side, network=net, weights=weights,
                                  seed=match_seed + 1)
            other = ScriptedAgent(state_side(side), seed=match_seed)
            imperial, zhodani = ((other, learner) if side == ZHODANI
                                 else (learner, other))
        else:
            imperial, zhodani = agents(match_seed)
        margin, result, positions = play_match(imperial, zhodani,
                                               seed=match_seed,
                                               max_turns=max_turns, record=True)
        label = math.tanh(margin / 200.0)
        for vector in positions:
            xs.append(features.perspective(vector, ZHODANI))
            ys.append(label)
        if progress is not None:
            progress(g, margin, result)
    if not xs:
        return net, 0.0
    x = np.asarray(xs)
    y = np.asarray(ys)
    loss = net.train(x, y, epochs=epochs, seed=seed)
    return net, loss


# --------------------------------------------------------------------------
def tournament(entries: dict, games: int = 3, max_turns: int = 45,
               seed: int = 0, progress=None):
    """Round-robin: every agent plays every other on both sides.

    ``entries`` maps a label to a factory ``f(side, seed) -> Agent``.
    Returns a table of results keyed by ``(imperial_label, zhodani_label)`` and
    a summary of each entry's average margin from its own point of view.
    """
    names = list(entries)
    table = {}
    summary = {n: {"games": 0, "score": 0.0, "wins": 0, "losses": 0,
                   "draws": 0} for n in names}
    for a in names:
        for b in names:
            if a == b:
                continue
            margins = []
            for g in range(games):
                match_seed = seed * 131 + hash((a, b)) % 1000 + g
                imperial = entries[a](IMPERIAL, match_seed)
                zhodani = entries[b](ZHODANI, match_seed + 1)
                margin, result, _ = play_match(imperial, zhodani,
                                               seed=match_seed,
                                               max_turns=max_turns)
                margins.append(margin)
                summary[a]["games"] += 1
                summary[b]["games"] += 1
                summary[a]["score"] += -margin
                summary[b]["score"] += margin
                if margin > 50:
                    summary[b]["wins"] += 1
                    summary[a]["losses"] += 1
                elif margin < -50:
                    summary[a]["wins"] += 1
                    summary[b]["losses"] += 1
                else:
                    summary[a]["draws"] += 1
                    summary[b]["draws"] += 1
                if progress is not None:
                    progress(a, b, g, margin, result)
            table[(a, b)] = sum(margins) / len(margins)
    for name in names:
        played = max(1, summary[name]["games"])
        summary[name]["average"] = summary[name]["score"] / played
    return table, summary


# --------------------------------------------------------------------------
def save_agent(path: str, weights: dict | None = None,
               network: ValueNetwork | None = None, meta: dict | None = None):
    payload = {"weights": weights, "meta": meta or {},
               "network": network.to_dict() if network is not None else None}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    return path


def load_agent(path: str):
    with open(path, encoding="utf-8") as fh:
        payload = json.load(fh)
    net = (ValueNetwork.from_dict(payload["network"])
           if payload.get("network") else None)
    return payload.get("weights"), net, payload.get("meta", {})

"""Training and evaluation harness.

Two learning methods are provided, because the two things worth learning in
Fifth Frontier War are different in kind:

``train_weights``
    Cross-entropy method over the doctrine weight vector.  A population of
    candidate doctrines plays against a fixed opponent; the best fraction is
    kept and the sampling distribution is refitted to it.  This is what tunes
    *where fleets go*.

``train_value_network``
    Regression on played games.  Every position is recorded with the final
    victory margin as its label and a small MLP is fitted to predict it.  The
    resulting network drives ``NeuralAgent``, which uses it to decide *how hard
    to press* -- when to gamble and when to consolidate.

Both are deliberately cheap: a whole game runs in a few seconds, so a useful
training run finishes inside a notebook cell.

A warning about the second one.  Every position in a game carries that game's
final margin, so the independent sample size is the number of *games*, not the
number of positions -- a few dozen games is a few dozen samples however many
thousand rows the training matrix has.  At that size the fit is weak and its
measured quality is unstable: ``evaluator_report`` returns a bootstrap interval
over games precisely so this is visible rather than hidden behind a
reassuringly small regression loss.  Treat the network as a rough thermostat,
not an oracle, and raise the game count a long way before trusting it further.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import features
from .agents import (HeuristicAgent, NeuralAgent, RandomAgent, ScriptedAgent,
                     ValueNetwork, WEIGHTS, weight_dict, weight_vector)
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

    def incumbents(self):
        """Score of the accepted centre after each generation."""
        return ([g["generation"] for g in self.generations],
                [g.get("incumbent_score") for g in self.generations])


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

    # Common random numbers: every candidate in every generation is judged on
    # the same set of games.  Without this the score of a generation depends on
    # which seeds it happened to draw, and the learning curve measures the
    # seeds rather than the doctrine.
    match_seeds = [seed * 1000 + g for g in range(games)]

    def evaluate(vector) -> float:
        total = 0.0
        for match_seed in match_seeds:
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
        return total / len(match_seeds)

    incumbent = evaluate(mean)
    best_overall, best_score = weight_dict(mean), incumbent

    for generation in range(generations):
        candidates = rng.normal(mean, spread, (population, len(mean)))
        scores = np.array([evaluate(v) for v in candidates])
        order = np.argsort(scores)[::-1]
        keep = candidates[order[:elite]]
        proposal = keep.mean(axis=0)
        proposal_score = evaluate(proposal)
        # Only move if the new centre is actually better on the same games.
        # Plain cross-entropy walks downhill happily when the elite sample was
        # lucky, which on a game this noisy is most of the time.
        if proposal_score >= incumbent:
            mean, incumbent = proposal, proposal_score
        # Injected noise: the standard deviation of four elite samples collapses
        # within a couple of generations and the search stops exploring, so the
        # elite spread is floored by a separately decayed exploration term.
        explore = sigma * (0.75 ** (generation + 1))
        spread = np.maximum(keep.std(axis=0), explore)
        top = float(scores[order[0]])
        if top > best_score:
            best_score, best_overall = top, weight_dict(candidates[order[0]])
        log.add(generation=generation, mean_score=float(scores.mean()),
                best_score=top, incumbent_score=incumbent,
                weights=weight_dict(mean))
        if progress is not None:
            progress(generation, float(scores.mean()), top)
    if incumbent > best_score:
        best_overall, best_score = weight_dict(mean), incumbent
    return best_overall, log


def state_side(side: str) -> str:
    return IMPERIAL if side == ZHODANI else ZHODANI


# --------------------------------------------------------------------------
def train_value_network(games: int = 18, epochs: int = 80, hidden: int = 12,
                        seed: int = 0, network: ValueNetwork | None = None,
                        max_turns: int = 45, agents=None, progress=None,
                        side: str = ZHODANI, weights: dict | None = None):
    """Regression on played games: predict the final margin from a position.

    One network serves both players -- ``features.perspective`` flips the state
    vector and ``NeuralAgent`` negates the score for the Imperium -- so games
    are collected from both seats.  ``side`` chooses whose doctrine weights are
    jittered to generate the learner's play, not which seat is sampled.

    Returns ``(network, final_training_loss)``.  The loss is not a measure of
    quality: see ``evaluator_report`` for that, and read the module docstring
    for why the two differ so much here.
    """
    net = network or ValueNetwork(hidden=hidden, seed=seed)
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[float] = []
    margins: list[float] = []

    # Two things have to be right or the fit is worthless.
    #
    # Outcome spread: self-play against one fixed opponent produces games that
    # all end the same way, and a regression on near-identical labels just
    # learns their mean.  So the opponent is varied and the learner's own
    # doctrine is jittered.
    #
    # Distribution match: training only on NeuralAgent games and then using the
    # network to judge ordinary heuristic games gives *negative* held-out
    # correlation -- the network learns the quirks of one agent's trajectories.
    # So the learner's own seat alternates between the network agent and the
    # standard doctrines, which is the mix the evaluator actually meets.
    opponents = [RandomAgent, HeuristicAgent, ScriptedAgent]

    for g in range(games):
        match_seed = seed * 7919 + g
        if agents is None:
            base = np.asarray(weight_vector(weights))
            jitter = rng.normal(0.0, 0.35, len(base))
            # One network serves both players, so it is trained on a balanced
            # mixture of seats.  Collecting only from one seat gave a network
            # that predicted that seat's games well and the other seat's
            # backwards.
            seat = side if g % 2 == 0 else state_side(side)
            if g % 4 < 2:
                learner = NeuralAgent(seat, network=net,
                                      weights=weight_dict(base + jitter),
                                      seed=match_seed + 1)
            else:
                learner = opponents[(g // 2) % len(opponents)](
                    seat, seed=match_seed + 1)
            other = opponents[g % len(opponents)](state_side(seat),
                                                  seed=match_seed)
            imperial, zhodani = ((other, learner) if seat == ZHODANI
                                 else (learner, other))
        else:
            imperial, zhodani = agents(match_seed)
        margin, result, positions = play_match(imperial, zhodani,
                                               seed=match_seed,
                                               max_turns=max_turns, record=True)
        label = math.tanh(margin / 200.0)
        margins.append(margin)
        for vector in positions:
            xs.append(features.perspective(vector, ZHODANI))
            ys.append(label)
        if progress is not None:
            progress(g, margin, result)
    if not xs:
        return net, 0.0
    x = np.asarray(xs)
    y = np.asarray(ys)
    loss = net.train(x, y, epochs=epochs, lr=0.03, seed=seed)
    net.label_spread = float(np.std(y))          # type: ignore[attr-defined]
    net.margin_spread = float(np.std(margins))   # type: ignore[attr-defined]
    return net, loss


def evaluator_report(net: ValueNetwork, games: int = 4, seed: int = 99,
                     max_turns: int = 30):
    """How well the network predicts outcomes on games it was not trained on.

    A regression loss near zero means nothing if every training label was the
    same, and hand-built probe positions do not help either: setting eighty
    worlds to Zhodani control on turn 1 is a position no game ever reaches, so
    scoring it measures extrapolation rather than judgement.  The honest test
    is held-out games -- play some, and see whether the network's opinion of
    each position correlates with how that game actually ended.

    Returns the correlation, the spread of predictions (near zero means the
    network has collapsed onto the mean), and the mean absolute error.
    """
    xs: list[np.ndarray] = []
    ys: list[float] = []
    game_of: list[int] = []
    opponents = [RandomAgent, HeuristicAgent, ScriptedAgent]
    for g in range(games):
        match_seed = seed * 31 + g
        imperial = opponents[g % len(opponents)](IMPERIAL, seed=match_seed)
        zhodani = opponents[(g + 1) % len(opponents)](ZHODANI, seed=match_seed + 1)
        margin, _, positions = play_match(imperial, zhodani, seed=match_seed,
                                          max_turns=max_turns, record=True)
        label = math.tanh(margin / 200.0)
        for vector in positions:
            xs.append(features.perspective(vector, ZHODANI))
            ys.append(label)
            game_of.append(g)
    if len(xs) < 3:
        return {"correlation": 0.0, "spread": 0.0, "mae": 0.0, "positions": 0,
                "games": games, "ci": (0.0, 0.0)}
    predictions = np.array([net(v) for v in xs])
    labels = np.asarray(ys)
    groups = np.asarray(game_of)

    def corr(mask) -> float:
        p, l = predictions[mask], labels[mask]
        if p.std() < 1e-9 or l.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(p, l)[0, 1])

    correlation = corr(np.ones(len(predictions), dtype=bool))

    # Every position in a game carries that game's final margin, so the
    # independent sample size is the number of *games*, not positions.  A
    # correlation over a handful of games is close to meaningless, and it will
    # happily read +0.5 on one evaluation seed and -0.3 on the next.  Bootstrap
    # over games so the reported uncertainty makes that visible.
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    samples = []
    for _ in range(200):
        drawn = rng.choice(ids, size=len(ids), replace=True)
        mask = np.concatenate([np.flatnonzero(groups == d) for d in drawn])
        p, l = predictions[mask], labels[mask]
        if p.std() > 1e-9 and l.std() > 1e-9:
            samples.append(float(np.corrcoef(p, l)[0, 1]))
    low, high = (float(np.percentile(samples, 5)),
                 float(np.percentile(samples, 95))) if samples else (0.0, 0.0)
    return {"correlation": correlation,
            "ci": (low, high),
            "spread": float(predictions.std()),
            "label_spread": float(labels.std()),
            "mae": float(np.mean(np.abs(predictions - labels))),
            "positions": len(xs),
            "games": int(len(ids))}


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

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


def play_paired(agent_a, agent_b, seed: int = 0, max_turns: int = 45):
    """Play a matchup twice on the same seed with the seats swapped.

    ``agent_a`` and ``agent_b`` are factories ``f(side, seed) -> Agent``.

    Fifth Frontier War is strongly asymmetric -- the Zhodani win most games --
    and a single game swings by tens of victory points on the setup alone.
    Comparing two agents by their raw margins therefore measures the seat and
    the dice at least as much as the play.  A paired game removes both: the two
    halves share a seed, so they share the deployment, the reinforcement rolls
    and most of the early combat, and the *difference* between the halves is
    what the agents did differently.

    Returns ``(advantage, first_margin, second_margin)`` where ``advantage`` is
    positive when ``agent_a`` outplayed ``agent_b`` across the two seats.
    """
    first, _, _ = play_match(agent_a(IMPERIAL, seed), agent_b(ZHODANI, seed + 1),
                             seed=seed, max_turns=max_turns)
    second, _, _ = play_match(agent_b(IMPERIAL, seed), agent_a(ZHODANI, seed + 1),
                              seed=seed, max_turns=max_turns)
    # a ran the Imperium in the first half and the Consulate in the second, so
    # a's advantage is (-first + second) / 2
    return (second - first) / 2.0, first, second


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
@dataclass
class LeagueLog:
    """Per-generation record for ``train_league``."""
    rows: list[dict] = field(default_factory=list)

    def add(self, **kw) -> None:
        self.rows.append(kw)

    def curve(self, side: str, key: str = "benchmark"):
        gens = [r["generation"] for r in self.rows if r["side"] == side]
        vals = [r[key] for r in self.rows if r["side"] == side]
        return gens, vals


def train_league(generations: int = 6, population: int = 8, elite: int = 3,
                 games: int = 1, sigma: float = 0.45, seed: int = 0,
                 max_turns: int = 40, pool: int = 4,
                 base_weights: dict | None = None, benchmark_games: int = 2,
                 progress=None):
    """Co-evolve doctrines for both sides against a growing pool of champions.

    ``train_weights`` optimises one side against a *fixed* opponent, which
    rewards doctrine that beats that particular opponent -- the classic way to
    produce a specialist that folds against anything else.  Here each side's
    candidates are scored against a sample of the other side's past champions,
    so the target moves as both sides improve.

    Two scores are logged per generation because they answer different
    questions:

    ``score``
        performance against the current opposing pool.  It is *not* comparable
        across generations: the pool improves, so a flat score can mean the
        doctrine improved exactly as fast as its opposition.
    ``benchmark``
        paired advantage against a fixed ``ScriptedAgent``, played on both
        seats.  This one *is* comparable across generations and is the number
        to read for "did it actually get better".

    Returns ``(champions, log)`` where ``champions`` maps each side to its final
    weight dict.
    """
    rng = np.random.default_rng(seed)
    base = np.asarray(weight_vector(base_weights), dtype=np.float64)
    mean = {IMPERIAL: base.copy(), ZHODANI: base.copy()}
    spread = {IMPERIAL: np.full(base.shape, sigma),
              ZHODANI: np.full(base.shape, sigma)}
    league = {IMPERIAL: [weight_dict(base)], ZHODANI: [weight_dict(base)]}
    incumbent = {IMPERIAL: None, ZHODANI: None}
    log = LeagueLog()

    def evaluate(side: str, vector, opponents, seeds) -> float:
        total = 0.0
        for opponent_weights in opponents:
            for match_seed in seeds:
                learner = HeuristicAgent(side, weight_dict(vector),
                                         seed=match_seed, label="candidate")
                other = HeuristicAgent(state_side(side), opponent_weights,
                                       seed=match_seed + 7)
                if side == ZHODANI:
                    margin, _, _ = play_match(other, learner, seed=match_seed,
                                              max_turns=max_turns)
                else:
                    margin, _, _ = play_match(learner, other, seed=match_seed,
                                              max_turns=max_turns)
                total += score_for(side, margin)
        return total / max(1, len(opponents) * len(seeds))

    def benchmark(side: str, weights) -> float:
        """Paired advantage over the stock ScriptedAgent, both seats."""
        learner = lambda s, d: HeuristicAgent(s, weights, seed=d)
        stock = lambda s, d: ScriptedAgent(s, seed=d)
        total = 0.0
        for g in range(benchmark_games):
            advantage, _, _ = play_paired(learner, stock,
                                          seed=seed * 613 + g,
                                          max_turns=max_turns)
            total += advantage
        return total / max(1, benchmark_games)

    for generation in range(generations):
        for side in (ZHODANI, IMPERIAL):
            opponents = league[state_side(side)][-pool:]
            seeds = [seed * 1000 + generation * 31 + g for g in range(games)]
            if incumbent[side] is None:
                incumbent[side] = evaluate(side, mean[side], opponents, seeds)
            else:
                # the pool moved, so the incumbent has to be re-scored on it
                incumbent[side] = evaluate(side, mean[side], opponents, seeds)
            candidates = rng.normal(mean[side], spread[side],
                                    (population, len(base)))
            scores = np.array([evaluate(side, v, opponents, seeds)
                               for v in candidates])
            order = np.argsort(scores)[::-1]
            keep = candidates[order[:elite]]
            proposal = keep.mean(axis=0)
            proposal_score = evaluate(side, proposal, opponents, seeds)
            accepted = proposal_score >= incumbent[side]
            if accepted:
                mean[side], incumbent[side] = proposal, proposal_score
            explore = sigma * (0.75 ** (generation + 1))
            spread[side] = np.maximum(keep.std(axis=0), explore)
            # The pool must gain *variety*, not repetition.  Appending the
            # incumbent fills it with copies of one doctrine the moment the
            # incumbent stops moving, and a league of clones is just a fixed
            # opponent wearing several hats.  The best candidate of the
            # generation is a genuinely different opponent whether or not it
            # displaced the champion.
            league[side].append(weight_dict(candidates[order[0]]))
            if len(league[side]) > pool * 3:
                league[side] = league[side][-pool * 3:]
            mark = benchmark(side, weight_dict(mean[side]))
            log.add(generation=generation, side=side,
                    score=float(incumbent[side]), benchmark=float(mark),
                    best_candidate=float(scores[order[0]]),
                    proposal=float(proposal_score), accepted=bool(accepted),
                    pool=len(opponents), weights=weight_dict(mean[side]))
            if progress is not None:
                progress(generation, side, float(incumbent[side]), float(mark))

    champions = {side: weight_dict(mean[side]) for side in mean}
    return champions, log


# --------------------------------------------------------------------------
#: how many turns ahead the ``delta`` target looks, and the victory-point
#: swing that saturates it
DELTA_HORIZON = 6
DELTA_SCALE = 40.0


def margins_of(vectors) -> list[float]:
    """Recover the victory margin at each recorded position.

    ``vp_margin`` is one of the recorded features, so the margin at every turn
    of a game comes back out of the feature matrix without the engine having to
    hand it over separately.
    """
    column = features.INDEX["vp_margin"]
    return [float(v[column]) * 300.0 for v in vectors]


def label_positions(vectors, final_margin: float, target: str = "final",
                    horizon: int = DELTA_HORIZON) -> list[float]:
    """One label per recorded position, in ``[-1, 1]``, Zhodani-positive.

    ``final``
        every position in a game carries that game's final margin.  Simple, and
        the reason the evaluator is so weak: a game is one independent sample
        however many positions it contributes, so seventy games is seventy
        samples.
    ``delta``
        each position carries the change in margin over the next ``horizon``
        turns.  Neighbouring positions now have genuinely different labels, so
        a run of seventy games is thousands of samples rather than seventy, and
        the quantity being predicted -- is this position about to improve --
        is the one a thermostat and a search leaf actually want.
    """
    margins = margins_of(vectors)
    if target == "final":
        return [math.tanh(final_margin / 200.0)] * len(margins)
    if target != "delta":
        raise ValueError("unknown target %r" % (target,))
    horizon = max(1, int(horizon))
    ahead = margins + [final_margin]
    end = len(margins)
    return [math.tanh((ahead[min(i + horizon, end)] - margins[i]) / DELTA_SCALE)
            for i in range(end)]


def train_value_network(games: int = 18, epochs: int = 80, hidden: int = 12,
                        seed: int = 0, network: ValueNetwork | None = None,
                        max_turns: int = 45, agents=None, progress=None,
                        side: str = ZHODANI, weights: dict | None = None,
                        target: str = "final", horizon: int = DELTA_HORIZON):
    """Regression on played games: predict how a position turns out.

    One network serves both players -- ``features.perspective`` flips the state
    vector and ``NeuralAgent`` negates the score for the Imperium -- so games
    are collected from both seats.  ``side`` chooses whose doctrine weights are
    jittered to generate the learner's play, not which seat is sampled.

    ``target`` selects what the network is asked to predict; see
    ``label_positions``.  It is recorded on the network, because a net trained
    on one target means something quite different from a net trained on the
    other and the two must not be silently swapped.

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
        margins.append(margin)
        labels = label_positions(positions, margin, target, horizon)
        for vector, label in zip(positions, labels):
            xs.append(features.perspective(vector, ZHODANI))
            ys.append(label)
        if progress is not None:
            progress(g, margin, result)
    if not xs:
        return net, 0.0
    x = np.asarray(xs)
    y = np.asarray(ys)
    loss = net.train(x, y, epochs=epochs, lr=0.03, seed=seed)
    net.target = target                          # type: ignore[attr-defined]
    net.horizon = horizon                        # type: ignore[attr-defined]
    net.label_spread = float(np.std(y))          # type: ignore[attr-defined]
    net.margin_spread = float(np.std(margins))   # type: ignore[attr-defined]
    return net, loss


def evaluator_report(net: ValueNetwork, games: int = 4, seed: int = 99,
                     max_turns: int = 30, target: str | None = None,
                     horizon: int | None = None):
    """How well the network predicts outcomes on games it was not trained on.

    A regression loss near zero means nothing if every training label was the
    same, and hand-built probe positions do not help either: setting eighty
    worlds to Zhodani control on turn 1 is a position no game ever reaches, so
    scoring it measures extrapolation rather than judgement.  The honest test
    is held-out games -- play some, and see whether the network's opinion of
    each position correlates with what actually happened next.

    The network is scored against the target it was trained on: a net fitted to
    final margins is asked about final margins, one fitted to margin deltas is
    asked about deltas.  Grading a delta net on final margins would report a
    number close to zero and mean nothing by it.

    Returns the correlation, the spread of predictions (near zero means the
    network has collapsed onto the mean), and the mean absolute error.
    """
    target = target or getattr(net, "target", "final")
    horizon = horizon or getattr(net, "horizon", DELTA_HORIZON)
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
        labels = label_positions(positions, margin, target, horizon)
        for vector, label in zip(positions, labels):
            xs.append(features.perspective(vector, ZHODANI))
            ys.append(label)
            game_of.append(g)
    if len(xs) < 3:
        return {"correlation": 0.0, "spread": 0.0, "mae": 0.0, "positions": 0,
                "games": games, "ci": (0.0, 0.0), "target": target}
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

    # A correlation with the outcome is not the same as judgement.  The feature
    # vector already contains the current victory margin, and late in a game
    # that *is* the answer, so a network that simply echoes one column scores
    # well.  Report that trivial baseline alongside, and report the early game
    # separately -- before the score has revealed the result is the only place
    # where an evaluator can show it knows anything.
    margin_column = np.asarray(xs)[:, features.INDEX["vp_margin"]]
    turn_column = np.asarray(xs)[:, features.INDEX["turn"]]

    def corr_of(a, b) -> float:
        if len(a) < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    baseline = corr_of(margin_column, labels)
    early = turn_column < (12.0 / 60.0)
    return {"correlation": correlation,
            "target": target,
            "horizon": horizon,
            "ci": (low, high),
            "baseline_correlation": baseline,
            "skill_over_baseline": correlation - baseline,
            "early_correlation": corr_of(predictions[early], labels[early]),
            "early_baseline": corr_of(margin_column[early], labels[early]),
            "early_positions": int(early.sum()),
            "spread": float(predictions.std()),
            "label_spread": float(labels.std()),
            "mae": float(np.mean(np.abs(predictions - labels))),
            "positions": len(xs),
            "games": int(len(ids))}


# --------------------------------------------------------------------------
def tournament(entries: dict, games: int = 3, max_turns: int = 45,
               seed: int = 0, progress=None, paired: bool = True):
    """Round-robin over paired games, with standard errors.

    ``entries`` maps a label to a factory ``f(side, seed) -> Agent``.

    Each pairing is played as mirrored games on a shared seed (see
    ``play_paired``), so the number reported for ``a`` against ``b`` is how much
    better ``a`` played than ``b`` *across both seats* -- not how much either
    won by, which in this game is dominated by which seat they drew.  The
    Zhodani win most games from any competent doctrine, so raw margins rank the
    seats rather than the agents.

    Returns ``(table, summary)``.  ``table[(a, b)]`` is a's advantage over b in
    victory points, antisymmetric so ``table[(b, a)] == -table[(a, b)]``.  Each
    summary entry carries ``average`` and ``stderr``; a difference smaller than
    a couple of standard errors is not a real difference.

    Set ``paired=False`` for the old behaviour: raw margins by seat ordering.
    """
    names = list(entries)
    table: dict = {}
    samples: dict = {n: [] for n in names}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            advantages = []
            for g in range(games):
                match_seed = seed * 131 + (abs(hash((a, b))) % 1000) + g
                if paired:
                    advantage, first, second = play_paired(
                        entries[a], entries[b], seed=match_seed,
                        max_turns=max_turns)
                    if progress is not None:
                        progress(a, b, g, advantage, (first, second))
                else:
                    margin, result, _ = play_match(
                        entries[a](IMPERIAL, match_seed),
                        entries[b](ZHODANI, match_seed + 1),
                        seed=match_seed, max_turns=max_turns)
                    advantage = -margin
                    if progress is not None:
                        progress(a, b, g, advantage, result)
                advantages.append(advantage)
            mean = sum(advantages) / len(advantages)
            table[(a, b)] = mean
            table[(b, a)] = -mean
            samples[a].extend(advantages)
            samples[b].extend(-x for x in advantages)

    summary = {}
    for name in names:
        values = samples[name]
        mean = sum(values) / len(values) if values else 0.0
        if len(values) > 1:
            variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            stderr = math.sqrt(variance / len(values))
        else:
            stderr = 0.0
        summary[name] = {"average": mean, "stderr": stderr,
                         "pairings": len(values),
                         "wins": sum(1 for v in values if v > 0),
                         "losses": sum(1 for v in values if v < 0)}
    return table, summary


def seat_bias(agent=None, games: int = 6, max_turns: int = 45, seed: int = 0):
    """How much the Zhodani seat is worth, holding the doctrine constant.

    Plays an agent against itself and averages the margin.  Any result far from
    zero is a property of the game as implemented, not of the players, and is
    the baseline every other comparison has to be read against.
    """
    factory = agent or (lambda side, s: ScriptedAgent(side, seed=s))
    margins = []
    for g in range(games):
        match_seed = seed * 977 + g
        margin, _, _ = play_match(factory(IMPERIAL, match_seed),
                                  factory(ZHODANI, match_seed + 1),
                                  seed=match_seed, max_turns=max_turns)
        margins.append(margin)
    mean = sum(margins) / len(margins)
    if len(margins) > 1:
        variance = sum((m - mean) ** 2 for m in margins) / (len(margins) - 1)
        stderr = math.sqrt(variance / len(margins))
    else:
        stderr = 0.0
    return {"mean_margin": mean, "stderr": stderr, "margins": margins,
            "games": len(margins)}


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

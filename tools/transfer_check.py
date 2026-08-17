"""Does a value network trained on one game predict anything in the other?

This is the question the ``strategy`` package exists to make askable, and it
has a cheap, honest answer that does not require training an agent first.

Fit a network on positions from game A, labelled with how game A turned out.
Then show it positions from game B and ask how well its output correlates with
how *those* games turned out.  If the shared encoding carries strategic
content, a network that learned "holding more, with more force, later in the
game, is good" on the Spinward Marches should say something true about Terra.
If it carries nothing, the cross-game correlation will sit on zero and the
right conclusion is that these sixteen numbers are not enough.

**A correlation against a constant is undefined, not zero**, and the two read
identically if you only print the number.  So the outcome spread of each game's
label set is checked first and reported: if one side's games all ended the same
way, no verdict involving it means anything, and the script says so instead of
printing a confident zero.  That is not hypothetical -- it is what the first
run did, because the *Invasion: Earth* doctrine never takes a city, so all
fourteen games ended on the same label.

Three numbers are printed per direction and they answer different questions:

* **own-game correlation** -- did the network learn anything at all?  A near-
  zero here means the run is broken and the transfer number is meaningless.
* **cross-game correlation** -- the actual result.
* **baseline** -- what the single ``score_margin`` slot achieves on its own,
  with no network.  A transfer number that does not beat this has demonstrated
  that one feature transfers, not that a network does.

    python tools/transfer_check.py [games_per_side] [max_turns] [seed]

The seed picks the block of games, so a result can be confirmed on a block it
was not found on -- which is the only way to tell a finding from a coincidence.
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                   # noqa: E402

from strategy import DIMENSIONS, FEATURE_NAMES, adapter_for, encode  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ffw", "data")


def agents_for(name: str, seed: int):
    """The stock doctrine for whichever game, on both seats."""
    if name == "ffw":
        from ffw.agents import ScriptedAgent
        from ffw.state import IMPERIAL, ZHODANI
        return {IMPERIAL: ScriptedAgent(IMPERIAL, seed=seed),
                ZHODANI: ScriptedAgent(ZHODANI, seed=seed + 1)}
    from ie.agents import HeuristicAgent
    from ie.state import IMPERIAL, SOLOMANI
    return {IMPERIAL: HeuristicAgent(IMPERIAL, seed=seed),
            SOLOMANI: HeuristicAgent(SOLOMANI, seed=seed + 1)}


def collect(name: str, games: int, max_turns: int, seed: int = 0):
    """Play games, recording every position and how the game finished.

    Positions are taken from the first side's point of view and labelled with
    that side's final normalised margin -- the same "how did this end" label
    ``ffw``'s evaluator uses, and the same known weakness: one label per game
    means the independent sample size is the number of games, not the number of
    rows.  That is acceptable here because the question is only whether the
    correlation is non-zero, not how well it can be fitted.
    """
    adapter = adapter_for(name)
    side = adapter.sides[0]
    rows, labels = [], []
    for g in range(games):
        state = adapter.new_game(seed=seed + g)
        snapshots = []
        agents = agents_for(name, seed + g)
        # step the game by hand so positions can be sampled along the way
        if name == "ffw":
            from ffw.engine import Engine
            engine = Engine(state, agents[adapter.sides[0]],
                            agents[adapter.sides[1]])
        else:
            from ie.engine import Engine
            engine = Engine(state, agents[adapter.sides[0]],
                            agents[adapter.sides[1]],
                            rng=random.Random(seed + g))
        for turn in range(max_turns):
            if state.game_over:
                break
            engine.play_turn()
            if turn % 3 == 0:
                snapshots.append(encode(adapter, state, side))
        label = adapter.normalised_margin(state, side)
        rows.extend(snapshots)
        labels.extend([label] * len(snapshots))
    return np.asarray(rows), np.asarray(labels)


def fit(rows, labels, hidden: int = 12, epochs: int = 300, seed: int = 0):
    """A small MLP, plain gradient descent, no dependencies beyond numpy."""
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0, 0.3, (rows.shape[1], hidden))
    b1 = np.zeros(hidden)
    w2 = rng.normal(0, 0.3, hidden)
    b2 = 0.0
    lr = 0.05
    for _ in range(epochs):
        h = np.tanh(rows @ w1 + b1)
        out = h @ w2 + b2
        error = out - labels
        grad_out = 2.0 * error / len(labels)
        gw2 = h.T @ grad_out
        gb2 = grad_out.sum()
        gh = np.outer(grad_out, w2) * (1 - h ** 2)
        gw1 = rows.T @ gh
        gb1 = gh.sum(axis=0)
        w1 -= lr * gw1
        b1 -= lr * gb1
        w2 -= lr * gw2
        b2 -= lr * gb2
    return (w1, b1, w2, b2)


def predict(net, rows):
    w1, b1, w2, b2 = net
    return np.tanh(rows @ w1 + b1) @ w2 + b2


def correlation(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.std() < 1e-9 or b.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main(games=12, max_turns=18, seed=900):
    start = time.time()
    print("== collecting positions ==", flush=True)
    data = {}
    for name in ("ffw", "ie"):
        rows, labels = collect(name, games, max_turns, seed=seed)
        data[name] = (rows, labels)
        print("   %-4s %4d positions from %d games   %4.0fs"
              % (name, len(rows), games, time.time() - start), flush=True)

    margin_slot = FEATURE_NAMES.index("score_margin")
    report = {"games": games, "max_turns": max_turns, "seed": seed,
              "directions": {}}

    # A label column with no variance makes every correlation involving it
    # zero by construction, and a zero that means "undefined" reads exactly
    # like a zero that means "no relationship".  Say which it is.
    for name, (_rows, labels) in data.items():
        spread = float(np.std(labels))
        report.setdefault("label_spread", {})[name] = spread
        if spread < 1e-9:
            print("\n   !! every %s game ended on the same label (%+.2f)."
                  % (name, float(labels[0])), flush=True)
            print("      Correlations against a constant are undefined, not "
                  "zero, so no", flush=True)
            print("      transfer verdict involving %s means anything until "
                  "its games" % name, flush=True)
            print("      have varied outcomes.", flush=True)

    for source, target in (("ffw", "ie"), ("ie", "ffw")):
        rows, labels = data[source]
        other_rows, other_labels = data[target]
        source_flat = float(np.std(labels)) < 1e-9
        target_flat = float(np.std(other_labels)) < 1e-9
        net = fit(rows, labels, seed=1)
        own = correlation(predict(net, rows), labels)
        cross = correlation(predict(net, other_rows), other_labels)
        baseline = correlation(other_rows[:, margin_slot], other_labels)
        print("\n== %s -> %s ==" % (source, target), flush=True)
        print("   own-game correlation    %+.3f%s"
              % (own, "   (trained on a constant label)" if source_flat else ""),
              flush=True)
        print("   cross-game correlation  %+.3f%s"
              % (cross, "   (graded against a constant label)"
                 if target_flat else ""), flush=True)
        print("   score_margin alone      %+.3f  (the baseline to beat)"
              % baseline, flush=True)
        if source_flat or target_flat:
            verdict = "unmeasurable: one side's outcomes never varied"
        elif abs(own) < 0.15:
            verdict = "inconclusive: the network learned nothing on its own game"
        elif abs(cross) > max(0.15, abs(baseline)):
            verdict = "transfers"
        else:
            verdict = "no better than the score slot alone"
        print("   -> %s" % verdict, flush=True)
        report["directions"]["%s->%s" % (source, target)] = {
            "own": own, "cross": cross, "baseline": baseline,
            "source_label_constant": source_flat,
            "target_label_constant": target_flat,
            "verdict": verdict}

    path = os.path.join(DATA, "transfer_check.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=float)
    print("\n   wrote %s   total %.0f s" % (path, time.time() - start),
          flush=True)
    return report


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12,
         int(sys.argv[2]) if len(sys.argv) > 2 else 18,
         int(sys.argv[3]) if len(sys.argv) > 3 else 900)

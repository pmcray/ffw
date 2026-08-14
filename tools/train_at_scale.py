"""Train the reference agents at a sample size where the result is measurable.

The notebook-scale runs (a handful of games per candidate) cannot distinguish a
20-point improvement from noise, so this script is the background job: league
co-evolution for the doctrines, a large self-play regression for the value
network, and -- most importantly -- a **verification** step that asks whether
the trained agents actually beat the stock ones, with standard errors, using
paired games.

    python tools/train_at_scale.py                 # everything, ~70 minutes
    python tools/train_at_scale.py league          # doctrines only
    python tools/train_at_scale.py network         # value network only
    python tools/train_at_scale.py verify          # just re-run the check

Training that cannot be shown to have helped is not worth committing, so the
verification result is printed last and saved into each agent's metadata.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffw.agents import HeuristicAgent, ScriptedAgent                  # noqa: E402
from ffw.state import IMPERIAL, ZHODANI                               # noqa: E402
from ffw.training import (load_agent, play_paired, save_agent,        # noqa: E402
                          seat_bias, train_league, train_value_network)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ffw", "data")


def elapsed(start):
    return "%5.0fs" % (time.time() - start)


# --------------------------------------------------------------------------
def run_league(start, generations=6, population=10, elite=4, games=2,
               pool=4, max_turns=30, benchmark_games=5, seed=11):
    print("== league co-evolution ==", flush=True)
    champions, log = train_league(
        generations=generations, population=population, elite=elite,
        games=games, pool=pool, max_turns=max_turns,
        benchmark_games=benchmark_games, seed=seed,
        progress=lambda g, side, score, mark: print(
            "   gen %d %-8s pool %+8.1f  vs stock %+7.1f   %s"
            % (g, side, score, mark, elapsed(start)), flush=True))
    for side in (IMPERIAL, ZHODANI):
        gens, marks = log.curve(side, "benchmark")
        print("   %-8s vs-stock curve: %s"
              % (side, " ".join("%+.0f" % m for m in marks)), flush=True)
    path = os.path.join(DATA, "league_champions.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"champions": champions,
                   "log": [{k: v for k, v in row.items() if k != "weights"}
                           for row in log.rows]}, fh, indent=1)
    print("   wrote %s  %s" % (path, elapsed(start)), flush=True)
    return champions


# --------------------------------------------------------------------------
def run_network(start, champions=None, games=70, max_turns=30, seed=13,
                target="final"):
    """Fit the value network.

    ``target`` is "final" (one label per game, the historical default) or
    "delta" (the change in margin over the next few turns).  The delta target
    fits a demonstrably better evaluator -- six times the skill over the
    trivial baseline -- but whether it makes ``NeuralAgent`` play better is
    still unresolved, so the default stays where the measurements are.  See
    ``tools/evaluator_trial.py``.
    """
    print("== value network, %d games, %s target ==" % (games, target),
          flush=True)
    weights = (champions or {}).get(ZHODANI)
    net, loss = train_value_network(
        games=games, epochs=90, hidden=16, seed=seed, max_turns=max_turns,
        side=ZHODANI, weights=weights, target=target,
        progress=lambda g, m, r: (
            print("   game %2d  margin %+7.1f  %s   %s"
                  % (g, m, r, elapsed(start)), flush=True)
            if g % 10 == 0 else None))
    print("   training loss %.4f, margin spread %.1f VP   %s"
          % (loss, net.margin_spread, elapsed(start)), flush=True)
    from ffw.training import evaluator_report
    report = evaluator_report(net, games=20, seed=99, max_turns=max_turns)
    print("   held-out: corr %+.3f  90%% CI [%+.2f, %+.2f]  over %d games   %s"
          % (report["correlation"], report["ci"][0], report["ci"][1],
             report["games"], elapsed(start)), flush=True)
    print("   trivial baseline %+.3f  ->  skill %+.3f"
          % (report["baseline_correlation"], report["skill_over_baseline"]),
          flush=True)
    return net, report


# --------------------------------------------------------------------------
def verify(champions, start, games=10, max_turns=40):
    """Does the trained doctrine actually beat the stock one?  Paired games."""
    print("== verification: trained vs stock, paired ==", flush=True)
    results = {}
    for side in (ZHODANI, IMPERIAL):
        weights = champions[side]
        trained = lambda s, d: HeuristicAgent(s, weights, seed=d)
        stock = lambda s, d: ScriptedAgent(s, seed=d)
        advantages = []
        for g in range(games):
            advantage, _, _ = play_paired(trained, stock, seed=4000 + g,
                                          max_turns=max_turns)
            advantages.append(advantage)
        mean = sum(advantages) / len(advantages)
        var = sum((a - mean) ** 2 for a in advantages) / max(1, len(advantages) - 1)
        stderr = (var / len(advantages)) ** 0.5
        verdict = ("better" if mean > 2 * stderr else
                   "worse" if mean < -2 * stderr else "indistinguishable")
        results[side] = {"advantage": mean, "stderr": stderr,
                         "games": games, "verdict": verdict}
        print("   %-8s %+7.1f +/- %5.1f VP over %d paired games -> %s   %s"
              % (side, mean, stderr, games, verdict, elapsed(start)), flush=True)
    return results


# --------------------------------------------------------------------------
def main(what="all"):
    start = time.time()
    champions = None
    net = report = None

    if what in ("all", "league"):
        champions = run_league(start)
    else:
        path = os.path.join(DATA, "league_champions.json")
        if os.path.exists(path):
            champions = json.load(open(path, encoding="utf-8"))["champions"]

    if what in ("all", "network"):
        net, report = run_network(start, champions)

    if champions and what in ("all", "league", "verify"):
        results = verify(champions, start)
        for side in (IMPERIAL, ZHODANI):
            meta = {"side": side, "trained_by": "train_at_scale",
                    "verification": results[side],
                    "trained_at": time.strftime("%Y-%m-%d")}
            if report is not None:
                meta["value_network"] = {
                    "correlation": report["correlation"],
                    "ci": report["ci"], "games": report["games"]}
            save_agent(os.path.join(DATA, "trained_%s.json" % side),
                       champions[side], net, meta)
            print("   wrote trained_%s.json" % side, flush=True)

    bias = seat_bias(games=8, max_turns=40, seed=4)
    print("== seat bias %+.1f +/- %.1f VP ==   total %s"
          % (bias["mean_margin"], bias["stderr"], elapsed(start)), flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")

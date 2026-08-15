"""Does labelling positions by margin *delta* give a usable evaluator?

The value network has always been fitted to one label per game -- the final
victory margin, copied onto every position of that game.  That makes the
independent sample size the number of games, so seventy games is seventy
samples however many thousand rows the training matrix has, and it explains why
the fitted network shows almost no skill over simply reading the current score.

The alternative is to label each position with the change in margin over the
next few turns.  Neighbouring positions then carry genuinely different labels,
a run of seventy games becomes thousands of samples, and the quantity being
predicted -- is this position about to improve -- is the one both the
thermostat and the search leaf actually want.

This script trains both on the same games and compares them three ways:

1.  held-out correlation, each net graded on its own target, against the
    trivial baseline of echoing the current score;
2.  paired games between a ``NeuralAgent`` driven by each net -- the decisive
    test, because an evaluator that predicts well and plays no better has not
    earned its place;
3.  paired games with the delta net used as the leaf evaluation of
    ``LookaheadAgent``, which is where a short-horizon prediction should help
    most.

Sample size matters more than it looks.  At ten paired games the second test
read +10.2 +/- 8.1 in the delta network's favour; at sixty it read +1.6 +/- 2.5,
which is nothing.  The default is sixty.

    python tools/evaluator_trial.py [games] [max_turns]
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffw.state import ZHODANI                                         # noqa: E402
from ffw.training import (AgentSpec, evaluate_paired,               # noqa: E402
                          evaluator_report, save_agent,
                          train_value_network)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ffw", "data")


def elapsed(start):
    return "%5.0fs" % (time.time() - start)


def paired(a, b, games, max_turns, seed):
    result = evaluate_paired(a, b, games=games, seed=seed, max_turns=max_turns)
    return {k: v for k, v in result.items() if k != "advantages"}


def main(games=50, max_turns=30, match_games=60, match_turns=40,
         lookahead_games=8):
    # The lookahead matches get their own, much smaller budget: the search
    # agent costs about forty times what the doctrine does per game, so
    # matching the sixty games used for the neural comparison would take most
    # of a day.  Their error bars are correspondingly wide, and are reported.
    start = time.time()
    nets = {}
    for target in ("final", "delta"):
        print("== training on the %s target, %d games ==" % (target, games),
              flush=True)
        net, loss = train_value_network(
            games=games, epochs=90, hidden=16, seed=13, max_turns=max_turns,
            side=ZHODANI, target=target,
            progress=lambda g, m, r: (
                print("   game %2d  margin %+7.1f   %s" % (g, m, elapsed(start)),
                      flush=True) if g % 10 == 0 else None))
        report = evaluator_report(net, games=20, seed=99, max_turns=max_turns)
        print("   loss %.4f   held-out corr %+.3f  90%% CI [%+.2f, %+.2f]"
              % (loss, report["correlation"], report["ci"][0], report["ci"][1]),
              flush=True)
        print("   baseline (echo the score) %+.3f   skill %+.3f   "
              "early game %+.3f vs %+.3f   %s"
              % (report["baseline_correlation"], report["skill_over_baseline"],
                 report["early_correlation"], report["early_baseline"],
                 elapsed(start)), flush=True)
        nets[target] = (net, report)

    print("== the decisive test: does the doctrine play better with it? ==",
          flush=True)
    results = {}
    final_net, delta_net = nets["final"][0], nets["delta"][0]
    results["neural"] = paired(
        AgentSpec(kind="neural", network=delta_net.to_dict()),
        AgentSpec(kind="neural", network=final_net.to_dict()),
        match_games, match_turns, seed=5100)
    print("   neural(delta) vs neural(final)  %+7.1f +/- %5.1f -> %s   %s"
          % (results["neural"]["advantage"], results["neural"]["stderr"],
             results["neural"]["verdict"], elapsed(start)), flush=True)

    for name, net in (("delta", delta_net), ("final", final_net)):
        key = "lookahead_%s" % name
        results[key] = paired(
            AgentSpec(kind="lookahead",
                      options={"evaluator": net.to_dict()}),
            AgentSpec(kind="scripted"),
            lookahead_games, match_turns, seed=5200)
        print("   lookahead(%s leaf) vs scripted  %+7.1f +/- %5.1f -> %s   %s"
              % (name, results[key]["advantage"], results[key]["stderr"],
                 results[key]["verdict"], elapsed(start)), flush=True)

    results["raw_lookahead"] = paired(
        AgentSpec(kind="lookahead"), AgentSpec(kind="scripted"),
        lookahead_games, match_turns, seed=5200)
    print("   lookahead(raw margin leaf) vs scripted  %+7.1f +/- %5.1f -> %s"
          "   %s" % (results["raw_lookahead"]["advantage"],
                     results["raw_lookahead"]["stderr"],
                     results["raw_lookahead"]["verdict"], elapsed(start)),
          flush=True)

    payload = {"reports": {t: {k: v for k, v in r.items() if k != "ci"}
                           for t, (_, r) in nets.items()},
               "matches": results}
    path = os.path.join(DATA, "evaluator_trial.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=float)
    save_agent(os.path.join(DATA, "evaluator_delta.json"), None, delta_net,
               {"target": "delta", "trial": results["neural"],
                "trained_at": time.strftime("%Y-%m-%d")})
    print("   wrote %s   total %s" % (path, elapsed(start)), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50,
         int(sys.argv[2]) if len(sys.argv) > 2 else 30)

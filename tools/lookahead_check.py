"""Is the search agent worth its running time?

Plays the lookahead agent against the stock doctrine in paired games (both
seats, shared seeds) and reports the advantage with a standard error, plus the
seconds per turn it actually costs.  A search agent that cannot be shown to
beat the doctrine it samples from is not a second architecture, it is a slower
first one.

    python tools/lookahead_check.py [games] [max_turns]
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffw.agents import HeuristicAgent, LookaheadAgent, ScriptedAgent  # noqa: E402
from ffw.training import play_paired                                  # noqa: E402


def main(games=6, max_turns=40):
    search = lambda s, d: LookaheadAgent(s, seed=d)
    stock = lambda s, d: ScriptedAgent(s, seed=d)
    doctrine = lambda s, d: HeuristicAgent(s, seed=d)
    start = time.time()
    for name, other in (("scripted", stock), ("heuristic", doctrine)):
        advantages = []
        for g in range(games):
            advantage, first, second = play_paired(search, other, seed=700 + g,
                                                   max_turns=max_turns)
            advantages.append(advantage)
            print("   game %d  %+7.1f  (%.0f / %.0f)   %5.0fs"
                  % (g, advantage, first, second, time.time() - start),
                  flush=True)
        mean = sum(advantages) / len(advantages)
        var = sum((a - mean) ** 2 for a in advantages) / max(1, len(advantages) - 1)
        stderr = (var / len(advantages)) ** 0.5
        verdict = ("better" if mean > 2 * stderr else
                   "worse" if mean < -2 * stderr else "indistinguishable")
        print("== lookahead vs %-9s %+7.1f +/- %5.1f VP over %d paired games"
              " -> %s ==" % (name, mean, stderr, games, verdict), flush=True)
    print("total %.0f s" % (time.time() - start), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6,
         int(sys.argv[2]) if len(sys.argv) > 2 else 40)

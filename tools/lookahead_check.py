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

from ffw.training import AgentSpec, evaluate_paired                   # noqa: E402


def main(games=20, max_turns=40):
    search = AgentSpec(kind="lookahead")
    start = time.time()
    for name, other in (("scripted", AgentSpec(kind="scripted")),
                        ("heuristic", AgentSpec(kind="heuristic"))):
        result = evaluate_paired(
            search, other, games=games, seed=700, max_turns=max_turns,
            progress=lambda g, adv, halves: print(
                "   game %2d  %+7.1f  (%.0f / %.0f)   %5.0fs"
                % (g, adv, halves[0], halves[1], time.time() - start),
                flush=True))
        print("== lookahead vs %-9s %+7.1f +/- %5.1f VP over %d paired games"
              " -> %s ==" % (name, result["advantage"], result["stderr"],
                             games, result["verdict"]), flush=True)
    print("total %.0f s" % (time.time() - start), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20,
         int(sys.argv[2]) if len(sys.argv) > 2 else 40)

"""Re-measure the project's headline numbers against an honest baseline.

The tournament showed ``ScriptedAgent``'s historical opening costs about nine
victory points against plain ``HeuristicAgent``.  Every "better than stock"
verdict in this repository was measured against that agent, so each one is
partly a measurement of a self-inflicted handicap.  Nothing published was wrong,
but "beats stock" means less than it sounded, and the only way to know how much
less is to run it again against an opponent that is not holding a brick.

Three things are re-measured here, all at sample sizes the old engine could not
afford:

* the trained doctrines, against **both** baselines, so the gap between the two
  readings is the size of the handicap;
* the seat bias, which is the number every other comparison is read against;
* the scripted opening itself, at enough games to state its cost precisely.

    python tools/rebaseline.py [games]
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffw.state import IMPERIAL, ZHODANI                               # noqa: E402
from ffw.training import (AgentSpec, evaluate_paired, load_agent,     # noqa: E402
                          paired_series, summarise)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ffw", "data")


def line(label, result, start):
    print("   %-46s %+7.1f +/- %5.1f  %-18s %5.0fs"
          % (label, result["advantage"], result["stderr"], result["verdict"],
             time.time() - start), flush=True)


def main(games=120, max_turns=40):
    start = time.time()
    out = {"games": games, "max_turns": max_turns}

    stock = AgentSpec("scripted")
    plain = AgentSpec("heuristic")

    print("== the handicap itself ==", flush=True)
    handicap = evaluate_paired(plain, stock, games=games, seed=31_000,
                               max_turns=max_turns)
    line("heuristic vs scripted (the opening's cost)", handicap, start)
    out["scripted_opening_cost"] = {k: v for k, v in handicap.items()
                                    if k != "advantages"}

    print("== trained doctrines, against both baselines ==", flush=True)
    out["trained"] = {}
    for side in (IMPERIAL, ZHODANI):
        path = os.path.join(DATA, "trained_%s.json" % side)
        if not os.path.exists(path):
            print("   (no trained_%s.json)" % side, flush=True)
            continue
        weights, _, meta = load_agent(path)
        trained = AgentSpec("heuristic", weights=weights)
        old = meta.get("verification", {})
        row = {"previous": old}
        for name, opponent in (("scripted", stock), ("heuristic", plain)):
            result = evaluate_paired(trained, opponent, games=games,
                                     seed=32_000, max_turns=max_turns)
            line("trained %-8s vs %-9s" % (side, name), result, start)
            row[name] = {k: v for k, v in result.items() if k != "advantages"}
        if old:
            print("      previously reported vs scripted: %+.1f +/- %.1f "
                  "over %d games" % (old.get("advantage", 0.0),
                                     old.get("stderr", 0.0), old.get("games", 0)),
                  flush=True)
        out["trained"][side] = row

    print("== seat bias, the number everything is read against ==", flush=True)
    # Same doctrine on both seats: whatever margin is left is the game, not the
    # players.  Measured with the *unhandicapped* agent, since the scripted
    # opening is itself a doctrine choice.
    margins = []
    for spec, label in ((plain, "heuristic"), (stock, "scripted")):
        halves = []
        paired_series(spec, spec, games=games, seed=33_000, max_turns=max_turns,
                      progress=lambda g, adv, both: halves.append(both[0]))
        stats = summarise(halves)
        print("   %-46s %+7.1f +/- %5.1f (Zhodani-positive)   %5.0fs"
              % ("seat bias, both sides play " + label, stats["advantage"],
                 stats["stderr"], time.time() - start), flush=True)
        out.setdefault("seat_bias", {})[label] = {
            k: v for k, v in stats.items() if k != "advantages"}
        margins.append(stats["advantage"])

    path = os.path.join(DATA, "rebaseline.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print("\n   wrote %s   total %.0f s" % (path, time.time() - start), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120,
         int(sys.argv[2]) if len(sys.argv) > 2 else 40)

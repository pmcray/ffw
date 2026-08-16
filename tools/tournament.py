"""The round robin, re-run against a game that plays to its own ending.

Every tournament number published before this ran to turn 40 or 45.  The
rulebook sets no turn limit -- play continues "until a player achieves an
automatic victory or until an armistice occurs" -- and rule 8's armistice
boundary sits at turn 52, so those runs stopped the game before the condition
that decides it could fire.  The rankings were still valid *as rankings by
margin*, which is what they claimed to be, but nothing in them said who won.

So each pairing is played once and scored twice:

* by **victory-point margin**, which is continuous and therefore the sharper
  instrument for ranking play; and
* by **victory level**, the nine-step table the rules actually settle on, with
  the armistice shift applied.

The two can disagree, and where they do the disagreement is the finding.  A
doctrine that grinds out fifty extra victory points inside a level boundary has
gained nothing the rules recognise, while one that nudges a stalemate into a
marginal victory has gained everything and barely moved the margin.

Sixty turns per game, so the boundary is reachable.  That is half again the
cost of the old runs, which parallel evaluation has made affordable.

    python tools/tournament.py [games] [max_turns]
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffw.training import (AgentSpec, level_index, load_agent,         # noqa: E402
                          paired_both, summarise)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ffw", "data")

#: The lookahead agent costs roughly forty times what the doctrine does per
#: game, so including it at these sample sizes would take most of a day.  It
#: has its own comparison in ``tools/lookahead_check.py``.
ENTRIES = {
    "heuristic": AgentSpec("heuristic"),
    "scripted": AgentSpec("scripted"),
    "doctrine": AgentSpec("doctrine"),
    "random": AgentSpec("random"),
}


def entries() -> dict:
    """The field, plus the trained doctrines if they are committed."""
    out = dict(ENTRIES)
    net = os.path.join(DATA, "evaluator_delta.json")
    if os.path.exists(net):
        _, network, _ = load_agent(net)
        if network is not None:
            out["neural"] = AgentSpec("neural", network=network.to_dict())
    for side in ("imperial", "zhodani"):
        path = os.path.join(DATA, "trained_%s.json" % side)
        if os.path.exists(path):
            weights, _, _ = load_agent(path)
            out["trained_%s" % side] = AgentSpec("heuristic", weights=weights)
    return out


def rank(samples: dict, key: str, unit: str) -> list:
    rows = []
    for name, values in samples.items():
        stats = summarise(values)
        rows.append((name, stats["advantage"], stats["stderr"],
                     sum(1 for v in values if v > 0),
                     sum(1 for v in values if v < 0)))
    rows.sort(key=lambda r: -r[1])
    width = max(len(r[0]) for r in rows)
    print("\n   %-*s %9s %8s   %s" % (width, key, unit, "stderr", "record"),
          flush=True)
    for name, avg, err, won, lost in rows:
        print("   %-*s %+9.2f  +/-%5.2f   %d-%d"
              % (width, name, avg, err, won, lost), flush=True)
    return [{"agent": n, "average": a, "stderr": e, "wins": w, "losses": l}
            for n, a, e, w, l in rows]


def main(games=40, max_turns=60):
    start = time.time()
    field = entries()
    names = list(field)
    print("== %d agents, %d paired games each, %d turns ==\n"
          % (len(names), games, max_turns), flush=True)

    margin_samples = {n: [] for n in names}
    level_samples = {n: [] for n in names}
    results: dict[str, int] = {}
    pairings = []

    for i, a in enumerate(names):
        for b in names[i + 1:]:
            seed = 60_000 + 977 * i + 13 * names.index(b)
            # both scores off one set of games: same seeds, so they were
            # always the same games, and playing them twice bought nothing
            both = paired_both(field[a], field[b], games=games, seed=seed,
                               max_turns=max_turns)
            margin, level = both["margin"], both["level"]
            margin_samples[a] += margin["advantages"]
            margin_samples[b] += [-v for v in margin["advantages"]]
            level_samples[a] += level["advantages"]
            level_samples[b] += [-v for v in level["advantages"]]
            for name, count in level["results"].items():
                results[name] = results.get(name, 0) + count
            # only worth flagging when both measures are individually
            # decisive and point opposite ways; two numbers hovering around
            # zero disagreeing is arithmetic, not a finding
            decisive = (abs(margin["advantage"]) > 2 * margin["stderr"]
                        and abs(level["advantage"]) > 2 * level["stderr"])
            agree = (margin["advantage"] > 0) == (level["advantage"] > 0)
            print("   %-18s vs %-18s  margin %+7.1f +/-%5.1f   "
                  "level %+5.2f +/-%4.2f  %s   %4.0fs"
                  % (a, b, margin["advantage"], margin["stderr"],
                     level["advantage"], level["stderr"],
                     "<- DISAGREE" if decisive and not agree else "",
                     time.time() - start), flush=True)
            pairings.append({
                "a": a, "b": b,
                "margin": {k: v for k, v in margin.items() if k != "advantages"},
                "level": {k: v for k, v in level.items() if k != "advantages"}})

    payload = {"games": games, "max_turns": max_turns,
               "by_margin": rank(margin_samples, "by victory-point margin",
                                 "VP"),
               "by_level": rank(level_samples, "by victory level", "levels"),
               "pairings": pairings,
               "results": dict(sorted(results.items(),
                                      key=lambda kv: level_index(kv[0])))}

    print("\n   how the games ended, across the whole tournament:", flush=True)
    total = sum(results.values())
    for name, count in payload["results"].items():
        print("     %-28s %4d  %4.1f%%"
              % (name, count, 100.0 * count / total), flush=True)

    path = os.path.join(DATA, "tournament.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=float)
    print("\n   wrote %s   total %.0f s" % (path, time.time() - start),
          flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 40,
         int(sys.argv[2]) if len(sys.argv) > 2 else 60)

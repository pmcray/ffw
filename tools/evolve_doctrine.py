"""Evolve a doctrine by asking Claude for edits and measuring every one.

    export ANTHROPIC_API_KEY=...          # or: ant auth login
    pip install anthropic

    python tools/evolve_doctrine.py --generations 5 --proposals 4 --games 12

Each generation the model is shown how the current doctrine actually performed
-- where the victory points went, which rules never fired once in a whole war,
what has already been tried and what it measured -- and proposes edits in the
rule vocabulary.  Every proposal is played against the incumbent over paired
games on shared seeds; only a winner is kept, and only if it wins again on
seeds it has never seen.

    --dry-run    run the whole loop against a scripted proposer instead of the
                 API: no key, no network, same code path.  Use it to check the
                 harness before spending tokens.

The result is written to ffw/data/doctrine_evolved.json and can be played
directly:  DoctrineAgent('imperial', 'ffw/data/doctrine_evolved.json')
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffw.agents import ScriptedAgent                                  # noqa: E402
from ffw.agents.doctrine import RuleSet, default_rules                # noqa: E402
from ffw.llm import (ClaudeProposer, StubProposer, evaluate, evolve,  # noqa: E402
                     save_run)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ffw", "data")

#: what --dry-run proposes: one sound idea, one syntactically invalid one (so
#: the rejection path is exercised), one posture change.
DRY_RUN_SCRIPT = [
    [
        {"title": "occupy free worlds further out",
         "hypothesis": "the Imperium should end holding 2-3 more undefended "
                       "neutrals, worth roughly 6 VP",
         "edits": [{"action": "replace", "name": "occupy free worlds within reach",
                    "when": ["claimable", "carrying>=1", "distance<=5"],
                    "then": "prefer 2.4",
                    "rationale": "the free worlds sit further from the Imperial "
                                 "concentration than three parsecs",
                    "set": []}]},
        {"title": "an intentionally malformed proposal",
         "hypothesis": "this should be rejected before it is ever played",
         "edits": [{"action": "add", "name": "nonsense",
                    "when": ["moon_phase>=3"], "then": "prefer 1",
                    "rationale": "", "set": []}]},
        {"title": "hold fire longer before breaking off",
         "hypothesis": "fewer disengagements should mean more worlds taken",
         "edits": [{"action": "posture", "name": "", "when": [], "then": "",
                    "rationale": "", "set": ["disengage_ratio=2.3"]}]},
    ],
]


def report(kind, generation, benchmark, proposal):
    if kind == "generation":
        print("\n== generation %d ==  incumbent vs stock: %+.1f +/- %.1f (%s)"
              % (generation, benchmark["advantage"], benchmark["stderr"],
                 benchmark["verdict"]), flush=True)
    elif kind == "proposal":
        print("   %s" % proposal.summary(), flush=True)
        if proposal.hypothesis and not proposal.error:
            print("      hypothesis: %s" % proposal.hypothesis, flush=True)
    elif kind == "rejected":
        print("   -> %s did not survive confirmation; keeping the incumbent"
              % proposal.title, flush=True)
    else:
        print("   -> kept: %s" % proposal.title, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--proposals", type=int, default=4)
    parser.add_argument("--games", type=int, default=12,
                        help="paired games per candidate")
    parser.add_argument("--confirm-games", type=int, default=8,
                        help="re-measure the winner on fresh seeds (0 to skip)")
    parser.add_argument("--accept-sigma", type=float, default=1.0,
                        help="standard errors a winner must clear")
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--effort", default="high",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--start", default=None,
                        help="rule set JSON to start from (default: the seed)")
    parser.add_argument("--out", default=os.path.join(DATA, "doctrine_evolved.json"))
    parser.add_argument("--dry-run", action="store_true",
                        help="scripted proposals; no key and no network")
    args = parser.parse_args(argv)

    start = RuleSet.load(args.start) if args.start else default_rules()
    proposer = (StubProposer(DRY_RUN_SCRIPT) if args.dry_run
                else ClaudeProposer(model=args.model, effort=args.effort))
    if args.dry_run:
        print("dry run: scripted proposals, no API call", flush=True)
    else:
        print("proposing with %s at effort %s" % (args.model, args.effort),
              flush=True)

    begin = time.time()
    stock = lambda s, d: ScriptedAgent(s, seed=d)
    before = evaluate(start, stock, games=args.games, max_turns=args.max_turns)

    ruleset, log = evolve(
        generations=args.generations, proposals=args.proposals,
        games=args.games, ruleset=start, proposer=proposer,
        max_turns=args.max_turns, accept_sigma=args.accept_sigma,
        confirm_games=args.confirm_games, progress=report)

    after = evaluate(ruleset, stock, games=args.games, max_turns=args.max_turns,
                     seed=99_000)
    print("\n== result after %.0f s ==" % (time.time() - begin), flush=True)
    print("   rules %d -> %d" % (len(start), len(ruleset)), flush=True)
    print("   vs stock, before: %+7.1f +/- %5.1f (%s)"
          % (before["advantage"], before["stderr"], before["verdict"]), flush=True)
    print("   vs stock, after:  %+7.1f +/- %5.1f (%s)"
          % (after["advantage"], after["stderr"], after["verdict"]), flush=True)
    accepted = [r for r in log.rows if r.get("accepted")]
    rejected = [r for r in log.rows if r.get("error")]
    print("   %d proposals, %d unparseable, %d kept"
          % (len(log.rows), len(rejected), len(accepted)), flush=True)

    save_run(args.out, ruleset, log,
             {"model": args.model if not args.dry_run else "stub",
              "before": before, "after": after,
              "generations": args.generations, "games": args.games})
    ruleset.save(os.path.splitext(args.out)[0] + "_rules.json")
    print("   wrote %s" % args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Train and verify an *Invasion: Earth* doctrine.

    python tools/train_ie.py                       # train the Imperial side
    python tools/train_ie.py --side solomani
    python tools/train_ie.py --generations 8 --population 12 --games 4
    python tools/train_ie.py --verify              # re-check what is committed

The search itself is ``strategy/search.py`` and is shared with *Fifth Frontier
War*; this is the command line, the verification step and the file format.

Nothing is written unless verification says the result is better than the stock
doctrine by more than two standard errors on games it was not trained on.
Training that cannot be shown to have helped is not worth committing, and this
project has a section of its README about the time that lesson cost.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy import doctrine_for, paired_advantage, train      # noqa: E402
from strategy.search import default_workers                     # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "ie", "data")

SIDES = ("imperial", "solomani")


def path_for(side: str) -> str:
    return os.path.join(DATA, "trained_%s.json" % side)


def load(side: str):
    """The committed weights for a side, or ``None`` if none were good enough."""
    try:
        with open(path_for(side), encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None


def verify(side: str, weights: dict, games: int, seed: int, max_turns: int,
           workers: int) -> dict:
    """Paired against the stock doctrine, on seeds training never saw."""
    return paired_advantage("ie", side, weights, games=games, seed=seed,
                            max_turns=max_turns, shape="margin",
                            workers=workers)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", default="imperial", choices=SIDES)
    parser.add_argument("--generations", type=int, default=6)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--elite", type=int, default=3)
    parser.add_argument("--games", type=int, default=3,
                        help="games per candidate while searching")
    parser.add_argument("--verify-games", type=int, default=12)
    parser.add_argument("--turns", type=int, default=24)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--verify", action="store_true",
                        help="only re-check the committed weights")
    args = parser.parse_args(argv)

    if args.verify:
        stored = load(args.side)
        if stored is None:
            print("nothing committed for %s" % args.side)
            return
        report = verify(args.side, stored["weights"], args.verify_games,
                        args.seed + 77, args.turns, args.workers)
        print("committed %s doctrine: %+.3f ± %.3f over %d games -> %s"
              % (args.side, report["advantage"], report["stderr"],
                 report["games"], report["verdict"]))
        return

    def progress(generation, mean, best, incumbent):
        print("  generation %d  mean %+.3f  best %+.3f  incumbent %+.3f"
              % (generation, mean, best, incumbent), flush=True)

    print("training the %s doctrine: %d generations of %d, %d games each, "
          "%d workers" % (args.side, args.generations, args.population,
                          args.games, args.workers))
    weights, log = train("ie", side=args.side, generations=args.generations,
                         population=args.population, elite=args.elite,
                         games=args.games, sigma=args.sigma, seed=args.seed,
                         max_turns=args.turns, workers=args.workers,
                         progress=progress)

    defaults, _cls = doctrine_for("ie")
    print("\nwhat moved:")
    for name in sorted(defaults):
        before, after = defaults[name], weights[name]
        if abs(after - before) > 0.05:
            print("  %-20s %6.2f -> %6.2f" % (name, before, after))

    report = verify(args.side, weights, args.verify_games, args.seed + 77,
                    args.turns, args.workers)
    print("\nverification, paired on seeds training never saw:")
    print("  %+.3f ± %.3f over %d games -> %s"
          % (report["advantage"], report["stderr"], report["games"],
             report["verdict"]))

    if report["verdict"] != "better":
        print("\nnot committed: a doctrine that cannot be shown to have helped "
              "is not an improvement, it is a different set of numbers.")
        return

    payload = {
        "weights": weights,
        "meta": {
            "side": args.side,
            "trained_by": "tools/train_ie.py",
            "trained_at": datetime.date.today().isoformat(),
            "generations": args.generations,
            "population": args.population,
            "games_per_candidate": args.games,
            "max_turns": args.turns,
            "seed": args.seed,
            "verification": {k: report[k] for k in
                             ("advantage", "stderr", "games", "verdict")},
        },
    }
    os.makedirs(DATA, exist_ok=True)
    with open(path_for(args.side), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print("\nwrote %s" % path_for(args.side))


if __name__ == "__main__":
    main()

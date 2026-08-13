"""Train the reference agent for a single side.

``python tools/train_one_side.py imperial`` regenerates only
``ffw/data/trained_imperial.json``, which is useful when one side's run has
already finished or has to be repeated.
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffw.state import IMPERIAL, ZHODANI                       # noqa: E402
from ffw.training import (save_agent, train_value_network,    # noqa: E402
                          train_weights)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "ffw", "data")


def main(side=ZHODANI, generations=6, population=10, elite=4, games=1,
         net_games=8, max_turns=35):
    start = time.time()
    print("== training %s doctrine ==" % side, flush=True)
    weights, log = train_weights(
        side=side, generations=generations, population=population,
        elite=elite, games=games, seed=7, max_turns=max_turns,
        progress=lambda g, m, b: print(
            "   gen %d  mean %+8.1f  best %+8.1f  (%.0fs)"
            % (g, m, b, time.time() - start), flush=True))
    _, incumbent = log.incumbents()
    print("   incumbent curve: %s"
          % " -> ".join("%+.1f" % v for v in incumbent), flush=True)
    print("== self-play value network for %s ==" % side, flush=True)
    net, loss = train_value_network(
        games=net_games, epochs=60, seed=7, max_turns=max_turns,
        side=side, weights=weights,
        progress=lambda g, m, r: print("   game %d  margin %+7.1f  %s"
                                       % (g, m, r), flush=True))
    path = os.path.join(DATA, "trained_%s.json" % side)
    save_agent(path, weights, net,
               {"side": side, "generations": generations,
                "population": population, "games_per_candidate": games,
                "value_loss": loss, "incumbent": incumbent[-1],
                "trained_at": time.strftime("%Y-%m-%d")})
    print("wrote %s (value loss %.4f) in %.0fs"
          % (path, loss, time.time() - start), flush=True)
    return path


if __name__ == "__main__":
    side = sys.argv[1] if len(sys.argv) > 1 else ZHODANI
    if side not in (IMPERIAL, ZHODANI):
        raise SystemExit("side must be 'imperial' or 'zhodani'")
    main(side)

"""Train the reference agents shipped with the repository.

Produces ``ffw/data/trained_zhodani.json`` and ``ffw/data/trained_imperial.json``
(doctrine weights plus a self-play value network) and prints a tournament table
comparing every agent type.  Re-runnable; the committed files are the output of
one run of this script.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffw.agents import HeuristicAgent, NeuralAgent, RandomAgent, ScriptedAgent
from ffw.state import IMPERIAL, ZHODANI
from ffw.training import (save_agent, tournament, train_value_network,
                          train_weights)

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "ffw", "data")


def main(generations=6, population=10, elite=4, games=1, net_games=8,
         max_turns=35):
    start = time.time()
    out = {}
    for side, filename in ((ZHODANI, "trained_zhodani.json"),
                           (IMPERIAL, "trained_imperial.json")):
        print("== training %s doctrine ==" % side, flush=True)
        weights, log = train_weights(
            side=side, generations=generations, population=population,
            elite=elite, games=games, seed=7, max_turns=max_turns,
            progress=lambda g, m, b: print(
                "   gen %d  mean %+8.1f  best %+8.1f  (%.0fs)"
                % (g, m, b, time.time() - start), flush=True))
        print("== self-play value network for %s ==" % side, flush=True)
        net, loss = train_value_network(
            games=net_games, epochs=60, seed=7, max_turns=max_turns,
            side=side, weights=weights,
            progress=lambda g, m, r: print("   game %d  margin %+7.1f  %s"
                                           % (g, m, r), flush=True))
        path = os.path.join(DATA, filename)
        save_agent(path, weights, net,
                   {"side": side, "generations": generations,
                    "population": population, "value_loss": loss,
                    "trained_at": time.strftime("%Y-%m-%d")})
        out[side] = path
        print("   wrote %s (value loss %.4f)" % (path, loss), flush=True)
    print("total %.0fs" % (time.time() - start))
    return out


if __name__ == "__main__":
    main()

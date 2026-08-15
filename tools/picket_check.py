"""Do pickets pay?  Paired games with the behaviour on and off.

Eighteen neutral worlds in the Marches have no defences at all and are worth
fifty-five victory points between them, and left to itself the doctrine
finishes a war with roughly half of them unclaimed.  ``HeuristicAgent`` can
detach a single squadron onto a spare fleet marker to go and sit on one.  This
script measures whether that is worth doing, separately for each side, in
paired games on shared seeds -- and counts the worlds actually claimed, so a
null result can be read as "it did not work" rather than "it did not fire".

    python tools/picket_check.py [games] [max_turns]
"""

from __future__ import annotations

import concurrent.futures as futures
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffw.agents import ScriptedAgent                                  # noqa: E402
from ffw.engine import new_game, play                                 # noqa: E402
from ffw.state import IMPERIAL, ZHODANI                               # noqa: E402
from ffw.training import default_workers, summarise                   # noqa: E402


def _one_pair(job):
    """One seed, played with this side's pickets off and then on.

    Module level so it can cross a process boundary; the whole point of raising
    the sample size is that the games now run across the machine's cores.
    """
    side, seed, max_turns = job
    outcome = {}
    for on in (False, True):
        state = new_game(seed=seed)
        play(state,
             ScriptedAgent(IMPERIAL, seed=seed, pickets=on and side == IMPERIAL),
             ScriptedAgent(ZHODANI, seed=seed + 1, pickets=on and side == ZHODANI),
             max_turns=max_turns)
        outcome[on] = state.victory_margin()
    change = outcome[True] - outcome[False]
    return change if side == ZHODANI else -change


def free_world_tally(games=4, max_turns=40, pickets=True, seed=200):
    """Undefended neutrals held at the end, by count and by victory points.

    The count is the direct measurement of whether the behaviour fires; the
    victory points are what it is worth.  Both are far less noisy than the
    game's final margin, which a single change in fleet routing perturbs
    chaotically, so they are reported alongside it.
    """
    held = {IMPERIAL: [0.0, 0.0], ZHODANI: [0.0, 0.0], None: [0.0, 0.0]}
    for g in range(games):
        state = new_game(seed=seed + g)
        play(state, ScriptedAgent(IMPERIAL, seed=g, pickets=pickets),
             ScriptedAgent(ZHODANI, seed=g + 1, pickets=pickets),
             max_turns=max_turns)
        for world in state.world_map:
            if world.owner != "neutral" or world.defense_battalions or world.sdb:
                continue
            row = held.setdefault(world.control, [0.0, 0.0])
            row[0] += 1.0
            row[1] += world.tech_level / 2.0
    return {k: (v[0] / games, v[1] / games) for k, v in held.items()}


def main(games=120, max_turns=40):
    start = time.time()
    for pickets in (False, True):
        tally = free_world_tally(games=max(8, games // 2), max_turns=max_turns,
                                 pickets=pickets)
        print("pickets %-5s  imperial %4.1f (%4.1f VP)   zhodani %4.1f (%4.1f VP)"
              "   unclaimed %4.1f (%4.1f VP)"
              % (pickets,
                 tally.get(IMPERIAL, (0, 0))[0], tally.get(IMPERIAL, (0, 0))[1],
                 tally.get(ZHODANI, (0, 0))[0], tally.get(ZHODANI, (0, 0))[1],
                 tally.get(None, (0, 0))[0], tally.get(None, (0, 0))[1]),
              flush=True)

    for side in (IMPERIAL, ZHODANI):
        # A paired comparison on the behaviour itself: same seed, same
        # opponent, same seats -- only this side's pickets change.  A seat swap
        # would be wrong here, because the thing under test belongs to one side
        # and swapping seats would hand it to the other.
        jobs = [(side, 900 + g, max_turns) for g in range(games)]
        workers = default_workers(len(jobs))
        if workers > 1:
            with futures.ProcessPoolExecutor(max_workers=workers) as pool:
                gains = list(pool.map(_one_pair, jobs))
        else:
            gains = [_one_pair(j) for j in jobs]
        stats = summarise(gains)
        print("== %-8s pickets on vs off: %+7.1f +/- %5.1f VP over %d paired"
              " games -> %s ==   %.0fs"
              % (side, stats["advantage"], stats["stderr"], games,
                 stats["verdict"], time.time() - start),
              flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120,
         int(sys.argv[2]) if len(sys.argv) > 2 else 40)

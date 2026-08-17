"""Measure an Imperial doctrine over whole campaigns of *Invasion: Earth*.

The victory level is the thing the rules settle on and it is far too coarse to
develop a doctrine against: for most of this project every game ended in the
same one.  What separates two Imperial doctrines is how much of the army got
ashore, how much of the fleet was left when it did, how many cities were
garrisoned and how many of the Solomani were still standing -- so this reports
all of them, over a block of seeds, with a standard error on each.

    python tools/ie_campaign.py                    # the doctrine, 8 games
    python tools/ie_campaign.py --games 20 --turns 36
    python tools/ie_campaign.py --imperial scripted

Both sides are played by the same doctrine class unless ``--solomani`` says
otherwise, which is what makes an Imperial change measurable: the defence is
held fixed while the attack varies.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ie.agents import (BeachheadAgent, HeuristicAgent,          # noqa: E402
                       RandomAgent, ScriptedAgent)
from ie.engine import Engine, new_game                             # noqa: E402
from ie.state import IMPERIAL, SOLOMANI, VICTORY_ORDER             # noqa: E402

AGENTS = {"heuristic": HeuristicAgent, "scripted": ScriptedAgent,
          "beachhead": BeachheadAgent, "random": RandomAgent}


def _mean_stderr(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n)


def campaign(seed: int, imperial_cls, solomani_cls, turns: int) -> dict:
    """Play one game out and report what became of both sides."""
    state = new_game(seed=seed)
    engine = Engine(state, imperial_cls(IMPERIAL, seed=seed * 2 + 1),
                    solomani_cls(SOLOMANI, seed=seed * 2 + 2))
    peak_ashore = 0.0
    while not state.game_over and state.turn <= turns:
        engine.play_turn()
        peak_ashore = max(peak_ashore, sum(
            u.current for u in state.surface.values()
            if u.side == IMPERIAL and u.carrier is None
            and isinstance(u.location, int) and not u.dead))
    if not state.game_over:
        state.game_over = True
        state.result = state.victory_level()
    return {
        "result": state.result,
        "taken": len(state.geometry.urban) - len(state.solomani_urban()),
        "points": state.victory_points(),
        "peak_ashore": peak_ashore,
        "ashore": sum(u.current for u in state.surface.values()
                      if u.side == IMPERIAL and u.carrier is None
                      and isinstance(u.location, int) and not u.dead),
        "stranded": sum(u.current for u in state.surface.values()
                        if u.side == IMPERIAL and not u.dead
                        and not isinstance(u.location, int)),
        "fleet": len([u for u in state.naval.values() if u.side == IMPERIAL]),
        "lift": sum(u.cls.capacity for u in state.naval.values()
                    if u.side == IMPERIAL),
        "solomani": sum(u.current for u in state.surface.values()
                        if u.side == SOLOMANI and not u.dead
                        and not u.cls.guerrilla),
        "batteries": sum(1 for u in state.surface.values()
                         if u.side == SOLOMANI and not u.dead
                         and u.cls.planetary_defense),
        "waves": state.waves_taken,
        "turns": state.turn - 1,
    }


def run(games: int, turns: int, imperial: str, solomani: str, first: int = 1):
    imperial_cls, solomani_cls = AGENTS[imperial], AGENTS[solomani]
    rows = [campaign(seed, imperial_cls, solomani_cls, turns)
            for seed in range(first, first + games)]
    summary = {}
    for key in ("taken", "points", "peak_ashore", "ashore", "stranded",
                "fleet", "lift", "solomani", "batteries", "waves", "turns"):
        summary[key] = _mean_stderr([r[key] for r in rows])
    levels = {}
    for row in rows:
        levels[row["result"]] = levels.get(row["result"], 0) + 1
    return rows, summary, levels


def report(games, turns, imperial, solomani, first=1):
    rows, summary, levels = run(games, turns, imperial, solomani, first)
    print("%s vs %s, %d games of %d turns"
          % (imperial, solomani, games, turns))
    labels = [
        ("taken", "cities taken"), ("points", "victory points"),
        ("peak_ashore", "peak factors ashore"), ("ashore", "factors ashore at the end"),
        ("stranded", "factors never landed"), ("fleet", "squadrons left"),
        ("lift", "lift capacity left"), ("solomani", "Solomani factors left"),
        ("batteries", "batteries left"), ("waves", "replacement waves"),
        ("turns", "turns played"),
    ]
    for key, label in labels:
        mean, stderr = summary[key]
        print("  %-26s %8.1f ± %.1f" % (label, mean, stderr))
    for level in VICTORY_ORDER:
        if levels.get(level):
            print("  %-26s %d" % (level, levels[level]))
    return rows, summary, levels


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=8)
    parser.add_argument("--turns", type=int, default=24)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--imperial", default="heuristic", choices=AGENTS)
    parser.add_argument("--solomani", default="heuristic", choices=AGENTS)
    args = parser.parse_args(argv)
    report(args.games, args.turns, args.imperial, args.solomani, args.first)


if __name__ == "__main__":
    main()

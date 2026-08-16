"""One feature vector, two games.

A network fitted to ``ffw.features`` reads sixty-odd slots about jump numbers,
refuelling status and squadron counts.  None of that exists in *Invasion:
Earth*, so such a network cannot be pointed at it -- not because it is a bad
network but because the vocabulary does not carry over.

This encoding is the vocabulary that does.  Every slot is a question both games
answer, phrased so the answers are comparable:

* **Territory.**  What share of the objectives do I hold?  Which way is it
  moving?  In *Fifth Frontier War* the objectives are worlds with victory point
  values; in *Invasion: Earth* they are the sixty-one urban hexes.  The share
  is a fraction in both, so the slot means the same thing.
* **Force.**  How much do I have left as a fraction of what I started with, how
  much of it is committed forward, and how concentrated is it?  A fleet in one
  game and an army in the other, but "have I got more than they have, and is it
  in one place" is the same question.
* **Tempo.**  How much of the clock is gone?  Both games are decided on a
  schedule -- rule 8's armistice boundary at turn 52, the quarterly victory
  check -- so elapsed fraction is the honest normalisation.
* **Reach.**  How far is my force from its objectives, and from theirs?
  Measured in hexes and divided by the map's own diameter, so a 32-parsec
  subsector and a 21-hex-radius planet come out on the same scale.

Everything is scaled to roughly [-1, 1] and written from the perspective of the
side being asked about, so a single network serves both seats -- the same trick
``ffw.features.perspective`` uses, generalised.

**What this is not.**  It is not a replacement for either game's own features.
``ffw.features`` knows about refuelling and jump plotting and is strictly
better *at Fifth Frontier War*; the point of this one is that it is the same in
both, which is a different and much weaker property.  A doctrine that only
plays well through these features is playing a coarser game than either
rulebook describes.  That is the price of transfer and it should be measured,
not assumed away.
"""

from __future__ import annotations

import math

FEATURE_NAMES = (
    # -- territory -----------------------------------------------------
    "objective_share",          # fraction of scoring locations held
    "objective_share_delta",    # change since the game began
    "objective_contested",      # fraction neither side securely holds
    # -- force ---------------------------------------------------------
    "force_ratio",              # own committed strength over both sides'
    "force_remaining",          # own strength as a fraction of the start
    "enemy_remaining",          # theirs, likewise
    "force_forward",            # fraction of own force committed to the front
    "force_concentration",      # largest single stack over total
    "force_reserve",            # fraction still uncommitted or off-map
    # -- tempo ---------------------------------------------------------
    "elapsed",                  # fraction of the scheduled length played
    "tempo_pressure",           # elapsed, weighted by whether we are ahead
    # -- reach ---------------------------------------------------------
    "distance_to_objectives",   # mean distance from own force to targets
    "distance_to_enemy",        # mean distance to the enemy's centre of mass
    "supply_health",            # fraction of own force in supply
    # -- the scoreboard ------------------------------------------------
    "score_margin",             # normalised victory margin, own-positive
    "score_level",              # normalised position on the victory table
)

DIMENSIONS = len(FEATURE_NAMES)


def _clip(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0):
    return numerator / denominator if denominator else default


def encode(adapter, state, side: str):
    """The shared feature vector for ``state`` from ``side``'s point of view.

    ``adapter`` is a :class:`strategy.adapter.GameAdapter`, which is what knows
    how to answer each question for its own game.  This function only decides
    what the questions are and how to scale the answers.
    """
    import numpy as np

    objectives = adapter.objectives(state)
    held = adapter.objectives_held(state, side)
    enemy_held = adapter.objectives_held(state, adapter.enemy_of(side))
    total = max(1, len(objectives))

    share = len(held) / total
    start_share = adapter.initial_objective_share(side)
    contested = 1.0 - (len(held) + len(enemy_held)) / total

    own = adapter.force_strength(state, side)
    foe = adapter.force_strength(state, adapter.enemy_of(side))
    own_start = max(1.0, adapter.initial_force(side))
    foe_start = max(1.0, adapter.initial_force(adapter.enemy_of(side)))

    forward, reserve, largest = adapter.force_posture(state, side)
    committed = max(1e-6, forward + reserve)

    elapsed = _clip(_safe_ratio(state.turn, adapter.scheduled_length), 0.0, 1.0)
    margin = adapter.normalised_margin(state, side)

    to_objectives, to_enemy = adapter.reach(state, side)
    diameter = max(1.0, adapter.map_diameter)

    values = [
        _clip(2.0 * share - 1.0),
        _clip(4.0 * (share - start_share)),
        _clip(2.0 * contested - 1.0),

        _clip(2.0 * _safe_ratio(own, own + foe, 0.5) - 1.0),
        _clip(2.0 * _safe_ratio(own, own_start) - 1.0),
        _clip(2.0 * _safe_ratio(foe, foe_start) - 1.0),
        _clip(2.0 * _safe_ratio(forward, committed) - 1.0),
        _clip(2.0 * _safe_ratio(largest, max(1e-6, own)) - 1.0),
        _clip(2.0 * _safe_ratio(reserve, committed) - 1.0),

        _clip(2.0 * elapsed - 1.0),
        _clip((2.0 * elapsed - 1.0) * (1.0 if margin >= 0 else -1.0)),

        _clip(1.0 - 2.0 * _safe_ratio(to_objectives, diameter)),
        _clip(1.0 - 2.0 * _safe_ratio(to_enemy, diameter)),
        _clip(2.0 * adapter.supply_health(state, side) - 1.0),

        _clip(margin),
        _clip(adapter.normalised_level(state, side)),
    ]
    return np.asarray(values, dtype=np.float64)


def describe(vector) -> str:
    """The vector as readable lines, for looking at what a network is fed."""
    return "\n".join("  %-24s %+6.3f" % (name, value)
                     for name, value in zip(FEATURE_NAMES, vector))

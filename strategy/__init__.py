"""What *Fifth Frontier War* and *Invasion: Earth* have in common.

The goal is agents that are good at strategy rather than good at one game, and
the way to find out whether you have one is to train it on a game it has never
seen.  That needs three things to be shared, and only three:

* a **description of a game** the training code can drive without knowing which
  game it is -- ``GameAdapter``;
* a **feature encoding** in which a position from either game becomes a vector
  of the same length, with each slot meaning the same thing in both --
  ``encode``;
* an **evaluation protocol** that scores play the same way in both -- paired
  games, victory margin and victory level, which already live in
  ``ffw.training`` and are reused rather than reimplemented.

Everything else stays where it is.  The two rulebooks are genuinely different
-- plotted jump movement against grav-mobile ground war, a war of manoeuvre
between two navies against an amphibious assault on a single planet -- and
pretending one engine could serve both would produce a worse model of each.
What transfers is not code but *skill*, and skill transfers through the
features.

The encoding is the interesting part, so it is worth saying what it is for.  A
value network fitted to ``ffw.features`` sees 60-odd slots about jump ranges,
refuelling and squadron counts; none of that exists in ``ie``.  The shared
encoding instead asks the questions both games answer:

    how much of the map do I hold, and is that share rising or falling?
    how much force do I have, where is it, and how concentrated is it?
    how much of the clock is gone?
    how far is my army from its supply, and from the enemy's centre of mass?

Those are the questions a general asks in either century.  A network fitted to
them on one game has learned something that means something in the other.
"""

from .adapter import GameAdapter, FFW, IE, adapters, adapter_for
from .features import DIMENSIONS, FEATURE_NAMES, encode, describe
from .search import (OBJECTIVES, TrainingLog, doctrine_for, objective,
                     paired_advantage, score_many, score_weights, train,
                     tunables)

__all__ = ["GameAdapter", "FFW", "IE", "adapters", "adapter_for",
           "DIMENSIONS", "FEATURE_NAMES", "encode", "describe",
           "OBJECTIVES", "TrainingLog", "doctrine_for", "objective",
           "paired_advantage", "score_many", "score_weights", "train",
           "tunables"]

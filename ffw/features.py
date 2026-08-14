"""Fixed-length feature vectors describing a position.

One value network serves both players, so a vector is written from the Zhodani
point of view and ``perspective`` rewrites it for the Imperium.  Three kinds of
feature behave differently under that flip:

``paired``
    ``something_imperial`` / ``something_zhodani`` describe the same quantity
    for the two sides and are swapped.
``signed``
    inherently directional -- positive means good for the Zhodani -- and are
    negated.
``neutral``
    mean the same thing to either viewer and are left alone.

The split is derived from the feature *names* rather than hard-coded indices,
so adding a feature cannot silently corrupt the flip.  ``FEATURE_VERSION`` is
bumped whenever the layout changes; ``ValueNetwork`` records it so a stale
network is rejected rather than quietly misread.
"""

from __future__ import annotations

import numpy as np

from . import hexmap
from .state import GameState, IMPERIAL, ZHODANI

#: bump when the feature layout changes; saved networks record it
FEATURE_VERSION = 2

FEATURE_NAMES = [
    # -- clock and score ---------------------------------------------------
    "turn",
    "vp_margin",
    # -- territory ---------------------------------------------------------
    "worlds_imperial", "worlds_zhodani", "worlds_neutral",
    "capitals_imperial", "capitals_zhodani",
    "penetration",
    "neutrals_unclaimed",
    # -- fleets ------------------------------------------------------------
    "sq_attack_imperial", "sq_attack_zhodani",
    "sq_count_imperial", "sq_count_zhodani",
    "sq_reduced_imperial", "sq_reduced_zhodani",
    "sq_stranded_imperial", "sq_stranded_zhodani",
    "fleets_imperial", "fleets_zhodani",
    "concentration_imperial", "concentration_zhodani",
    "depth_imperial", "depth_zhodani",
    # -- armies and lift ---------------------------------------------------
    "troops_imperial", "troops_zhodani",
    "afloat_imperial", "afloat_zhodani",
    "lift_free_imperial", "lift_free_zhodani",
    # -- command -----------------------------------------------------------
    "admiral_quality_imperial", "admiral_quality_zhodani",
    "admirals_idle_imperial", "admirals_idle_zhodani",
    # -- static defences ---------------------------------------------------
    "sdb_remaining_imperial", "sdb_remaining_zhodani",
    "defbn_intact_imperial", "defbn_intact_zhodani",
    # -- production --------------------------------------------------------
    "rp_sq_imperial", "rp_sq_zhodani",
    "rp_tr_imperial", "rp_tr_zhodani",
    # -- contact -----------------------------------------------------------
    "front_pressure",
    "guerrilla_strength",
]

N_FEATURES = len(FEATURE_NAMES)
INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}

#: features whose sign already means "good for the Zhodani"
SIGNED = ("vp_margin", "penetration", "guerrilla_strength")

#: (imperial index, zhodani index) pairs, derived from the names
SWAP_PAIRS = tuple(
    (INDEX[name], INDEX[name[: -len("_imperial")] + "_zhodani"])
    for name in FEATURE_NAMES
    if name.endswith("_imperial")
    and name[: -len("_imperial")] + "_zhodani" in INDEX
)


def extract(state: GameState) -> np.ndarray:
    """A normalised feature vector for the current position."""
    worlds = list(state.world_map)
    controlled = {IMPERIAL: [], ZHODANI: []}
    neutral = 0
    unclaimed_free = 0
    defbn_full = {IMPERIAL: 0.0, ZHODANI: 0.0}
    defbn_now = {IMPERIAL: 0.0, ZHODANI: 0.0}
    sdb_now = {IMPERIAL: 0.0, ZHODANI: 0.0}
    base_hexes = {IMPERIAL: [], ZHODANI: []}
    for w in worlds:
        if w.control is None:
            neutral += 1
            if w.owner == "neutral" and w.defense_battalions == 0 and w.sdb == 0:
                unclaimed_free += 1
        else:
            controlled[w.control].append(w)
            if w.base:
                base_hexes[w.control].append(w.hex)
        origin = w.original_control
        if origin is not None:
            defbn_full[origin] += w.defense_battalions
            if w.control == origin:
                defbn_now[origin] += w.defense_current
                sdb_now[origin] += w.sdb_current

    squadrons = {IMPERIAL: [], ZHODANI: []}
    for s in state.squadrons.values():
        squadrons[s.side].append(s)

    troops = {IMPERIAL: 0.0, ZHODANI: 0.0}
    afloat = {IMPERIAL: 0.0, ZHODANI: 0.0}
    guerrilla = 0.0
    for t in state.troops.values():
        if t.losses >= 100:
            continue
        troops[t.side] += t.current
        if t.aboard is not None:
            afloat[t.side] += t.current
        if t.guerrilla:
            guerrilla += t.current

    lift_free = {IMPERIAL: 0.0, ZHODANI: 0.0}
    stranded = {IMPERIAL: 0, ZHODANI: 0}
    stacks = {IMPERIAL: {}, ZHODANI: {}}
    for side in (IMPERIAL, ZHODANI):
        for s in squadrons[side]:
            carried = sum(state.troops[u].current for u in s.troops
                          if u in state.troops)
            lift_free[side] += max(0, s.capacity - carried)
            if not s.refuelled:
                stranded[side] += 1
            if s.location in state.world_map.worlds:
                stacks[side][s.location] = stacks[side].get(s.location, 0) + s.attack

    # how far the fleet is operating from its own bases: overextension
    depth = {IMPERIAL: 0.0, ZHODANI: 0.0}
    for side in (IMPERIAL, ZHODANI):
        bases = base_hexes[side]
        located = [s for s in squadrons[side]
                   if s.location in state.world_map.worlds]
        if bases and located:
            total = 0
            for s in located:
                total += min(hexmap.distance(s.location, b) for b in bases)
            depth[side] = total / len(located)

    admiral_quality = {IMPERIAL: 0.0, ZHODANI: 0.0}
    admirals_idle = {IMPERIAL: 0, ZHODANI: 0}
    commanding = {IMPERIAL: [], ZHODANI: []}
    for a in state.admirals.values():
        if not a.location:
            continue
        if a.fleet is None:
            admirals_idle[a.side] += 1
        else:
            commanding[a.side].append(a)
    for side in (IMPERIAL, ZHODANI):
        if commanding[side]:
            # a low planning factor is good, so invert it into a 0..1 quality
            admiral_quality[side] = sum(max(0.0, 5 - a.planning) / 5.0
                                        for a in commanding[side]) \
                / len(commanding[side])

    penetration = sum(1 for w in worlds
                      if w.owner == "imperial" and w.control == ZHODANI)
    counter = sum(1 for w in worlds
                  if w.owner in ("zhodani", "sword_worlds", "vargr")
                  and w.control == IMPERIAL)
    contact = sum(1 for w in worlds
                  if state.squadrons_at(w.hex, IMPERIAL)
                  and state.squadrons_at(w.hex, ZHODANI))

    def frac(now, full):
        return now / full if full > 0 else 0.0

    values = {
        "turn": state.turn / 60.0,
        "vp_margin": state.victory_margin() / 300.0,
        "worlds_imperial": len(controlled[IMPERIAL]) / 146.0,
        "worlds_zhodani": len(controlled[ZHODANI]) / 146.0,
        "worlds_neutral": neutral / 146.0,
        "capitals_imperial": sum(1 for w in controlled[IMPERIAL]
                                 if w.subsector_capital) / 8.0,
        "capitals_zhodani": sum(1 for w in controlled[ZHODANI]
                                if w.subsector_capital) / 8.0,
        "penetration": (penetration - counter) / 40.0,
        "neutrals_unclaimed": unclaimed_free / 20.0,
        "sq_attack_imperial": sum(s.attack for s in squadrons[IMPERIAL]) / 200.0,
        "sq_attack_zhodani": sum(s.attack for s in squadrons[ZHODANI]) / 200.0,
        "sq_count_imperial": len(squadrons[IMPERIAL]) / 60.0,
        "sq_count_zhodani": len(squadrons[ZHODANI]) / 60.0,
        "sq_reduced_imperial": sum(1 for s in squadrons[IMPERIAL]
                                   if s.reduced) / 30.0,
        "sq_reduced_zhodani": sum(1 for s in squadrons[ZHODANI]
                                  if s.reduced) / 30.0,
        "sq_stranded_imperial": stranded[IMPERIAL] / 30.0,
        "sq_stranded_zhodani": stranded[ZHODANI] / 30.0,
        "fleets_imperial": sum(1 for f in state.fleets.values()
                               if f.active and f.side == IMPERIAL) / 14.0,
        "fleets_zhodani": sum(1 for f in state.fleets.values()
                              if f.active and f.side == ZHODANI) / 14.0,
        "concentration_imperial": (max(stacks[IMPERIAL].values(), default=0)
                                   / 60.0),
        "concentration_zhodani": (max(stacks[ZHODANI].values(), default=0)
                                  / 60.0),
        "depth_imperial": depth[IMPERIAL] / 12.0,
        "depth_zhodani": depth[ZHODANI] / 12.0,
        "troops_imperial": troops[IMPERIAL] / 5000.0,
        "troops_zhodani": troops[ZHODANI] / 5000.0,
        "afloat_imperial": afloat[IMPERIAL] / 1000.0,
        "afloat_zhodani": afloat[ZHODANI] / 1000.0,
        "lift_free_imperial": lift_free[IMPERIAL] / 3000.0,
        "lift_free_zhodani": lift_free[ZHODANI] / 3000.0,
        "admiral_quality_imperial": admiral_quality[IMPERIAL],
        "admiral_quality_zhodani": admiral_quality[ZHODANI],
        "admirals_idle_imperial": admirals_idle[IMPERIAL] / 14.0,
        "admirals_idle_zhodani": admirals_idle[ZHODANI] / 14.0,
        "sdb_remaining_imperial": sdb_now[IMPERIAL] / 3000.0,
        "sdb_remaining_zhodani": sdb_now[ZHODANI] / 3000.0,
        "defbn_intact_imperial": frac(defbn_now[IMPERIAL], defbn_full[IMPERIAL]),
        "defbn_intact_zhodani": frac(defbn_now[ZHODANI], defbn_full[ZHODANI]),
        "rp_sq_imperial": state.replacement_points[IMPERIAL]["squadron"] / 30.0,
        "rp_sq_zhodani": state.replacement_points[ZHODANI]["squadron"] / 30.0,
        "rp_tr_imperial": state.replacement_points[IMPERIAL]["troop"] / 30.0,
        "rp_tr_zhodani": state.replacement_points[ZHODANI]["troop"] / 30.0,
        "front_pressure": contact / 10.0,
        "guerrilla_strength": guerrilla / 1000.0,
    }
    return np.asarray([values[name] for name in FEATURE_NAMES],
                      dtype=np.float64)


def perspective(vector: np.ndarray, side: str) -> np.ndarray:
    """Rewrite a vector so that positive always means 'good for ``side``'."""
    if side == ZHODANI:
        return vector
    flipped = np.asarray(vector, dtype=np.float64).copy()
    for a, b in SWAP_PAIRS:
        flipped[a], flipped[b] = flipped[b], flipped[a]
    for name in SIGNED:
        flipped[INDEX[name]] = -flipped[INDEX[name]]
    return flipped


def describe(vector: np.ndarray) -> dict:
    """``{name: value}`` for inspection and debugging."""
    return {name: float(value) for name, value in zip(FEATURE_NAMES, vector)}

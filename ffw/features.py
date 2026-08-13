"""Fixed-length feature vectors describing a position.

Used by the learned agents and by the training harness for logging.  Features
are written from the Zhodani point of view and negated for the Imperium so a
single value network serves both sides.
"""

from __future__ import annotations

import numpy as np

from .state import GameState, IMPERIAL, ZHODANI

FEATURE_NAMES = [
    "turn",
    "vp_margin",
    "worlds_imperial", "worlds_zhodani", "worlds_neutral",
    "capitals_imperial", "capitals_zhodani",
    "sq_attack_imperial", "sq_attack_zhodani",
    "sq_count_imperial", "sq_count_zhodani",
    "sq_reduced_imperial", "sq_reduced_zhodani",
    "troops_imperial", "troops_zhodani",
    "fleets_imperial", "fleets_zhodani",
    "rp_sq_imperial", "rp_sq_zhodani",
    "rp_tr_imperial", "rp_tr_zhodani",
    "penetration", "front_pressure",
    "guerrilla_strength", "sdb_remaining_imperial",
    "invasion_troops_afloat_zhodani", "invasion_troops_afloat_imperial",
]

N_FEATURES = len(FEATURE_NAMES)


def extract(state: GameState) -> np.ndarray:
    """A normalised feature vector for the current position."""
    worlds = list(state.world_map)
    imp_worlds = [w for w in worlds if w.control == IMPERIAL]
    zho_worlds = [w for w in worlds if w.control == ZHODANI]
    neutral = [w for w in worlds if w.control is None]

    sq = {IMPERIAL: [], ZHODANI: []}
    for s in state.squadrons.values():
        sq[s.side].append(s)
    tr = {IMPERIAL: 0.0, ZHODANI: 0.0}
    afloat = {IMPERIAL: 0.0, ZHODANI: 0.0}
    guerrilla = 0.0
    for t in state.troops.values():
        if t.losses >= 100:
            continue
        tr[t.side] += t.current
        if t.aboard is not None:
            afloat[t.side] += t.current
        if t.guerrilla:
            guerrilla += t.current

    # how deep the Zhodani have driven into the Imperium
    penetration = sum(1 for w in worlds
                      if w.owner == "imperial" and w.control == ZHODANI)
    counter = sum(1 for w in worlds
                  if w.owner in ("zhodani", "sword_worlds", "vargr")
                  and w.control == IMPERIAL)
    contact = 0
    for w in worlds:
        if not state.squadrons_at(w.hex, IMPERIAL):
            continue
        if state.squadrons_at(w.hex, ZHODANI):
            contact += 1

    sdb_left = sum(w.sdb_current for w in worlds
                   if w.owner == "imperial" and w.control == IMPERIAL)

    values = [
        state.turn / 60.0,
        state.victory_margin() / 300.0,
        len(imp_worlds) / 146.0, len(zho_worlds) / 146.0, len(neutral) / 146.0,
        sum(1 for w in imp_worlds if w.subsector_capital) / 8.0,
        sum(1 for w in zho_worlds if w.subsector_capital) / 8.0,
        sum(s.attack for s in sq[IMPERIAL]) / 200.0,
        sum(s.attack for s in sq[ZHODANI]) / 200.0,
        len(sq[IMPERIAL]) / 60.0, len(sq[ZHODANI]) / 60.0,
        sum(1 for s in sq[IMPERIAL] if s.reduced) / 30.0,
        sum(1 for s in sq[ZHODANI] if s.reduced) / 30.0,
        tr[IMPERIAL] / 5000.0, tr[ZHODANI] / 5000.0,
        sum(1 for f in state.fleets.values()
            if f.active and f.side == IMPERIAL) / 14.0,
        sum(1 for f in state.fleets.values()
            if f.active and f.side == ZHODANI) / 14.0,
        state.replacement_points[IMPERIAL]["squadron"] / 30.0,
        state.replacement_points[ZHODANI]["squadron"] / 30.0,
        state.replacement_points[IMPERIAL]["troop"] / 30.0,
        state.replacement_points[ZHODANI]["troop"] / 30.0,
        (penetration - counter) / 40.0,
        contact / 10.0,
        guerrilla / 1000.0,
        sdb_left / 3000.0,
        afloat[ZHODANI] / 1000.0, afloat[IMPERIAL] / 1000.0,
    ]
    return np.asarray(values, dtype=np.float64)


def perspective(vector: np.ndarray, side: str) -> np.ndarray:
    """Flip the vector so that positive always means 'good for ``side``'."""
    if side == ZHODANI:
        return vector
    flipped = vector.copy()
    # swap the paired imperial/zhodani columns and negate the signed ones
    pairs = [(2, 3), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
             (15, 16), (17, 18), (19, 20), (25, 26)]
    for a, b in pairs:
        flipped[a], flipped[b] = flipped[b], flipped[a]
    for idx in (1, 21):
        flipped[idx] = -flipped[idx]
    return flipped

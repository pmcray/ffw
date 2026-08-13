"""A human-controlled player.

The agent keeps the heuristic doctrine as its "staff work" default and asks the
human only for the decisions that matter: which system each fleet is ordered to
next, and whether to break off a losing battle.  Orders are supplied through a
callback so the same class serves the notebook widgets, a text prompt, or a
scripted test.
"""

from __future__ import annotations

from .. import hexmap
from ..engine import fleet_jump
from ..state import GameState
from .heuristic import HeuristicAgent


class HumanAgent(HeuristicAgent):
    """Ask ``ask`` for orders; fall back to the doctrine when it returns None."""

    name = "human"

    def __init__(self, side: str, ask=None, seed: int | None = None,
                 label: str | None = None):
        super().__init__(side, seed=seed, label=label)
        self.ask = ask or (lambda request: None)
        self.pending: list[dict] = []

    # -- orders ------------------------------------------------------------
    def fleet_options(self, state: GameState, fleet, origin: str) -> list[str]:
        jump = fleet_jump(state, fleet)
        if jump <= 0 or origin not in state.world_map.worlds:
            return []
        options = [h for h in state.world_map.worlds
                   if 0 < hexmap.distance(origin, h) <= jump]
        options.sort(key=lambda h: (hexmap.distance(origin, h),
                                    state.world_map.get(h).name))
        return options

    def plot_fleet(self, state, side, fleet, turn, engine):
        origin = fleet.plot.get(turn - 1, ("hold", fleet.location))[1]
        options = self.fleet_options(state, fleet, origin)
        request = {
            "kind": "plot_fleet",
            "turn": turn,
            "fleet": fleet.name,
            "fleet_uid": fleet.uid,
            "origin": origin,
            "origin_name": (state.world_map.get(origin).name
                            if origin in state.world_map.worlds else origin),
            "squadrons": len(fleet.squadrons),
            "attack": sum(state.squadrons[u].attack for u in fleet.squadrons
                          if u in state.squadrons),
            "options": options,
            "option_names": [state.world_map.get(h).name for h in options],
            "recommended": super()._best_destination(
                state, side, fleet, origin, fleet_jump(state, fleet)),
        }
        answer = self.ask(request)
        if answer is None:
            return super().plot_fleet(state, side, fleet, turn, engine)
        if answer in ("hold", origin, None):
            return ("hold", origin)
        if answer in options:
            return ("jump", answer)
        return ("hold", origin)

    def wants_to_disengage(self, state, side, hex_id, engine):
        own = state.squadrons_at(hex_id, side)
        foe = state.squadrons_at(hex_id, state.enemy_of(side))
        if not own or not foe:
            return False
        answer = self.ask({
            "kind": "disengage",
            "hex": hex_id,
            "world": state.world_map.get(hex_id).name
                     if hex_id in state.world_map.worlds else hex_id,
            "own_attack": sum(s.attack for s in own),
            "own_squadrons": len(own),
            "enemy_attack": sum(s.attack for s in foe),
            "enemy_squadrons": len(foe),
            "recommended": super().wants_to_disengage(state, side, hex_id, engine),
        })
        if answer is None:
            return super().wants_to_disengage(state, side, hex_id, engine)
        return bool(answer)

    def declare_armistice(self, state, engine):
        answer = self.ask({"kind": "armistice", "turn": state.turn,
                           "margin": state.victory_margin()})
        if answer is None:
            return False
        return bool(answer)

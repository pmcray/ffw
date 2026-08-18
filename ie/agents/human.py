"""A human-controlled player for *Invasion: Earth*.

The agent keeps the doctrine as its staff work and asks the human only for the
decisions that decide the campaign.  On the Imperial side that is: where to put
the lodgement, what the fleet bombards, where each stack marches, and whether to
buy a replacement wave.  On the Solomani side it is when to bring the boats up
and where the field army goes.  Everything else -- loading transports, allocating
fire within a hex, hiding guerrillas -- is left to the doctrine, because a game
that asked a human to reload sixty transports would not be played twice.

Orders come through an ``ask`` callback so the same class serves the notebook
widgets, a text prompt, or a scripted test, exactly as ``ffw.agents.human``
does.  A callback that returns ``None`` defers to the doctrine, which means a
human can take one decision a turn and leave the rest to the staff.
"""

from __future__ import annotations

from .. import oob
from ..state import IMPERIAL, SOLOMANI
from .heuristic import HeuristicAgent

#: How many candidates to put in front of the human.  A shortlist, because the
#: alternatives are 184 land hexes and a choice nobody can read is not a choice.
SHORTLIST = 10


class HumanAgent(HeuristicAgent):
    """Ask ``ask`` for orders; fall back to the doctrine when it returns None."""

    name = "human"

    def __init__(self, side: str, ask=None, seed: int | None = None,
                 label: str | None = None):
        super().__init__(side, seed=seed, label=label)
        self.ask = ask or (lambda request: None)

    # -- where to land -------------------------------------------------
    def _pick_anchor(self, state):
        """The one decision that shapes everything after it."""
        recommended = super()._pick_anchor(state)
        if self.side != IMPERIAL:
            return recommended
        geo = state.geometry
        field = (self._threat_map(state), self._gun_map(state),
                 self._close_map(state))
        scored = sorted(
            ((self._landing_score(state, c, field), c) for c in geo.land),
            key=lambda pair: (-pair[0], pair[1]))
        options = [cell for score, cell in scored[:SHORTLIST] if score > -1e8]
        answer = self.ask({
            "kind": "lodgement",
            "turn": state.turn,
            "options": options,
            "detail": [self.describe_cell(state, cell) for cell in options],
            "recommended": recommended,
        })
        if answer is None or answer not in geo.land:
            return recommended
        return answer

    def describe_cell(self, state, cell) -> dict:
        """What a player would want to know about a hex before choosing it."""
        geo = state.geometry
        lat, lon = geo.lat_lon[cell]
        guns = self._gun_map(state).get(cell, 0.0)
        return {
            "cell": cell,
            "terrain": geo.terrain[cell],
            "lat": lat,
            "lon": lon,
            "cities_within_3": sum(1 for c in geo.within(cell, 3)
                                   if c in geo.urban and c not in state.garrisoned),
            "cities_within_7": sum(1 for c in geo.within(cell, 7)
                                   if c in geo.urban and c not in state.garrisoned),
            "gun_factors": guns,
            "solomani_within_3": self._close_map(state).get(cell, 0.0),
            "imperial_here": sum(u.current for u in state.surface_at(cell, IMPERIAL)),
            "solomani_here": sum(u.current for u in state.surface_at(cell, SOLOMANI)),
        }

    # -- what to shoot at ----------------------------------------------
    def _bombardment_targets(self, state):
        recommended = super()._bombardment_targets(state)
        if self.side != IMPERIAL:
            return recommended
        geo = state.geometry
        guns = self._gun_map(state)
        scored: dict = {}
        for unit in state.surface.values():
            if unit.side != SOLOMANI or unit.dead or unit.cls.guerrilla:
                continue
            if not isinstance(unit.location, int):
                continue
            scored[unit.location] = scored.get(unit.location, 0.0) \
                + self._target_value(state, unit)
        options = sorted(scored, key=lambda c: (-scored[c], c))[:SHORTLIST]
        strength = sum(u.bombard_factor() for u in state.naval.values()
                       if u.side == IMPERIAL and u.cls.bombard > 0
                       and not u.cargo)
        answer = self.ask({
            "kind": "bombardment",
            "turn": state.turn,
            "factors_available": strength,
            "options": options,
            "detail": [{
                "cell": cell,
                "terrain": geo.terrain[cell],
                "solomani": sum(u.current
                                for u in state.surface_at(cell, SOLOMANI)),
                "batteries": [u.cls.name for u in state.surface_at(cell, SOLOMANI)
                              if u.cls.planetary_defense],
                "return_fire": guns.get(cell, 0.0),
            } for cell in options],
            "recommended": recommended,
        })
        if answer is None:
            return recommended
        if isinstance(answer, int):
            answer = [answer]
        chosen = [c for c in answer if c in scored]
        return chosen or recommended

    # -- where the army goes -------------------------------------------
    def surface_moves(self, state, side, engine):
        plan = super().surface_moves(state, side, engine)
        if side != self.side:
            return plan
        stacks: dict = {}
        for unit in state.surface.values():
            if unit.side != side or unit.dead or unit.carrier is not None:
                continue
            if not unit.cls.mobile or not isinstance(unit.location, int):
                continue
            if unit.landed_turn == state.turn:
                continue
            stacks.setdefault(unit.location, []).append(unit)
        if not stacks:
            return plan

        enemy_zoc = state.zone_of_control(
            SOLOMANI if side == IMPERIAL else IMPERIAL)
        occupied = {u.location for u in state.surface.values()
                    if u.side != side and u.carrier is None and not u.dead
                    and isinstance(u.location, int)}
        geo = state.geometry
        for origin, units in sorted(stacks.items()):
            costs = engine.movement_costs(units[0], enemy_zoc, occupied)
            reachable = [c for c in costs
                         if c != origin and c in geo.land]
            reachable.sort(key=lambda c: (
                -self._garrison_value(state, c, enemy_zoc, occupied), costs[c], c))
            options = reachable[:SHORTLIST]
            recommended = plan.get(units[0].uid, origin)
            if recommended in costs and recommended not in options \
                    and recommended != origin:
                # the staff's own pick always appears, even when the shortlist
                # ranks it below ten others
                options = [recommended] + options[:SHORTLIST - 1]
            answer = self.ask({
                "kind": "move",
                "turn": state.turn,
                "origin": origin,
                "strength": sum(u.current for u in units),
                "units": [u.cls.name for u in units],
                "in_supply": engine._in_supply(units[0]),
                "options": options,
                "detail": [{
                    "cell": cell,
                    "terrain": geo.terrain[cell],
                    "cost": costs.get(cell, 0),
                    "garrisons": self._garrison_value(state, cell, enemy_zoc,
                                                      occupied),
                    "solomani": sum(u.current
                                    for u in state.surface_at(cell, SOLOMANI
                                                              if side == IMPERIAL
                                                              else IMPERIAL)),
                } for cell in options],
                "recommended": recommended,
            })
            if answer is None:
                continue                     # the staff plan stands
            if answer in ("hold", origin):
                for unit in units:
                    plan.pop(unit.uid, None)
                continue
            if answer not in costs or answer not in geo.land:
                continue
            for unit in units:
                plan[unit.uid] = answer
        return plan

    # -- the quarterly special turn -------------------------------------
    def replacement_waves(self, state, engine):
        recommended = super().replacement_waves(state, engine)
        if self.side != IMPERIAL:
            return recommended
        answer = self.ask({
            "kind": "replacements",
            "turn": state.turn,
            "waves_taken": state.waves_taken,
            "victory_points": state.victory_points(),
            "dead_units": len(state.pools.get("dead_surface", [])),
            "recommended": recommended,
        })
        return recommended if answer is None else max(0, int(answer))

    def abandon_invasion(self, state, engine):
        if self.side != IMPERIAL:
            return super().abandon_invasion(state, engine)
        answer = self.ask({
            "kind": "abandon",
            "turn": state.turn,
            "cities_taken": len(state.geometry.urban) - len(state.solomani_urban()),
            "victory_points": state.victory_points(),
            "recommended": False,
        })
        return bool(answer) if answer is not None else False

    # -- the defence ----------------------------------------------------
    def sdb_surface(self, state, side, engine):
        recommended = super().sdb_surface(state, side, engine)
        if self.side != SOLOMANI or side != SOLOMANI:
            return recommended
        from ..state import CLOSE_ORBIT
        fleet = state.naval_at(CLOSE_ORBIT, IMPERIAL)
        hidden = [u for u in state.naval.values()
                  if u.is_sdb and u.hidden and u.side == SOLOMANI and not u.dummy]
        if not hidden:
            return recommended
        answer = self.ask({
            "kind": "surface_boats",
            "turn": state.turn,
            "fleet_overhead": len(fleet),
            "fleet_defence": sum(u.cls.defense for u in fleet),
            "wings_hidden": len(hidden),
            "recommended": len(recommended),
        })
        if answer is None:
            return recommended
        wanted = max(0, int(answer))
        hidden.sort(key=lambda u: (-u.cls.bombard, u.uid))
        return {u.uid for u in hidden[:wanted]}


def summarise(state) -> str:
    """One line a human can read between turns."""
    ashore = sum(u.current for u in state.surface.values()
                 if u.side == IMPERIAL and u.carrier is None
                 and isinstance(u.location, int) and not u.dead)
    afloat = sum(u.current for u in state.surface.values()
                 if u.side == IMPERIAL and u.carrier is not None and not u.dead)
    waiting = sum(u.current for u in state.surface.values()
                  if u.side == IMPERIAL and u.carrier is None and not u.dead
                  and not isinstance(u.location, int))
    solomani = sum(u.current for u in state.surface.values()
                   if u.side == SOLOMANI and not u.dead and not u.cls.guerrilla)
    return ("turn %2d | ashore %5.0f afloat %4.0f waiting %5.0f | "
            "Solomani %5.0f | %2d/%d cities | %d VP"
            % (state.turn, ashore, afloat, waiting, solomani,
               len(state.geometry.urban) - len(state.solomani_urban()),
               len(state.geometry.urban), state.victory_points()))

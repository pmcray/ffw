"""A doctrine for both sides, written as a weighted scoring rule.

The two problems in *Invasion: Earth* are not symmetrical and the doctrine is
not either.  The Imperial player has a shipping problem before he has a
fighting one: every unit he owns starts in the out-system box, a battle
squadron lifts twenty combat factors, and the only way to score is to put
troops on cities and keep them there.  The Solomani player owns the ground
already and has to make the orbit expensive -- thirty-four SDB wings hidden in
the oceans, planetary defences that shoot at anything landing within three
hexes, and guerrillas that cannot be bombarded at all.

The Imperial half of this file is written around three things the rules make
true and the first version of it missed:

* **A hex is garrisoned by a zone of control, and lost to one.**  Rule 7 gives
  a hex to the Imperium when it is in an Imperial zone of control and neither
  occupied by a Solomani unit nor in a Solomani zone of control.  So the
  objective is never the city itself; it is a hex to stand on near several
  cities, with the local Solomani dead.
* **Grav units have ten movement points and a hex is 1100 km.**  A doctrine
  that only ever looks at the six hexes next door is playing a tenth of the
  board.  Both sides can cross a third of the planet in a turn, which is what
  makes a beachhead a place to fight from rather than a place to hide.
* **The fleet is the only weapon that does not have to be shipped.**  A
  hundred-odd bombardment factors pooled on one static planetary defence
  division kills it in two turns, and every one killed is a city that can be
  garrisoned, a gun that stops firing at the fleet, and a gun that stops firing
  at the next landing.

The weights are named and collected in ``WEIGHTS`` so they can be optimised the
same way ``ffw``'s were, and so a doctrine trained on one game has something
with the same shape to transfer into.
"""

from __future__ import annotations

from .. import oob, terra
from ..state import (BOMBARDMENT, CLOSE_ORBIT, DEEP_SPACE, FAR_ORBIT, IMPERIAL,
                     LUNA, OUT_SYSTEM, OVERWATCH, SOLOMANI, enemy_of)
from .base import Agent

#: Every tunable, with the default that plays a reasonable game.
WEIGHTS = {
    # -- Imperial: where to land ---------------------------------------
    "urban_value": 1.0,          # how much a city in reach is worth
    "starport_value": 0.6,       # starports feed SDB construction
    "defended": -0.8,            # weight on the defenders already there
    "elbow_room": 0.5,           # land to stand on: an island is not a theatre
    "supply_reach": 0.7,         # stay inside a base's five hexes
    "guerrilla_cover": -0.3,     # terrain that shelters what shoots back
    "concentration": 0.6,        # how hard to mass rather than spread
    # -- Imperial: the convoy ------------------------------------------
    "wave_appetite": 0.25,       # each wave is a victory point
    "bases_ashore": 6.0,         # how many bases the supply net wants
    # -- Imperial: guns and ground -------------------------------------
    "bombard_first": 0.8,        # soften a hex before landing on it
    "counter_battery": 1.5,      # weight on planetary defences as targets
    "assault_odds": 3.0,         # odds wanted before walking into a hex
    "garrison_appetite": 0.6,    # how readily a corps detaches to hold ground
    "exposure": 0.5,             # weight on what can reach a hex next turn
    # -- Solomani: when to surface the boats -----------------------------
    "sdb_patience": 0.55,        # how big a target is worth coming up for
    "pd_vs_fleet": 0.5,          # split PD fire between orbit and ground
    "guerrilla_boldness": 0.35,  # how readily guerrillas fight in the open
    "reinforce_front": 0.7,      # pull troops towards the landings
    "break_garrisons": 0.8,      # peel units off to un-garrison cities
}


def weight_vector(weights=None):
    w = weights or WEIGHTS
    return [float(w.get(k, WEIGHTS[k])) for k in sorted(WEIGHTS)]


def weight_dict(vector):
    return {k: float(v) for k, v in zip(sorted(WEIGHTS), vector)}


class HeuristicAgent(Agent):
    """Plays both sides from one weight vector."""

    name = "heuristic"

    def __init__(self, side: str, weights=None, seed: int | None = None,
                 label: str | None = None):
        super().__init__(side, seed, label)
        self.w = dict(WEIGHTS)
        if weights:
            self.w.update({k: float(v) for k, v in weights.items()
                           if k in WEIGHTS})
        self._beachhead = None

    def begin_game(self, state, side, engine):
        self._beachhead = None

    # ==================================================================
    #  reading the board
    # ==================================================================
    def _batteries(self, state) -> list:
        """Every live planetary defence unit, with where it is and what it fires."""
        return [u for u in state.surface.values()
                if u.side == SOLOMANI and not u.dead and u.cls.planetary_defense
                and isinstance(u.location, int)]

    def _threat(self, state, cell) -> float:
        """Solomani strength that could be on this hex within a turn."""
        geo = state.geometry
        total = 0.0
        for unit in state.surface.values():
            if unit.side != SOLOMANI or unit.dead or unit.carrier is not None:
                continue
            if not unit.cls.mobile or not isinstance(unit.location, int):
                continue
            if geo.distance(unit.location, cell) <= oob.MOVEMENT_POINTS:
                total += unit.current
        return total

    def _threat_map(self, state) -> dict:
        """``_threat`` for every hex at once, which is what the ground pass wants.

        Asked one hex at a time it is a loop over the Solomani army; asked for
        thirty destinations for each of thirty units it is a million distance
        comparisons a turn.  Pushed outwards from each Solomani unit instead it
        is one pass over the army, and the radius bands are cached on the
        geometry and shared between clones.
        """
        geo = state.geometry
        out: dict = {}
        for unit in state.surface.values():
            if unit.side != SOLOMANI or unit.dead or unit.carrier is not None:
                continue
            if not unit.cls.mobile or not isinstance(unit.location, int):
                continue
            for cell in geo.within(unit.location, oob.MOVEMENT_POINTS):
                out[cell] = out.get(cell, 0.0) + unit.current
        return out

    def _gun_map(self, state) -> dict:
        """Planetary defence factors that can fire into each hex."""
        geo = state.geometry
        out: dict = {}
        for unit in self._batteries(state):
            for cell in geo.within(unit.location, oob.PD_RANGE):
                out[cell] = out.get(cell, 0.0) + float(unit.cls.bombard)
        return out

    def _bases_ashore(self, state) -> list:
        return [b for b in state.bases.values()
                if b.carrier is None and not b.dead
                and isinstance(b.location, int)]

    def _in_supply_range(self, state, cell) -> bool:
        """Cheap proxy for rule 5's trace: a base within four hexes.

        The engine traces a real path of up to five hexes and the trace can be
        blocked -- "the supply line may not be traced into or through a hex
        occupied by an enemy unit or in an enemy zone of control".  A doctrine
        that ran the full trace for every candidate hex would spend the whole
        turn doing it, so this asks for one hex in hand instead: a base four
        hexes away still supplies when the direct line is blocked and the trace
        has to go round.
        """
        geo = state.geometry
        return any(geo.distance(b.location, cell) <= oob.SUPPLY_RANGE - 1
                   for b in self._bases_ashore(state))

    # ==================================================================
    #  space
    # ==================================================================
    def naval_moves(self, state, side, engine):
        if side == IMPERIAL:
            return self._imperial_naval(state)
        return self._solomani_naval(state)

    def _waiting_out_system(self, state) -> float:
        total = sum(u.current for u in state.surface.values()
                    if u.side == IMPERIAL and u.carrier is None
                    and u.location == OUT_SYSTEM and not u.dead)
        total += sum(oob.BASE_TRANSPORT_COST for b in state.bases.values()
                     if b.carrier is None and b.location == OUT_SYSTEM
                     and not b.dead)
        return total

    def _imperial_naval(self, state):
        """Two jobs, and a box for each.

        The transports shuttle out-system to close orbit and back for the next
        load; the gun line lives in close orbit, because a bombardment mission
        may only be given to a unit that is already there.  Everything else
        waits in far orbit, where the planetary defences cannot reach it -- a
        squadron with nothing to do in close orbit is not idle, it is a defence
        factor being subtracted from the fleet at the rate of five or six a
        turn.

        Luna looks like the answer to that and is not, which is worth writing
        down because it took a rewrite to find out.  Luna sits in the far orbit
        box, out of range of every planetary defence unit on the planet, and
        "naval units may only load from or unload onto the surface of Luna" --
        so the transports could put the army down there and never enter close
        orbit at all.  The transport capability chart forbids it: a battle
        squadron lifts twenty combat factors and a cruiser five, so the only
        hulls that could carry a 100-factor corps off Luna again are the
        transports themselves.  An army unloaded onto Luna is an army parked on
        Luna for the rest of the war.
        """
        moves = {}
        waiting = self._waiting_out_system(state)

        for unit in state.naval.values():
            if unit.side != IMPERIAL or unit.hidden or unit.is_sdb:
                continue
            box = unit.location
            if not isinstance(box, str) or box == LUNA:
                continue

            if unit.cls.capacity >= oob.BASE_TRANSPORT_COST:
                if box == OUT_SYSTEM:
                    if unit.cargo:
                        moves[unit.uid] = DEEP_SPACE
                elif box == DEEP_SPACE:
                    if unit.cargo:
                        moves[unit.uid] = CLOSE_ORBIT
                    elif waiting > 0:
                        moves[unit.uid] = OUT_SYSTEM
                    else:
                        moves[unit.uid] = FAR_ORBIT
                elif box == FAR_ORBIT:
                    if unit.cargo:
                        moves[unit.uid] = CLOSE_ORBIT
                    elif waiting > 0:
                        moves[unit.uid] = DEEP_SPACE
                elif box == CLOSE_ORBIT and not unit.cargo:
                    # empty and under the guns: back out for the next load
                    moves[unit.uid] = (DEEP_SPACE if waiting > 0 else FAR_ORBIT)
                continue

            if unit.cargo:              # anything loaded goes to the beach
                if box == OUT_SYSTEM:
                    moves[unit.uid] = DEEP_SPACE
                elif box in (DEEP_SPACE, FAR_ORBIT):
                    moves[unit.uid] = CLOSE_ORBIT
                continue

            # the gun line, and everything with no gun kept out of the fire
            if unit.cls.bombard > 0:
                if box != CLOSE_ORBIT:
                    moves[unit.uid] = (DEEP_SPACE if box == OUT_SYSTEM
                                       else CLOSE_ORBIT)
            elif box == OUT_SYSTEM:
                moves[unit.uid] = DEEP_SPACE
            elif box != FAR_ORBIT:
                moves[unit.uid] = FAR_ORBIT
        return moves

    def _solomani_naval(self, state):
        """The squadrons are a fleet in being: they cannot jump and cannot win.

        Their job is to be somewhere the Imperium has to deal with, which means
        close orbit while it is cheap and deep space once it is not.
        """
        moves = {}
        pressure = len(state.naval_at(CLOSE_ORBIT, IMPERIAL))
        for unit in state.naval.values():
            if unit.side != SOLOMANI or unit.is_sdb or unit.hidden:
                continue
            if pressure >= 6 and unit.location != DEEP_SPACE:
                moves[unit.uid] = DEEP_SPACE
            elif pressure < 3 and unit.location == DEEP_SPACE:
                moves[unit.uid] = CLOSE_ORBIT
        return moves

    def hide_in_deep_space(self, state, side, engine):
        if side != SOLOMANI:
            return set()
        return {u.uid for u in state.naval.values()
                if u.side == SOLOMANI and u.location == DEEP_SPACE
                and not u.is_sdb}

    def disengage(self, state, side, box, engine):
        """Break off when the exchange has stopped being worth it."""
        own = state.naval_at(box, side)
        foe = state.naval_at(box, enemy_of(side))
        if not own or not foe:
            return False
        mine = sum(u.cls.attack + u.bombard_factor() for u in own)
        theirs = sum(u.cls.attack + u.bombard_factor() for u in foe)
        if side == SOLOMANI:
            # the Solomani cannot replace a squadron; the Imperium can
            return mine < theirs * 0.9
        return mine < theirs * 0.6

    def take_another_naval_phase(self, state, side, engine):
        """Press a won battle, once."""
        if side != IMPERIAL:
            return False
        return not state.naval_at(CLOSE_ORBIT, SOLOMANI)

    # ==================================================================
    #  space-surface
    # ==================================================================
    def missions(self, state, side, engine):
        """Point the guns, in pools big enough to matter.

        The surface bombardment table is read on the total of everything firing
        at one unit, so the shape of this decision is how many pools to make out
        of a hundred-odd bombardment factors.  Two pools of fifty average forty
        percent a turn each; four of twenty-five average twenty-three.  Four
        pools kill slightly more planetary defence per turn and two kill each
        one faster, and killing faster is what counts: "the PD unit's
        bombardment factor always remains at full strength until the PD unit is
        entirely eliminated", so a battery at ninety percent losses is still
        firing a full broadside at the fleet every turn.  Damage that does not
        finish something buys nothing at all.

        A squadron on a bombardment mission is "placed on the map in the hex
        containing the surface unit it is to bombard", which is also the only
        way to be somewhere other than close orbit and still be shooting.
        """
        if side != IMPERIAL:
            return {}
        orders = {}
        targets = self._bombardment_targets(state)
        guns = [u for u in state.naval_at(CLOSE_ORBIT, IMPERIAL)
                if u.cls.bombard > 0 and not u.cargo]
        guns.sort(key=lambda u: (-u.cls.bombard, u.uid))
        if not guns:
            return orders
        # A wing that surfaces unopposed gets a free shot at the fleet, so keep
        # a watch on the water -- but only when there is something in close
        # orbit worth surfacing for.  The Solomani boats come up for a fleet and
        # stay down for an empty box, and everything on a bombardment mission is
        # on the map rather than in the box.
        #
        # The watch is drawn from the *cheapest* squadrons, and that is the
        # whole point of it.  On a turn when a transport is unloading, the
        # planetary defences fire one pooled attack into close orbit and "the
        # owning player chooses which naval units are eliminated" -- cheapest
        # first, so a screen of colonial squadrons is the armour that keeps a
        # six-hundred-factor transport and the regiments inside it alive for
        # another run.  Made of the best guns instead, as it was, the screen is
        # worth less than the transport it fails to save.
        convoy = [u for u in state.naval_at(CLOSE_ORBIT, IMPERIAL) if u.cargo]
        screen = []
        if convoy and len(guns) > 6:
            for unit in sorted(guns, key=lambda u: (u.cls.defense, u.uid)):
                if sum(s.cls.defense for s in screen) >= 8:
                    break
                screen.append(unit)
        for unit in screen:
            orders[unit.uid] = (OVERWATCH, None)
        firing = [u for u in guns if u not in screen]
        if not targets or self.w["bombard_first"] <= 0:
            return orders
        # Two pools at least, whatever the fleet is down to: the planetary
        # defences answer each bombarding group with one pooled attack, so
        # splitting the guns splits the fire coming back at them.
        pools = min(len(targets), max(2, int(
            sum(u.bombard_factor() for u in firing) // 36)))
        for i, unit in enumerate(firing):
            orders[unit.uid] = (BOMBARDMENT, targets[i % pools])
        return orders

    def _target_value(self, state, unit) -> float:
        """What one bombardment attack on this unit is worth, in combat factors.

        The table returns a *percentage* of printed strength, so the arithmetic
        is not the one it looks like.  Thirty percent of a 500-factor field army
        is a hundred and fifty factors destroyed in a phase, which is a won
        battle every turn and more than the Solomani rebuild in a quarter.
        Thirty percent of a 20-factor planetary defence division is six.

        What makes a battery worth shooting at anyway is everything except its
        combat factor: it fires on the fleet every turn, it fires on every
        landing within three hexes at -3, it cannot be replaced -- and its zone
        of control freezes the city it sits in out of the scoreboard.  Those are
        priced here in factors so the two kinds of target can be compared at
        all, and the price rises steeply as the unit nears elimination, because
        "the PD unit's bombardment factor always remains at full strength until
        the PD unit is entirely eliminated" -- damage short of a kill buys
        nothing.
        """
        geo, w = state.geometry, self.w
        if not unit.cls.planetary_defense:
            return 0.35 * unit.current
        cell = unit.location
        cities = sum(1 for c in geo.adjacent(cell) + (cell,)
                     if (c in geo.urban or c in geo.starports)
                     and c not in state.garrisoned)
        value = w["counter_battery"] * (8.0 * unit.cls.bombard + 15.0 * cities)
        return value * (1.0 + 2.0 * unit.losses / 100.0)

    def _bombardment_targets(self, state):
        """Which hexes the guns are worth pointing at.

        Guerrillas are skipped: "planetary bombardment attacks may not be made
        against guerrilla units", so a squadron sent at one wastes its turn.

        The other correction is what shoots back.  A squadron on a bombardment
        mission stands in the hex it is bombarding, where every planetary
        defence unit within three hexes can fire on it at full factors instead
        of the halved ones it would use against close orbit.  A battery with two
        friends inside three hexes is not one target, it is three guns pointed
        at the fleet, and the value of hitting it is divided accordingly.
        """
        geo = state.geometry
        beach = self._beachhead
        guns = self._gun_map(state)
        scored = {}
        for unit in state.surface.values():
            if unit.side != SOLOMANI or unit.dead or unit.cls.guerrilla:
                continue
            if not isinstance(unit.location, int):
                continue
            cell = unit.location
            value = self._target_value(state, unit)
            if beach is not None:
                value /= 1.0 + geo.distance(beach, cell) / 8.0
            scored[cell] = scored.get(cell, 0.0) + value
        for cell in scored:
            scored[cell] /= 1.0 + guns.get(cell, 0.0) / 4.0
        if not scored:
            return []
        ranked = sorted(scored, key=lambda c: (-scored[c], c))
        return ranked[:4]

    def bombardment_target(self, state, side, cell, candidates, engine):
        """In a bombarded hex, whichever unit the attack is worth most against."""
        if side != IMPERIAL:
            return super().bombardment_target(state, side, cell, candidates,
                                              engine)
        return max(candidates,
                   key=lambda u: (self._target_value(state, u), u.uid))

    def sdb_surface(self, state, side, engine):
        """Come up when there is enough overhead to be worth the exposure.

        A surfaced wing shoots once and is shot at by every overwatch squadron
        in orbit, so the question is whether the fleet above is big enough to
        pay for the wing.
        """
        if side != SOLOMANI:
            return set()
        fleet = state.naval_at(CLOSE_ORBIT, IMPERIAL)
        if not fleet:
            return set()
        worth = sum(u.cls.defense for u in fleet)
        threshold = 6 + 40 * self.w["sdb_patience"]
        if worth < threshold:
            return set()
        wings = [u for u in state.naval.values()
                 if u.is_sdb and u.hidden and u.side == SOLOMANI and not u.dummy]
        wings.sort(key=lambda u: -u.cls.bombard)
        return {u.uid for u in wings[:max(1, len(wings) // 3)]}

    def defence_fire(self, state, side, engine):
        """Point the planetary defences at whichever is worth more this turn.

        Three targets are on offer and they are not equal.  A squadron that has
        come down to bombard is "placed on the map in the hex containing the
        surface unit it is to bombard" -- on the map, inside a battery's three
        hexes, and fired on at *full* bombardment factors, where fire at the
        close orbit box is halved.  A battery with a bombarding squadron in
        reach has twice the gun it would have pointing upwards, and the
        squadron it is shooting at is the one doing the damage.
        """
        if side != SOLOMANI:
            return {}
        orders = {}
        fleet = state.naval_at(CLOSE_ORBIT, IMPERIAL)
        fleet_worth = sum(u.cls.defense for u in fleet)
        raiders = {}
        for unit in state.naval.values():
            if unit.side != IMPERIAL or not isinstance(unit.location, int):
                continue
            raiders[unit.location] = raiders.get(unit.location, 0) \
                + unit.cls.defense

        def nearest_raid(origin):
            if not isinstance(origin, int):
                return None
            best, best_key = None, None
            for cell, worth in raiders.items():
                d = state.geometry.distance(origin, cell)
                if d > oob.PD_RANGE:
                    continue
                key = (-worth, d, cell)
                if best_key is None or key < best_key:
                    best, best_key = cell, key
            return best

        for unit in state.surface.values():
            if unit.side != SOLOMANI or unit.dead or not unit.cls.planetary_defense:
                continue
            raid = nearest_raid(unit.location)
            if raid is not None:
                orders[unit.uid] = ("naval", raid)
                continue
            ashore = self._nearest_landed(state, unit.location)
            if ashore is None or fleet_worth * self.w["pd_vs_fleet"] > 8:
                if fleet:
                    orders[unit.uid] = ("naval", CLOSE_ORBIT)
            else:
                orders[unit.uid] = ("surface", ashore.uid)
        for wing in state.naval.values():
            if wing.is_sdb and not wing.hidden and wing.side == SOLOMANI:
                raid = nearest_raid(wing.location)
                if raid is not None:
                    orders[wing.uid] = ("naval", raid)
                elif fleet:
                    orders[wing.uid] = ("naval", CLOSE_ORBIT)
        return orders

    def _nearest_landed(self, state, origin):
        if not isinstance(origin, int):
            return None
        best, best_d = None, 99
        for unit in state.surface.values():
            if unit.side != IMPERIAL or unit.dead or unit.carrier is not None:
                continue
            if not isinstance(unit.location, int):
                continue
            d = state.geometry.distance(origin, unit.location)
            if d <= oob.PD_RANGE and d < best_d:
                best, best_d = unit, d
        return best

    # ==================================================================
    #  the convoy
    # ==================================================================
    def landings(self, state, side, engine):
        if side != IMPERIAL:
            return []
        orders = []
        orders += self._load_out_system(state)
        orders += self._put_ashore(state)
        return orders

    def _cargo_cost(self, state, uid):
        if uid in state.bases:
            return oob.BASE_TRANSPORT_COST
        unit = state.surface.get(uid)
        return unit.current if unit else 0.0

    def _room(self, state, carrier) -> float:
        return carrier.cls.capacity - sum(self._cargo_cost(state, uid)
                                          for uid in carrier.cargo)

    def _base_demand(self, state) -> int:
        """How many bases to put on the next convoy, which is one or none.

        Never the whole supply net at once.  Four bases is four hundred factors
        of lift out of about twelve hundred a run, the army needs that lift
        more, and all four land in the same hex anyway because that is the only
        hex the army is standing in yet -- four bases in one hex is one base.
        So: a steady one a convoy until three are down, and after that only when
        there are troops outside every existing base's five hexes, which is what
        an advance looks like from the supply net's point of view.
        """
        placed = [b.location for b in self._bases_ashore(state)]
        in_transit = sum(1 for b in state.bases.values()
                         if not b.dead and b.carrier is not None)
        if in_transit:
            return 0
        if len(placed) + in_transit >= int(self.w["bases_ashore"]):
            return 0
        if len(placed) < 3:
            # Get a net down while there is still lift to do it with.  A base
            # needs a hundred factors of transport capability, which means a
            # transport squadron, and the transports are what the planetary
            # defences kill first: bases not landed by the time the last one
            # goes are bases that stay out-system for the rest of the war.
            return 1
        geo = state.geometry
        uncovered = 0.0
        for unit in state.surface.values():
            if unit.side != IMPERIAL or unit.dead or unit.carrier is not None:
                continue
            if not isinstance(unit.location, int):
                continue
            if not any(geo.distance(p, unit.location) <= oob.SUPPLY_RANGE - 1
                       for p in placed):
                uncovered += unit.current
        return 1 if uncovered >= 100 else 0

    def _load_out_system(self, state):
        """Fill every hull waiting out-system, biggest unit that fits first.

        This is the lesson ``ffw`` paid for: a loader that offers units
        biggest-first and stops at the head of the list can leave a fleet empty
        because the largest army does not fit in a cruiser.  Take the largest
        unit that *fits*.
        """
        orders = []
        waiting = sorted(
            (u for u in state.surface.values()
             if u.side == IMPERIAL and u.carrier is None
             and u.location == OUT_SYSTEM and not u.dead),
            key=lambda u: (-u.current, u.uid))
        bases = [b for b in state.bases.values()
                 if b.carrier is None and b.location == OUT_SYSTEM and not b.dead]
        hulls = sorted((u for u in state.naval.values()
                        if u.side == IMPERIAL and u.location == OUT_SYSTEM),
                       key=lambda u: (-u.cls.capacity, u.uid))
        if state.turn < oob.WITHDRAWAL_TURN:
            # Two transport squadrons are withdrawn on turn 2 and whatever is
            # aboard them goes too.  It is printed on the order of battle
            # chart, so there is no excuse for loading the army onto them.
            doomed = [u for u in hulls
                      if u.cls.kind == oob.WITHDRAWAL_KIND][:oob.WITHDRAWAL_COUNT]
            hulls = [u for u in hulls if u not in doomed]
        wanted_bases = self._base_demand(state)
        for carrier in hulls:
            room = self._room(state, carrier)
            if room <= 0:
                continue
            # At most two a run.  A base is a hundred factors of lift, the
            # invasion has about twelve hundred a run, and a transport lost on
            # the way in takes everything aboard with it -- so a convoy carrying
            # the whole supply net is one bad roll from an army that can never
            # be in supply anywhere.
            aboard = 0
            while bases and room >= oob.BASE_TRANSPORT_COST \
                    and wanted_bases > 0 and aboard < 2:
                base = bases.pop()
                orders.append(("load", base.uid, carrier.uid))
                room -= oob.BASE_TRANSPORT_COST
                wanted_bases -= 1
                aboard += 1
            for unit in list(waiting):
                if unit.current <= room:
                    orders.append(("load", unit.uid, carrier.uid))
                    room -= unit.current
                    waiting.remove(unit)
                if room <= 0:
                    break
        return orders

    # -- landings ------------------------------------------------------
    def _put_ashore(self, state):
        """Land the army across the lodgement, assault troops first.

        Landing straight onto a defended city is how this invasion dies, and
        the rules say so twice.  A unit landing in or leaving a hex is fired on
        by every planetary defence within three hexes, at a **-3** on the
        surface bombardment table unless it is a marine or jump troop unit --
        and a corps that survives that arrives in a hex holding a 500-factor
        field army, having already spent the turn it landed.

        So the ground is chosen for being empty and out of range, the assault
        troops go first because they are the ones the -3 does not apply to, and
        the army is spread across the lodgement rather than piled on one hex:
        rule 3 allows a thousand combat factors in a hex and the Imperium owns
        four thousand.
        """
        cells = self._lodgement(state)
        if not cells:
            return []
        anchor = cells[0]
        room = {c: oob.STACKING_LIMIT
                - sum(u.current for u in state.surface_at(c, IMPERIAL))
                for c in cells}
        cargo = []
        for carrier in state.naval_at(CLOSE_ORBIT, IMPERIAL):
            cargo.extend(carrier.cargo)

        def priority(uid):
            unit = state.surface.get(uid)
            if unit is None:
                return (2, 0.0)          # a base goes last of all
            if unit.cls.marine or unit.cls.jump:
                return (0, -unit.current)
            return (1, -unit.current)

        orders, sites = [], []
        for uid in sorted(cargo, key=priority):
            unit = state.surface.get(uid)
            if unit is None:
                where = self._base_site(state, cells, sites)   # a base
                if where is None:
                    continue
                sites.append(where)
                orders.append(("unload", uid, where))
                continue
            # Fill the anchor before spilling into the next hex.  Spreading a
            # six-hundred-factor wave over four hexes is four fights the
            # Solomani win; rule 3's thousand-factor limit is the only reason
            # there is a second hex at all.
            for cell in sorted(cells, key=lambda c: (c != anchor, -room[c], c)):
                if unit.current <= room[cell]:
                    orders.append(("unload", uid, cell))
                    room[cell] -= unit.current
                    break
        return orders

    def _base_site(self, state, cells, taken):
        """Where the next base goes: wherever the army has outrun the last one.

        A base is the only Imperial source of supply, it cannot walk once it is
        down, and rule 5 traces supply five hexes.  An army that advances on the
        cities leaves that circle almost at once, and a unit out of supply "may
        not fire" and defends at half strength -- so it is not a weakened corps,
        it is a corps that has stopped being a combat unit at all.

        So the site is chosen for the troops it brings back into supply, not for
        the beachhead it tidies.  ``taken`` carries the sites already chosen in
        this landing phase, because the bases are not on the ground yet and four
        of them picking the same best hex is how the supply net ends up being
        one base four times over.
        """
        geo = state.geometry
        placed = [b.location for b in self._bases_ashore(state)] + list(taken)
        if len(placed) >= int(self.w["bases_ashore"]):
            return None
        guns = self._gun_map(state)
        troops: dict = {}
        for unit in state.surface.values():
            if unit.side != IMPERIAL or unit.dead or unit.carrier is not None:
                continue
            if isinstance(unit.location, int):
                troops[unit.location] = troops.get(unit.location, 0.0) \
                    + unit.current
        threat = self._threat_map(state)
        candidates = set()
        for cell in troops:
            candidates.update(geo.within(cell, oob.SUPPLY_RANGE - 1))
        candidates |= set(cells)
        candidates = [c for c in candidates
                      if c in geo.land
                      and geo.terrain[c] not in (terra.TUNDRA, terra.PERMANENT_ICE,
                                                 terra.SEASONAL_ICE)
                      and not state.surface_at(c, SOLOMANI)]
        if not candidates:
            return None

        def covered(cell) -> float:
            """Factors this site would bring into supply that are not already."""
            total = 0.0
            for where, strength in troops.items():
                if geo.distance(cell, where) > oob.SUPPLY_RANGE:
                    continue
                if any(geo.distance(p, where) <= oob.SUPPLY_RANGE for p in placed):
                    continue
                total += strength
            return total

        def score(cell) -> float:
            value = covered(cell)
            # and, when there is nothing new to cover, lay the net towards the
            # cities: the army will not advance out of supply, so the base goes
            # first and the corps follow it
            value += 25.0 * sum(1 for c in geo.within(cell, oob.SUPPLY_RANGE - 1)
                                if c in geo.urban and c not in state.garrisoned)
            value -= 30.0 * guns.get(cell, 0.0)
            # a base "is automatically destroyed when there are enemy surface
            # units but no friendly surface units in the base's hex", and it
            # cannot walk away, so it is put down where the field armies are not
            value -= threat.get(cell, 0.0) / 4.0
            if placed and min(geo.distance(cell, p) for p in placed) <= 1:
                value -= 100.0        # two bases in one hex is one base
            return value

        return max(candidates, key=lambda c: (score(c), troops.get(c, 0.0), -c))

    def _lodgement(self, state):
        """The hexes the invasion is landing on: an anchor and its neighbours.

        Held across turns.  Supply is traced to Imperial bases, a base cannot
        move once landed, and a unit out of supply may not fire and defends at
        half strength -- so the lodgement is abandoned only when the ground
        under it is gone.
        """
        geo = state.geometry
        anchor = self._beachhead
        if anchor is None or not self._lodgement_holds(state, anchor):
            anchor = self._pick_anchor(state)
            self._beachhead = anchor
        if anchor is None:
            return []
        cells = [anchor]
        for cell in sorted(geo.adjacent(anchor)):
            if cell in geo.land and not state.surface_at(cell, SOLOMANI):
                cells.append(cell)
        return cells

    def _lodgement_holds(self, state, anchor) -> bool:
        if state.surface_at(anchor, SOLOMANI):
            return False
        if self._bases_ashore(state):
            return True                  # the base is there; the ground stays
        return self._threat(state, anchor) < 1500

    def _pick_anchor(self, state):
        """Score every land hex once, on maps built once."""
        field = (self._threat_map(state), self._gun_map(state),
                 self._close_map(state))
        best, best_score = None, -1e18
        for cell in sorted(state.geometry.land):
            score = self._landing_score(state, cell, field)
            if score > best_score:
                best, best_score = cell, score
        return best

    def _close_map(self, state) -> dict:
        """Solomani strength within three hexes of each cell.

        The distinction between this and ``_threat_map`` is the whole beachhead
        problem.  Everything on Terra is within ten movement points of
        everything else, so the strength that *could* arrive next turn is
        nearly a constant and cannot choose between two hexes.  The strength
        that is already on top of a hex can, and it is what decides whether the
        first eight hundred factors ashore survive to be reinforced.
        """
        geo = state.geometry
        out: dict = {}
        for unit in state.surface.values():
            if unit.side != SOLOMANI or unit.dead or unit.carrier is not None:
                continue
            if not isinstance(unit.location, int):
                continue
            for cell in geo.within(unit.location, 3):
                out[cell] = out.get(cell, 0.0) + unit.current
        return out

    def _landing_score(self, state, cell, field=None) -> float:
        """What makes a hex worth landing on.

        Not its own value -- an empty field scores nothing by itself.  What is
        being bought is a place to stand: clear of the defenders already in
        reach, out from under the planetary defences, with room around it for
        four thousand combat factors at a thousand to the hex, and with cities
        inside a march.

        *Inside a march* is the correction that matters.  Counting only cities
        within three hexes, while charging thirty points a gun for every
        planetary defence within three hexes, asks for a hex that is next to the
        cities and away from the guns -- and the guns are *in* the cities, so no
        such hex exists.  What maximised that score was an empty island with
        three distant cities and no way off it, and an invasion that landed
        there and sat out the war.  A grav corps has ten movement points; cities
        seven hexes away are a turn's march, not another theatre.
        """
        geo, w = state.geometry, self.w
        if field is None:
            field = (self._threat_map(state), self._gun_map(state),
                     self._close_map(state))
        threat, guns, close = field
        terrain = geo.terrain[cell]
        if terrain in (terra.PERMANENT_ICE, terra.SEASONAL_ICE, terra.TUNDRA):
            # A base may not be landed in a tundra or ice hex, and an Imperial
            # unit with no base within five hexes is out of supply: it may not
            # fire and it defends at half strength.  A lodgement that cannot
            # hold a base is not a lodgement, it is a hostage.
            return -1e9
        room = sum(1 for c in geo.within(cell, 2) if c in geo.land)
        if room < 5:
            return -1e9          # a rock in the ocean is not a theatre
        score = w["elbow_room"] * room
        defenders = sum(u.current for u in state.surface_at(cell, SOLOMANI))
        score += w["defended"] * defenders / 20.0
        score -= w["exposure"] * close.get(cell, 0.0) / 25.0
        # and what can reach it next turn, which on this map is nearly
        # everything: grav troops have ten movement points and a hex is
        # 1100 km, so the only real cover is distance from the mass
        score -= w["exposure"] * threat.get(cell, 0.0) / 400.0
        near = sum(1 for c in geo.within(cell, 3)
                   if c in geo.urban and c not in state.garrisoned)
        march = sum(1 for c in geo.within(cell, 7)
                    if c in geo.urban and c not in state.garrisoned)
        score += w["urban_value"] * (near + march)
        ports = sum(1 for c in geo.within(cell, 4) if c in geo.starports)
        score += w["starport_value"] * ports
        # This is the heaviest term in the whole score, and deliberately.  A
        # unit landing within three hexes of the planetary defences is fired on
        # at -3 on the surface bombardment table before it has done anything,
        # which at any real strength is thirty to fifty percent losses on
        # arrival.  Nothing a beachhead can be worth pays for that.
        score -= 3.0 * guns.get(cell, 0.0)
        if state.surface_at(cell, IMPERIAL):
            score += w["concentration"] * 3.0
        if self._in_supply_range(state, cell):
            score += w["supply_reach"] * 2.0
        if terrain in terra.GUERRILLA_COVER:
            score += w["guerrilla_cover"]
        return score

    # ==================================================================
    #  surface
    # ==================================================================
    def surface_moves(self, state, side, engine):
        if side == IMPERIAL:
            return self._imperial_surface(state, engine)
        return self._solomani_surface(state)

    def _garrison_value(self, state, cell, solomani_zoc, occupied) -> int:
        """Cities a zone of control planted on this hex would garrison.

        "An Imperial unit having a zone of control is able to garrison all hexes
        in its zone of control.  A hex ... is considered to be garrisoned as
        long as the hex is neither occupied by a Solomani unit nor in the zone
        of control of a Solomani unit."  So the question a corps asks about a
        hex is never what is on it, but what is in the seven hexes it would
        cover, and which of those the Solomani have already spoiled.
        """
        geo = state.geometry
        value = 0
        for c in (cell,) + tuple(geo.adjacent(cell)):
            if c not in geo.urban and c not in geo.starports:
                continue
            if c in state.garrisoned or c in occupied or c in solomani_zoc:
                continue
            value += 1
        return value

    def _imperial_surface(self, state, engine):
        """Fight where the odds are good; otherwise go and hold something.

        Two passes.  The first looks for Solomani stacks the army can actually
        beat and sends enough at them to win the odds, because the troop combat
        table is steeply convex and a 1:1 attack is worse than no attack at all.
        The second sends whatever is left to the hexes that garrison the most
        cities, which is how the game is scored.

        Both passes are held down by the same constraint, and it is the one the
        first version of this doctrine did not have.  There are sixty-one cities
        and a hundred-factor corps can garrison three of them, so a scoring rule
        that only counts cities sends every corps to a different one -- and
        thirty-six hundred factors of grav-mobile Solomani troops, which can
        reach any hex on the planet in a turn, eat them one at a time.  A hex is
        only worth standing on if what is standing there can survive what can
        reach it.  Early that leaves the army in one stack; the war is won by
        making it stop being true, not by pretending it isn't.
        """
        geo, w = state.geometry, self.w
        solomani_zoc = state.zone_of_control(SOLOMANI)
        occupied = {u.location for u in state.surface.values()
                    if u.side == SOLOMANI and u.carrier is None and not u.dead
                    and isinstance(u.location, int)}
        units = [u for u in state.surface.values()
                 if u.side == IMPERIAL and not u.dead and u.carrier is None
                 and u.cls.mobile and isinstance(u.location, int)
                 and u.landed_turn != state.turn]
        if not units:
            return {}
        units.sort(key=lambda u: (-u.combat_strength(), u.uid))
        costs = {u.uid: engine.movement_costs(u, solomani_zoc, occupied)
                 for u in units}
        planned = engine._stacking(IMPERIAL)
        field = (self._threat_map(state), self._gun_map(state))
        threat = field[0]
        moves: dict = {}
        spent: set = set()

        def safe(cell, strength) -> bool:
            """Can this much of the army hold this hex against what can reach it?

            Rule 3's thousand-factor stacking limit is what makes the question
            answerable: however many field armies are in range, only a thousand
            factors of them can be in the hex at the end of their movement, so a
            stack near the limit is never facing worse than even odds.
            """
            arriving = min(threat.get(cell, 0.0), float(oob.STACKING_LIMIT))
            return strength >= arriving * self.w["exposure"]

        # -- pass one: attacks worth making ----------------------------
        defence = {}
        for cell in occupied:
            defence[cell] = sum(u.defence_strength()
                                for u in state.surface_at(cell, SOLOMANI))
        for cell in sorted(occupied, key=lambda c: (defence[c], c)):
            want = defence[cell] * w["assault_odds"]
            if want <= 0:
                continue
            able = [u for u in units
                    if u.uid not in spent and cell in costs[u.uid]]
            able.sort(key=lambda u: (-u.combat_strength(), u.uid))
            have = sum(u.current for u in state.surface_at(cell, IMPERIAL))
            going, strength = [], sum(u.combat_strength()
                                      for u in state.surface_at(cell, IMPERIAL))
            for unit in able:
                if strength >= want:
                    break
                if have + unit.current > oob.STACKING_LIMIT:
                    break
                going.append(unit)
                strength += unit.combat_strength()
                have += unit.current
            if strength < want or not going:
                continue
            if not safe(cell, have):
                continue          # winning the hex and losing the force in it
            for unit in going:
                moves[unit.uid] = cell
                spent.add(unit.uid)
                planned[unit.location] = planned.get(unit.location, 0.0) - unit.current
                planned[cell] = planned.get(cell, 0.0) + unit.current

        # -- pass two: ground worth holding ----------------------------
        for unit in units:
            if unit.uid in spent:
                continue
            here = unit.location
            best, best_score = here, self._post_score(
                state, unit, here, 0, planned, solomani_zoc, occupied, field,
                safe)
            for cell, cost in costs[unit.uid].items():
                if cell == here or cell not in geo.land or cell in occupied:
                    continue
                if planned.get(cell, 0.0) + unit.current > oob.STACKING_LIMIT:
                    continue
                score = self._post_score(state, unit, cell, cost, planned,
                                         solomani_zoc, occupied, field, safe)
                if score > best_score:
                    best, best_score = cell, score
            if best != here:
                moves[unit.uid] = best
                planned[here] = planned.get(here, 0.0) - unit.current
                planned[best] = planned.get(best, 0.0) + unit.current
        return moves

    def _post_score(self, state, unit, cell, cost, planned, solomani_zoc,
                    occupied, field, safe) -> float:
        """What a hex is worth to one unit, in cities and in risk.

        The safety term is a cliff and not a slope, deliberately.  Three cities
        garrisoned is worth about seven points here and a corps is worth a
        hundred combat factors; there is no exchange rate at which walking into
        a hex that cannot be held is worth the cities it would hold for one
        turn, so the score does not offer one.
        """
        w = self.w
        threat, guns = field
        friends = planned.get(cell, 0.0)
        if unit.location == cell:
            friends -= unit.current
        score = w["urban_value"] * w["garrison_appetite"] * 3.0 * \
            self._garrison_value(state, cell, solomani_zoc, occupied)
        score += w["concentration"] * min(friends, 600.0) / 200.0
        if not safe(cell, friends + unit.current):
            score -= 20.0 * w["exposure"]
        if not self._in_supply_range(state, cell):
            # "A unit that is out of supply may not fire (attack) ... and is
            # halved when fired upon."  It still garrisons -- a zone of control
            # does not need supply -- but it has stopped being a combat unit,
            # and a corps that cannot fight is a corps the Solomani can take
            # the city back from at their convenience.  This cliff is set above
            # anything the garrison term can offer on purpose: the army follows
            # its supply, and the way to advance is to land a base first.
            score -= 15.0 * w["supply_reach"]
        score -= 0.3 * guns.get(cell, 0.0)
        score -= 0.05 * cost
        return score

    def _solomani_surface(self, state):
        """Meet the landing, and spoil what it has taken.

        Two jobs, and the second is cheaper than the first.  A hex is
        garrisoned only while it is out of every Solomani zone of control, so a
        single division parked next to a captured city takes it back off the
        scoreboard without fighting for it -- and every ungarrisoned city is a
        replacement point a turn.
        """
        geo, w = state.geometry, self.w
        landings = {}
        for unit in state.surface.values():
            if unit.side != IMPERIAL or unit.dead or unit.carrier is not None:
                continue
            if isinstance(unit.location, int):
                landings[unit.location] = landings.get(unit.location, 0.0) \
                    + unit.current
        if not landings or w["reinforce_front"] <= 0:
            return {}
        spoilers = sorted(state.garrisoned)
        claimed: set = set()
        moves = {}
        movers = [u for u in state.surface.values()
                  if u.side == SOLOMANI and not u.dead and u.carrier is None
                  and u.cls.mobile and not u.cls.guerrilla
                  and isinstance(u.location, int)]
        movers.sort(key=lambda u: (u.current, u.uid))
        for unit in movers:
            here = unit.location
            # the small units go and break garrisons; the heavy ones mass
            if unit.current <= 20 and spoilers and w["break_garrisons"] > 0:
                target = min(
                    (c for c in spoilers if c not in claimed),
                    key=lambda c: (geo.distance(here, c), c), default=None)
                if target is not None and \
                        geo.distance(here, target) <= oob.MOVEMENT_POINTS:
                    claimed.add(target)
                    moves[unit.uid] = target
                    continue
            target = min(landings, key=lambda c: (geo.distance(here, c), c))
            d = geo.distance(here, target)
            if 0 < d <= oob.MOVEMENT_POINTS:
                moves[unit.uid] = target
        return moves

    def allocate_fire(self, state, side, cell, engine):
        """Concentrate on the target that most improves the odds.

        Splitting fire is legal and usually wrong: the troop combat table is
        steeply convex in the odds, so two 1:1 attacks are worth much less than
        one 2:1.  The exception is a target already at very low strength, which
        soaks far more fire than it needs.
        """
        firers = [u for u in state.surface_at(cell, side)
                  if not (u.cls.guerrilla and u.hiding)]
        targets = state.surface_at(cell, enemy_of(side))
        if not firers or not targets:
            return []
        total = sum(u.combat_strength() for u in firers)
        orders = []
        remaining = total
        # spend just enough on each weak target to overwhelm it, then put the
        # rest on the strongest thing present
        weak = sorted((t for t in targets if t.defence_strength() * 3 < total),
                      key=lambda t: t.defence_strength())
        pool = list(firers)
        for target in weak[:2]:
            need = target.defence_strength() * 3
            if need <= 0 or need > remaining * 0.5:
                continue
            spent = 0.0
            while pool and spent < need:
                unit = pool.pop(0)
                orders.append((unit.uid, target.uid, unit.combat_strength()))
                spent += unit.combat_strength()
            remaining -= spent
        if pool:
            main = max(targets, key=lambda t: t.defence_strength())
            for unit in pool:
                orders.append((unit.uid, main.uid, unit.combat_strength()))
        return orders

    def guerrilla_hiding(self, state, side, engine):
        """Hide unless the odds are good enough to be worth firing.

        Hiding buys +3 on every attack against the unit and costs it its fire.
        In cover -- urban, desert, wilderness -- that trade is nearly always
        right; in the open the unit is dead either way and may as well shoot.
        """
        out = set()
        for unit in state.surface.values():
            if unit.side != SOLOMANI or not unit.cls.guerrilla or unit.dead:
                continue
            if not isinstance(unit.location, int):
                continue
            terrain = state.geometry.terrain[unit.location]
            if terrain not in terra.GUERRILLA_COVER:
                continue
            enemy = sum(u.current for u in state.surface_at(unit.location,
                                                            IMPERIAL))
            if enemy > unit.current * self.w["guerrilla_boldness"]:
                out.add(unit.uid)
        return out

    # ==================================================================
    #  the quarterly special turn
    # ==================================================================
    def replacement_waves(self, state, engine):
        """Buy replacements only while they can still buy the war.

        Every wave is a victory point off a total that starts at ten and is
        already paying a point a quarter, so a player who buys six waves has
        given away more than the invasion is likely to win.  A wave also
        replaces three eliminated squadrons, which is the only way lost lift
        ever comes back.
        """
        losses = len(state.pools.get("dead_surface", []))
        budget = int(max(0.0, self.w["wave_appetite"] * 8))
        if state.waves_taken >= budget:
            return 0
        ashore = any(u.side == IMPERIAL and u.carrier is None
                     and isinstance(u.location, int) and not u.dead
                     for u in state.surface.values())
        if not ashore:
            return 0                 # nothing to reinforce yet
        return 1 if losses >= 4 else 0

    def spend_replacements(self, state, side, engine):
        """The engine spends the accumulated points; nothing extra to decide."""

    def place_guerrillas(self, state, count, engine):
        """Put them in cover, spread out, and near what the Imperium wants."""
        geo = state.geometry
        wanted = sorted(geo.urban | geo.starports)
        scored = []
        for cell in geo.land:
            if geo.terrain[cell] not in terra.GUERRILLA_COVER:
                continue
            near = min((geo.distance(cell, c) for c in wanted), default=99)
            scored.append((near, cell))
        scored.sort()
        out, used = [], set()
        for _near, cell in scored:
            if len(out) >= count:
                break
            if any(geo.distance(cell, c) <= 2 for c in used):
                continue
            used.add(cell)
            out.append(cell)
        return out

    def build_sdb(self, state, engine):
        """Lay down a wing at every starport still in Solomani hands."""
        return [c for c in sorted(state.geometry.starports)
                if c not in state.garrisoned]

    def deploy_sdb(self, state, ready, engine):
        """Send a finished wing back under water rather than into orbit."""
        geo = state.geometry
        taken = {u.location for u in state.naval.values()
                 if u.is_sdb and isinstance(u.location, int)}
        out = {}
        for cell, _cls in ready:
            for candidate in geo.within(cell, 3):
                if candidate in geo.deep_sea and candidate not in taken:
                    out[cell] = candidate
                    taken.add(candidate)
                    break
        return out

    def abandon_invasion(self, state, engine):
        """Never: it is the worst result on the table.

        There is a real case for it -- a player deep in replacement waves with
        no cities taken has already lost more than a major defeat costs -- but
        the arithmetic is the same either way, so the doctrine plays it out.
        """
        return False


class ScriptedAgent(HeuristicAgent):
    """The historical plan: go for the starports first.

    The Imperium's problem is that the Solomani build SDB wings at starports
    every quarter, so an invasion that ignores them fights a fleet that keeps
    growing.  Taking the three starports early costs tempo against the cities
    and is the obvious alternative doctrine to measure the open one against.
    """

    name = "scripted"

    def _landing_score(self, state, cell, field=None) -> float:
        """As the doctrine, but a starport in reach outweighs everything else.

        The bonus has to be large to mean anything.  A starport carries a
        planetary defence corps with a bombardment factor of nine, so the open
        doctrine's own scoring charges about twenty-seven points for landing
        inside its reach -- correctly, because that is what the -3 assault
        modifier costs.  This doctrine pays it deliberately: the Solomani lay
        down a new SDB wing at every ungarrisoned starport each quarter, and a
        fleet that never closes the ports fights a navy that keeps growing.
        """
        score = super()._landing_score(state, cell, field)
        if score <= -1e8:
            return score
        geo = state.geometry
        ports = sum(1 for c in geo.within(cell, 3)
                    if c in geo.starports and c not in state.garrisoned)
        return score + 30.0 * ports

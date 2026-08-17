"""The Imperial doctrine as it stood before the rewrite, kept to measure against.

This is not a doctrine anyone should play.  It is the one that was in
``heuristic.py`` when *Invasion: Earth* was first finished, and it is here so
that the claim "the rewrite improved the invasion" can be a measurement in this
engine rather than a memory of a number from an older commit -- the engine has
changed too, and comparing across both changes at once would mean comparing
nothing.

What it does, and what each of those cost, is written up in the README.  In
short: it lands on one hex chosen for being empty and far from the planetary
defences, which on this map is an island; it moves a unit only into one of the
six hexes next door, and only if that hex is a city; it lifts the army straight
into close orbit and parks the empty hulls there under the guns; and it takes
the beachhead's neighbourhood as the whole war.  It never garrisoned a city in
any game measured, and it left about two thirds of the army out-system.

Everything not overridden here comes from the current doctrine, which is what
makes the comparison a comparison of the parts that changed.
"""

from __future__ import annotations

from .. import oob, terra
from ..state import (BOMBARDMENT, CLOSE_ORBIT, DEEP_SPACE, FAR_ORBIT, IMPERIAL,
                     OUT_SYSTEM, OVERWATCH, SOLOMANI)
from .heuristic import HeuristicAgent


class BeachheadAgent(HeuristicAgent):
    """One beachhead, held, and the six hexes around it."""

    name = "beachhead"

    def begin_game(self, state, side, engine):
        super().begin_game(state, side, engine)
        self._beachhead = None

    def _imperial_naval(self, state):
        """Bring the fleet in: out-system, deep space, far orbit, close orbit.

        A jump can only be made from deep space, and only one a turn, so the
        squadron that arrives this turn cannot also close.  Loaded transports
        follow the warships rather than lead them.
        """
        moves = {}
        for unit in state.naval.values():
            if unit.side != IMPERIAL or unit.hidden:
                continue
            if unit.location == OUT_SYSTEM:
                # A hull with lift capacity waits until it is carrying
                # something.  Sending the transports out empty is the single
                # easiest way to lose this game: the whole army is out-system,
                # and a squadron in deep space cannot come back for it and
                # jump again in the same turn.
                if unit.cargo or unit.cls.capacity == 0 \
                        or not self._waiting_to_load(state):
                    moves[unit.uid] = DEEP_SPACE
            elif unit.location == DEEP_SPACE:
                # an empty hull in deep space with an army still waiting jumps
                # back for it rather than joining the queue over Terra
                if self._should_shuttle(state, unit):
                    moves[unit.uid] = OUT_SYSTEM
                else:
                    moves[unit.uid] = FAR_ORBIT
            elif unit.location == FAR_ORBIT:
                moves[unit.uid] = (DEEP_SPACE if self._should_shuttle(state, unit)
                                   else CLOSE_ORBIT)
            elif unit.location == CLOSE_ORBIT and self._should_shuttle(state, unit):
                moves[unit.uid] = FAR_ORBIT
        return moves

    def _should_shuttle(self, state, unit) -> bool:
        """Is this hull's next job another load?

        The Imperium can lift about 1600 combat factors at once and owns four
        thousand, so the invasion is three convoys, not one.  A transport that
        unloads and then parks in close orbit is the difference between landing
        forty percent of the army and landing all of it -- and it spends the
        rest of the game absorbing planetary defence fire for nothing.
        """
        if unit.cls.capacity <= 0 or unit.cargo:
            return False
        return self._waiting_to_load(state)

    def _waiting_to_load(self, state) -> bool:
        """Is there anything left out-system worth waiting for?

        Once the army has sailed, an empty hull sitting in the marshalling
        ground is doing nothing at all; it may as well be another target in
        orbit soaking planetary defence fire the warships would otherwise take.
        """
        for unit in state.surface.values():
            if unit.side == IMPERIAL and unit.carrier is None \
                    and unit.location == OUT_SYSTEM and not unit.dead:
                return True
        return any(b.carrier is None and b.location == OUT_SYSTEM
                   for b in state.bases.values())

    def missions(self, state, side, engine):
        if side != IMPERIAL:
            return {}
        orders = {}
        watchers = 0
        targets = self._bombardment_targets(state)
        for unit in state.naval_at(CLOSE_ORBIT, IMPERIAL):
            if unit.cls.bombard == 0:
                continue
            # keep some guns pointed at the water: a wing that surfaces
            # unopposed gets a free shot at the fleet
            if watchers < 3 and unit.cls.bombard >= 3:
                orders[unit.uid] = (OVERWATCH, None)
                watchers += 1
            elif targets and self.w["bombard_first"] > 0:
                orders[unit.uid] = (BOMBARDMENT, targets[len(orders) % len(targets)])
        return orders

    def _bombardment_targets(self, state):
        """Where the guns are worth pointing: at the enemy, not at the beach.

        Bombardment is the Imperium's one advantage that does not have to be
        shipped, and the surface bombardment table pays for concentration --
        fire on a hex is totalled before the column is read.  So the targets
        are the heaviest Solomani stacks that threaten the beachhead, with the
        beachhead's own defenders first if any are still standing.

        Guerrillas are skipped: "planetary bombardment attacks may not be made
        against guerrilla units", so a squadron sent at one wastes its turn.
        """
        geo = state.geometry
        beach = self._beachhead
        stacks = {}
        for unit in state.surface.values():
            if unit.side != SOLOMANI or unit.dead or unit.cls.guerrilla:
                continue
            if not isinstance(unit.location, int):
                continue
            stacks[unit.location] = stacks.get(unit.location, 0.0) + unit.current
        if not stacks:
            return []

        def worth(cell):
            value = stacks[cell]
            if beach is not None:
                value /= 1.0 + geo.distance(beach, cell)
            return value

        ranked = sorted(stacks, key=lambda c: -worth(c))
        return ranked[:3]

    def defence_fire(self, state, side, engine):
        """Point the planetary defences at whichever is worth more this turn."""
        if side != SOLOMANI:
            return {}
        orders = {}
        fleet = state.naval_at(CLOSE_ORBIT, IMPERIAL)
        fleet_worth = sum(u.cls.defense for u in fleet)
        for unit in state.surface.values():
            if unit.side != SOLOMANI or unit.dead or not unit.cls.planetary_defense:
                continue
            ashore = self._nearest_landed(state, unit.location)
            if ashore is None or fleet_worth * self.w["pd_vs_fleet"] > 8:
                if fleet:
                    orders[unit.uid] = ("naval", CLOSE_ORBIT)
            else:
                orders[unit.uid] = ("surface", ashore.uid)
        for wing in state.naval.values():
            if wing.is_sdb and not wing.hidden and wing.side == SOLOMANI:
                if fleet:
                    orders[wing.uid] = ("naval", CLOSE_ORBIT)
        return orders

    def landings(self, state, side, engine):
        if side != IMPERIAL:
            return []
        orders = []
        orders += self._load_out_system(state)
        orders += self._put_ashore(state)
        return orders

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
            key=lambda u: -u.current)
        bases = [b for b in state.bases.values()
                 if b.carrier is None and b.location == OUT_SYSTEM]
        hulls = sorted((u for u in state.naval.values()
                        if u.side == IMPERIAL and u.location == OUT_SYSTEM),
                       key=lambda u: -u.cls.capacity)
        if state.turn < oob.WITHDRAWAL_TURN:
            # Two transport squadrons are withdrawn on turn 2 and whatever is
            # aboard them goes too.  It is printed on the order of battle
            # chart, so there is no excuse for loading the army onto them.
            doomed = [u for u in hulls
                      if u.cls.kind == oob.WITHDRAWAL_KIND][:oob.WITHDRAWAL_COUNT]
            hulls = [u for u in hulls if u not in doomed]
        for carrier in hulls:
            room = carrier.cls.capacity - sum(
                self._cargo_cost(state, uid) for uid in carrier.cargo)
            if room <= 0:
                continue
            if bases and room >= oob.BASE_TRANSPORT_COST and \
                    self._bases_landed(state) < 3:
                base = bases.pop()
                orders.append(("load", base.uid, carrier.uid))
                room -= oob.BASE_TRANSPORT_COST
            for unit in list(waiting):
                if unit.current <= room:
                    orders.append(("load", unit.uid, carrier.uid))
                    room -= unit.current
                    waiting.remove(unit)
                if room <= 0:
                    break
        return orders

    def _cargo_cost(self, state, uid):
        if uid in state.bases:
            return oob.BASE_TRANSPORT_COST
        unit = state.surface.get(uid)
        return unit.current if unit else 0.0

    def _bases_landed(self, state) -> int:
        return sum(1 for b in state.bases.values()
                   if b.carrier is None and isinstance(b.location, int))

    def _put_ashore(self, state):
        """Build a beachhead, then feed it.

        Landing straight onto a defended city is how this invasion dies, and
        the rules say so twice.  A unit landing in or leaving a hex is fired on
        by every planetary defence within three hexes, at a **-3** on the
        surface bombardment table unless it is a marine or jump troop unit --
        and a corps that survives that arrives in a hex holding a 500-factor
        field army, having already spent the turn it landed.

        So the beachhead is chosen for being *empty*, the assault troops go
        first because they are the ones the -3 does not apply to, and the rest
        follows onto ground that is already held.
        """
        target = self._landing_zone(state)
        if target is None:
            return []
        landed = self._current_strength_at(state, target)
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

        orders = []
        for uid in sorted(cargo, key=priority):
            unit = state.surface.get(uid)
            cost = unit.current if unit is not None else oob.BASE_TRANSPORT_COST
            if landed + cost > oob.STACKING_LIMIT:
                continue                 # "no more than 1000 combat factors"
            orders.append(("unload", uid, target))
            landed += cost
        return orders

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

    def _current_strength_at(self, state, cell) -> float:
        return sum(u.current for u in state.surface_at(cell, IMPERIAL))

    def _landing_zone(self, state):
        """Score every land hex and keep the best, sticky across turns.

        Switching beachheads every turn is how an invasion loses: supply is
        traced to Imperial bases, a base cannot move once landed, and a unit
        out of supply may not fire and defends at half strength.  So the choice
        is held until the ground is no longer ours.
        """
        geo = state.geometry
        held = self._beachhead
        if held is not None and not state.surface_at(held, SOLOMANI) \
                and self._threat(state, held) < 400:
            return held
        best, best_score = None, -1e9
        for cell in sorted(geo.land):
            score = self._landing_score(state, cell)
            if score > best_score:
                best, best_score = cell, score
        self._beachhead = best
        return best

    def _landing_score(self, state, cell) -> float:
        """What makes a hex worth landing on.

        Not its own value -- an empty field scores nothing by itself.  What is
        being bought is a place to stand: clear of defenders, out from under
        the planetary defences, and within a march of as many cities as
        possible.
        """
        geo, w = state.geometry, self.w
        if geo.terrain[cell] in (terra.PERMANENT_ICE, terra.SEASONAL_ICE):
            return -1e9
        defenders = sum(u.current for u in state.surface_at(cell, SOLOMANI))
        score = w["defended"] * defenders / 20.0
        # and what can reach it next turn, which on this map is nearly
        # everything: grav troops have ten movement points and a hex is
        # 1100 km, so the only real cover is distance from the mass
        score += w["defended"] * self._threat(state, cell) / 200.0
        nearby = sum(1 for c in geo.within(cell, 3)
                     if c in geo.urban and c not in state.garrisoned)
        score += w["urban_value"] * nearby
        ports = sum(1 for c in geo.within(cell, 3) if c in geo.starports)
        score += w["starport_value"] * ports
        # every planetary defence within three hexes fires on everything
        # landing here and on everything leaving, so it is a reason not to be
        guns = 0.0
        for unit in state.surface.values():
            if unit.side != SOLOMANI or unit.dead or not unit.cls.planetary_defense:
                continue
            if isinstance(unit.location, int) \
                    and geo.distance(unit.location, cell) <= oob.PD_RANGE:
                guns += unit.cls.bombard
        # This is the heaviest term in the whole score, and deliberately.  A
        # unit landing within three hexes of the planetary defences is fired on
        # at -3 on the surface bombardment table before it has done anything,
        # which at any real strength is thirty to fifty percent losses on
        # arrival.  Nothing a beachhead can be worth pays for that.
        score -= 3.0 * guns
        if state.surface_at(cell, IMPERIAL):
            score += w["concentration"] * 3.0
        for base in state.bases.values():
            if isinstance(base.location, int) and \
                    geo.distance(base.location, cell) <= oob.SUPPLY_RANGE:
                score += w["supply_reach"] * 2.0
                break
        if geo.terrain[cell] in terra.GUERRILLA_COVER:
            score += w["guerrilla_cover"]
        return score

    # ==================================================================
    #  surface
    # ==================================================================

    def _imperial_surface(self, state, engine=None):
        """Expand from the beachhead one hex at a time, and never alone.

        The temptation is to send every corps at a different city: there are
        sixty-one of them and the game is won by garrisoning fifty-two.  It
        loses every time.  Solomani troops are grav-mobile with ten movement
        points, which on this map is a third of the way round the planet, so
        they can mass on any single Imperial stack at any moment -- and an
        Imperial unit more than five hexes from a base is out of supply, which
        means it may not fire and defends at half strength.

        So a unit only moves when the stack it is leaving can spare it and the
        hex it is going to is next door, and a hex is only taken when what is
        going there beats what is already sitting on it.  Garrisoning is by
        zone of control, so one corps holding the right hex covers its own and
        six more.
        """
        geo, w = state.geometry, self.w
        moves = {}
        units = [u for u in state.surface.values()
                 if u.side == IMPERIAL and not u.dead and u.carrier is None
                 and u.cls.mobile and isinstance(u.location, int)
                 and u.landed_turn != state.turn]
        if not units:
            return moves
        by_hex: dict[int, list] = {}
        for unit in units:
            by_hex.setdefault(unit.location, []).append(unit)

        claimed = set()
        for here, stack in sorted(by_hex.items()):
            strength = sum(u.combat_strength() for u in stack)
            options = []
            for cell in geo.adjacent(here):
                if cell not in geo.land or cell in claimed:
                    continue
                if cell not in geo.urban and cell not in geo.starports:
                    continue
                if cell in state.garrisoned:
                    continue
                opposition = sum(u.defence_strength()
                                 for u in state.surface_at(cell, SOLOMANI))
                options.append((opposition, cell))
            if not options:
                continue
            options.sort()
            opposition, target = options[0]
            # send the smallest force that still wins the odds, and only if
            # what stays behind can hold the ground it is standing on
            need = max(opposition * 2.0, 1.0)
            going, sent = [], 0.0
            for unit in sorted(stack, key=lambda u: -u.combat_strength()):
                if sent >= need:
                    break
                going.append(unit)
                sent += unit.combat_strength()
            if sent < need or sent > strength * w["concentration"] + need:
                continue
            claimed.add(target)
            for unit in going:
                moves[unit.uid] = target
        return moves

    def _solomani_surface(self, state):
        """Move towards whatever has come ashore, weighted by how much of it."""
        geo = state.geometry
        landings = [c for c in geo.land if state.surface_at(c, IMPERIAL)]
        if not landings or self.w["reinforce_front"] <= 0:
            return {}
        moves = {}
        for unit in state.surface.values():
            if unit.side != SOLOMANI or unit.dead or unit.carrier is not None:
                continue
            if not unit.cls.mobile or not isinstance(unit.location, int):
                continue
            if unit.cls.guerrilla:
                continue
            here = unit.location
            target = min(landings, key=lambda c: (geo.distance(here, c), c))
            d = geo.distance(here, target)
            if 0 < d <= 6:
                moves[unit.uid] = target
        return moves

    def replacement_waves(self, state, engine):
        """Buy replacements only while they can still buy the war.

        Every wave is a victory point, and the table only has ten to give, so a
        player who buys six waves has already given away a decisive victory.
        """
        if self.side != IMPERIAL and self.side is not None:
            pass
        losses = len(state.pools.get("dead_surface", []))
        appetite = self.w["wave_appetite"]
        budget = int(max(0.0, appetite * 8))
        if state.waves_taken >= budget:
            return 0
        return 1 if losses >= 4 else 0

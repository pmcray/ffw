"""The *Invasion: Earth* rules engine: setup, sequence of play, combat.

Rule 2 is emphatic that the order is the game: "the sequence of play strictly
defines when the various game activities may be performed.  No activity may be
performed outside this sequence."  So ``Engine.play_turn`` is written as the
sequence, one method per phase, in the printed order, and every decision inside
a phase is asked of the agent rather than taken by the engine.
"""

from __future__ import annotations

import random

from . import hexmap, oob, tables, terra
from .state import (BOMBARDMENT, CLOSE_ORBIT, DEEP_SPACE, FAR_ORBIT, IMPERIAL,
                    IN_SYSTEM, LUNA, OUT_SYSTEM, OVERWATCH, SOLOMANI,
                    TURNS_PER_QUARTER, VICTORY_URBAN_THRESHOLD, Base, Geometry,
                    GameState, NavalUnit, SurfaceUnit, enemy_of)

#: The campaign ran "from the second quarter of 1002 ... until nearly the end
#: of the year", which is three quarters.  Play does not stop there -- the
#: rules run until a victory check ends it -- but a runaway game has to stop
#: somewhere, and eight quarters is well past the point where the question is
#: settled.
MAX_TURNS = 48

MAX_COMBAT_ROUNDS = 12


# ==========================================================================
#  setup
# ==========================================================================
def new_game(seed: int | None = None, options: dict | None = None) -> GameState:
    """Deploy both sides per the order of battle chart."""
    rng = random.Random(seed)
    state = GameState()
    if options:
        state.options.update(options)
    state.rng = rng                                   # type: ignore[attr-defined]

    world = terra.load()
    state.geometry = Geometry(world["terrain"])
    state.starport_names = world["starports"]

    _deploy_solomani(state, rng)
    _deploy_imperial(state, rng)
    state.note("game set up; %d naval and %d surface units in play"
               % (len(state.naval), len(state.surface)))
    return state


def _add_naval(state, cls, navy, where, hidden=False, dummy=False) -> NavalUnit:
    unit = NavalUnit(uid=state.new_uid(), cls=cls, navy=navy, location=where,
                     hidden=hidden, dummy=dummy)
    state.naval[unit.uid] = unit
    return unit


def _add_surface(state, cls, navy, where) -> SurfaceUnit:
    unit = SurfaceUnit(uid=state.new_uid(), cls=cls, navy=navy, location=where)
    state.surface[unit.uid] = unit
    return unit


def _deploy_solomani(state: GameState, rng: random.Random) -> None:
    """The deployment paragraph of the order of battle chart, line by line.

    "Naval units are placed in far orbit or in near orbit.  One PD corps is
    placed at each starport.  All PD divisions are placed in urban hexes, no
    more than one per hex.  One army is placed in any urban hex on each of the
    following continents: Africa, Asia, North America, and South America.  All
    corps are placed in urban hexes or starports, with no more than one
    army-sized or corps-sized troop unit per hex.  All remaining troop and PD
    units may be placed in any land hexes."
    """
    geo = state.geometry
    urban = sorted(geo.urban)
    starports = sorted(geo.starports)
    rng.shuffle(urban)

    for cls in oob.expand(oob.SOLOMANI_NAVAL):
        _add_naval(state, cls, SOLOMANI,
                   FAR_ORBIT if rng.random() < 0.5 else CLOSE_ORBIT)
    # The SDB wings start hidden in the oceans: that is the whole Solomani
    # plan, and the rules let a wing be placed in any deep sea hex.
    deep = sorted(geo.deep_sea)
    rng.shuffle(deep)
    for i, cls in enumerate(oob.expand(oob.SOLOMANI_SDB)):
        if i < len(deep):
            _add_naval(state, cls, SOLOMANI, deep[i], hidden=True)
        else:                                          # pragma: no cover
            _add_naval(state, cls, SOLOMANI, CLOSE_ORBIT)
    for i in range(oob.DUMMY_SDB):
        cell = deep[len(oob.expand(oob.SOLOMANI_SDB)) + i]
        _add_naval(state, oob.SOLOMANI_SDB[0][0], SOLOMANI, cell,
                   hidden=True, dummy=True)

    pd = oob.expand(oob.SOLOMANI_PD)
    corps = [c for c in pd if c.size == "corps"]
    divisions = [c for c in pd if c.size == "division"]
    rest = [c for c in pd if c.size not in ("corps", "division")]
    for cls, cell in zip(corps, starports):
        _add_surface(state, cls, SOLOMANI, cell)
    for cls, cell in zip(divisions, urban):
        _add_surface(state, cls, SOLOMANI, cell)

    troops = oob.expand(oob.SOLOMANI_TROOPS)
    armies = [c for c in troops if c.size == "army"]
    troop_corps = [c for c in troops if c.size == "corps"]
    others = [c for c in troops if c.size not in ("army", "corps")] + rest

    taken: set = set()
    for cls, continent in zip(armies, ("Africa", "Asia", "North America",
                                       "South America")):
        home = [c for c in urban if c not in taken
                and terra.continent(*geo.lat_lon[c]) == continent]
        cell = home[0] if home else next(c for c in urban if c not in taken)
        taken.add(cell)
        _add_surface(state, cls, SOLOMANI, cell)
    for cls in troop_corps:
        free = [c for c in urban + starports if c not in taken]
        if not free:                                   # pragma: no cover
            free = urban
        cell = free[0]
        taken.add(cell)
        _add_surface(state, cls, SOLOMANI, cell)

    land = sorted(geo.land)
    rng.shuffle(land)
    for i, cls in enumerate(others):
        _add_surface(state, cls, SOLOMANI, land[i % len(land)])


def _deploy_imperial(state: GameState, rng: random.Random) -> None:
    """"All forces are placed in the out-system box."

    The whole invasion force starts off the board, which is what makes the
    Imperial problem a shipping problem before it is a fighting one.
    """
    for cls in oob.expand(oob.IMPERIAL_NAVAL):
        _add_naval(state, cls, IMPERIAL, OUT_SYSTEM)
    for spec in (oob.IMPERIAL_TROOPS, oob.IMPERIAL_COLONIAL,
                 oob.IMPERIAL_MARINES, oob.IMPERIAL_MERCENARIES):
        for cls in oob.expand(spec):
            _add_surface(state, cls, IMPERIAL, OUT_SYSTEM)
    for _ in range(oob.IMPERIAL_BASES):
        base = Base(uid=state.new_uid(), location=OUT_SYSTEM)
        state.bases[base.uid] = base
    state.pools["guerrillas"] = oob.expand(oob.SOLOMANI_GUERRILLAS)
    state.pools["dead_imperial_naval"] = []
    state.pools["dead_surface"] = []
    state.pools["sdb_building"] = {}


# ==========================================================================
#  the engine
# ==========================================================================
class Engine:
    def __init__(self, state: GameState, imperial, solomani,
                 rng: random.Random | None = None):
        self.state = state
        self.agents = {IMPERIAL: imperial, SOLOMANI: solomani}
        self.rng = rng or getattr(state, "rng", None) or random.Random(0)
        for side, agent in self.agents.items():
            agent.begin_game(state, side, self)

    # -- dice ----------------------------------------------------------
    def d6(self) -> int:
        return self.rng.randint(1, 6)

    def d66(self) -> int:
        return self.rng.randint(1, 6) + self.rng.randint(1, 6)

    # -- the turn ------------------------------------------------------
    def play_turn(self) -> None:
        s = self.state
        if s.game_over:
            return
        for side in (IMPERIAL, SOLOMANI):
            self.agents[side].begin_turn(s, side, self)
        # Rule 4 halves a bombarding unit's factor "if it did not start the
        # turn in the close orbit or if it moved to any other space box at any
        # time during its naval phases in the turn", so the flag is a property
        # of this turn and has to be reset at the top of it.
        for unit in s.naval.values():
            unit.manoeuvred = unit.location != CLOSE_ORBIT
            unit.jumped = False
        self.initial_segment()
        self.space_segment()
        self.space_surface_segment()
        self.surface_segment()
        if s.is_quarter_end():
            self.quarterly_special_turn()
        s.turn += 1
        if s.turn > MAX_TURNS and not s.game_over:
            self._end("ran out of turns")

    # ---- 1. initial segment ------------------------------------------
    def initial_segment(self) -> None:
        s = self.state
        for side in (IMPERIAL, SOLOMANI):
            self._consolidate(side)
            self._emergency_replacements(side)
        self._withdrawals()
        # "Solomani surface unit replacements are accumulated each turn during
        # the initial segment ... one replacement point from each ungarrisoned
        # urban hex."
        s.replacement_points[SOLOMANI] += len(s.solomani_urban())

    def _consolidate(self, side: str) -> None:
        s = self.state
        for donor_uid, target_uid in self.agents[side].consolidate(s, side, self):
            donor, target = s.surface.get(donor_uid), s.surface.get(target_uid)
            if donor is None or target is None or donor is target:
                continue
            if donor.side != side or target.side != side:
                continue
            if donor.location != target.location or donor.dead or target.dead:
                continue
            if donor.cls.guerrilla or donor.cls.planetary_defense:
                continue
            # the four restrictions on consolidation, in the order printed
            if target.cls.elite and not donor.cls.elite:
                continue
            if target.cls.armor and not donor.cls.armor:
                continue
            if donor.cls.tech < target.cls.tech:
                continue
            for special in ("jump", "marine", "commando"):
                if getattr(target.cls, special) != getattr(donor.cls, special):
                    break
            else:
                room = target.cls.factor - target.current
                given = min(donor.current, room)
                target.losses = max(0, int(round(
                    100 * (1 - (target.current + given) / target.cls.factor))))
                donor.losses = 100

    def _emergency_replacements(self, side: str) -> None:
        s = self.state
        want = self.agents[side].emergency_replacements(s, side, self)
        if want <= 0:
            return
        if side == SOLOMANI:
            # "two replacement points for each combat factor replaced"
            self._rebuild_surface(SOLOMANI, s.replacement_points[SOLOMANI], 2.0)
        else:
            s.waves_taken += want
            self._rebuild_surface(IMPERIAL, want * 50, 1.0)

    def _withdrawals(self) -> None:
        """Rule 6: two transport squadrons leave on turn 2, whatever it costs."""
        s = self.state
        if s.turn != oob.WITHDRAWAL_TURN:
            return
        gone = 0
        # The player chooses which two, and an empty hull is the obvious
        # choice: anything aboard a withdrawn squadron leaves with it.
        candidates = sorted(
            (u for u in s.naval.values()
             if u.side == IMPERIAL and u.cls.kind == oob.WITHDRAWAL_KIND),
            key=lambda u: (len(u.cargo), u.uid))
        for unit in candidates:
            if gone >= oob.WITHDRAWAL_COUNT:
                break
            self._destroy_naval(unit, withdrawn=True)
            gone += 1
        if gone:
            s.note("%d transport squadrons withdrawn" % gone)

    # ---- 2. space segment --------------------------------------------
    def space_segment(self) -> None:
        for side in (IMPERIAL, SOLOMANI):
            for _ in range(4):          # a bounded "may elect to take another"
                fought = self.naval_phase(side)
                if not fought:
                    break
                if not self.agents[side].take_another_naval_phase(
                        self.state, side, self):
                    break

    def naval_phase(self, side: str) -> bool:
        """Move, then fight.  Returns whether a space battle was fought."""
        s = self.state
        agent = self.agents[side]

        hiding = agent.hide_in_deep_space(s, side, self)
        for unit in s.naval.values():
            if unit.side != side or unit.is_sdb:
                continue
            if unit.location == DEEP_SPACE:
                unit.hidden = unit.uid in hiding
            unit.jumped = False

        for uid, destination in agent.naval_moves(s, side, self).items():
            unit = s.naval.get(uid)
            if unit is None or unit.side != side or unit.hidden:
                continue
            self._move_naval(unit, destination)

        fought = False
        for box in IN_SYSTEM:
            if self._space_battle(box):
                fought = True
        return fought

    def _move_naval(self, unit: NavalUnit, destination) -> bool:
        """One naval move, refusing anything the rules do not allow."""
        s = self.state
        origin = unit.location
        if not isinstance(origin, str) or origin == LUNA:
            return False
        if destination == origin:
            return True
        if unit.is_sdb:
            return False               # SDB wings move in their own phase
        if destination == OUT_SYSTEM or origin == OUT_SYSTEM:
            # "only a naval unit capable of making an interstellar jump may
            # move between the Sol system space boxes and the out-system box",
            # once a turn, and never from close or far orbit
            if not unit.cls.jump or unit.jumped:
                return False
            if origin in (CLOSE_ORBIT, FAR_ORBIT):
                return False
            if destination not in (OUT_SYSTEM, DEEP_SPACE):
                return False
            unit.jumped = True
            unit.location = destination
            unit.manoeuvred = True
            return True
        if destination not in IN_SYSTEM:
            return False
        # in-system movement is along the line, and stops on contact
        path = IN_SYSTEM[min(IN_SYSTEM.index(origin), IN_SYSTEM.index(destination)):
                         max(IN_SYSTEM.index(origin), IN_SYSTEM.index(destination)) + 1]
        if IN_SYSTEM.index(destination) < IN_SYSTEM.index(origin):
            path = list(reversed(path))
        for box in path[1:]:
            unit.location = box
            if box != CLOSE_ORBIT:
                unit.manoeuvred = True
            if self._enemy_present(box, unit.side):
                break                  # "must cease its in-system movement
                                       # immediately upon entering a space box
                                       # containing enemy naval units"
        if origin == CLOSE_ORBIT or unit.location != CLOSE_ORBIT:
            unit.manoeuvred = True
        return True

    def _enemy_present(self, box: str, side: str) -> bool:
        return any(u.side != side and not u.hidden
                   for u in self.state.naval_at(box))

    # -- space battles -------------------------------------------------
    def _space_battle(self, box: str) -> bool:
        s = self.state
        imp = [u for u in s.naval_at(box, IMPERIAL)]
        sol = [u for u in s.naval_at(box, SOLOMANI)]
        if not imp or not sol:
            return False
        for _round in range(MAX_COMBAT_ROUNDS):
            imp = [u for u in s.naval_at(box, IMPERIAL)]
            sol = [u for u in s.naval_at(box, SOLOMANI)]
            if not imp or not sol:
                break
            self._combat_round(box, imp, sol)
            imp = [u for u in s.naval_at(box, IMPERIAL)]
            sol = [u for u in s.naval_at(box, SOLOMANI)]
            if not imp or not sol:
                break
            # "At the end of each round of a space battle, the Solomani player
            # may disengage from the battle.  If the Solomani player does not
            # disengage, the Imperial player may choose to do so."
            for side in (SOLOMANI, IMPERIAL):
                if self.agents[side].disengage(s, side, box, self):
                    self._disengage(box, side)
                    return True
        return True

    def _combat_round(self, box: str, imp: list, sol: list) -> None:
        """One simultaneous exchange, with the Solomani split into two groups.

        "At the start of each round, the players must divide their naval units
        into two groups each if the Solomani player has both squadrons and SDB
        wings present in the box.  For the Solomani player, all squadrons must
        be placed in one group and all SDB wings in the other."
        """
        sol_squadrons = [u for u in sol if not u.is_sdb]
        sol_sdb = [u for u in sol if u.is_sdb]

        damage_to_sol_squadrons = damage_to_sol_sdb = damage_to_imp = 0

        if sol_squadrons and sol_sdb:
            # the Imperial player splits; a simple, legal split is to send the
            # heavier half at the squadrons, which are what can shoot back hard
            half = max(1, len(imp) // 2)
            against_squadrons, against_sdb = imp[:half], imp[half:]
        elif sol_sdb:
            against_squadrons, against_sdb = [], imp
        else:
            against_squadrons, against_sdb = imp, []

        if against_squadrons and sol_squadrons:
            strength = sum(u.cls.attack for u in against_squadrons)
            damage_to_sol_squadrons = tables.space_attack(strength, self.d6())
        if against_sdb and sol_sdb:
            strength = sum(u.bombard_factor() for u in against_sdb)
            damage_to_sol_sdb = tables.space_bombard(strength, self.d6())

        if sol_squadrons:
            damage_to_imp += tables.space_attack(
                sum(u.cls.attack for u in sol_squadrons), self.d6())
        if sol_sdb:
            damage_to_imp += tables.space_bombard(
                sum(u.bombard_factor() for u in sol_sdb), self.d6())

        self._apply_damage(sol_squadrons, damage_to_sol_squadrons)
        self._apply_damage(sol_sdb, damage_to_sol_sdb)
        self._apply_damage(imp, damage_to_imp)

    def _apply_damage(self, units: list, damage: int) -> None:
        """Eliminate units until the defence factors removed cover the damage.

        "The battle damage taken by the naval units is the (minimum) number of
        defense factors that must be eliminated by eliminating some or all of
        the affected naval units.  The owning player chooses which naval units
        are eliminated."  The cheapest legal choice -- weakest first -- is the
        one taken here.
        """
        if damage <= 0 or not units:
            return
        removed = 0
        for unit in sorted(units, key=lambda u: (u.cls.defense, u.uid)):
            if removed >= damage:
                break
            removed += unit.cls.defense
            self._destroy_naval(unit)

    def _disengage(self, box: str, side: str) -> None:
        """Break off, paying whatever the box charges for it."""
        s = self.state
        leaving = s.naval_at(box, side)
        if box == CLOSE_ORBIT:
            # "the disengaging naval units are first placed in the far orbit
            # box, where they may be fired upon by any enemy naval units in the
            # box.  This fire is conducted in the same manner as a round of
            # space combat, except that the disengaging naval units may not fire."
            watchers = [u for u in s.naval_at(FAR_ORBIT)
                        if u.side != side and not u.hidden]
            if watchers:
                strength = sum(u.cls.attack for u in watchers)
                self._apply_damage(leaving, tables.space_attack(strength, self.d6()))
                leaving = [u for u in leaving if u.uid in s.naval]
        for unit in leaving:
            if unit.is_sdb and not self._imperial_in_close_orbit():
                cell = self._free_deep_sea()
                if cell is not None:
                    unit.location, unit.hidden = cell, True
                    continue
            unit.location, unit.hidden = DEEP_SPACE, True
        s.note("%s disengages from %s" % (side, box))

    def _imperial_in_close_orbit(self) -> bool:
        return bool(self.state.naval_at(CLOSE_ORBIT, IMPERIAL))

    def _free_deep_sea(self):
        s = self.state
        taken = {u.location for u in s.naval.values()
                 if u.is_sdb and isinstance(u.location, int)}
        for cell in sorted(s.geometry.deep_sea):
            if cell not in taken:
                return cell
        return None                                    # pragma: no cover

    def _destroy_naval(self, unit: NavalUnit, withdrawn: bool = False) -> None:
        s = self.state
        for uid in list(unit.cargo):
            carried = s.surface.get(uid)
            if carried is not None:
                carried.losses = 100                   # "any surface unit it is
                carried.carrier = None                 # transporting is also
            base = s.bases.get(uid)                    # eliminated"
            if base is not None:
                base.losses = 100
                base.carrier = None
        for other in s.surface.values():
            if other.carrier == unit.uid:
                other.losses = 100
                other.carrier = None
        for base in s.bases.values():
            if base.carrier == unit.uid:
                base.losses = 100
                base.carrier = None
        del s.naval[unit.uid]
        if not withdrawn and unit.side == IMPERIAL:
            s.pools["dead_imperial_naval"].append(unit.cls)

    # ---- 3. space-surface segment ------------------------------------
    def space_surface_segment(self) -> None:
        self.mission_allocation_phase()
        surfaced = self.sdb_activity_phase()
        self.overwatch_combat_phase(surfaced)
        self.bombardment_phase(surfaced)
        self.landing_phase()
        self.restoration_phase(surfaced)

    def mission_allocation_phase(self) -> None:
        s = self.state
        for side in (IMPERIAL, SOLOMANI):
            for uid, order in self.agents[side].missions(s, side, self).items():
                unit = s.naval.get(uid)
                if unit is None or unit.side != side:
                    continue
                if unit.location != CLOSE_ORBIT or unit.cls.bombard == 0:
                    continue     # "a naval unit with a bombardment factor of 0
                mission, target = order                # may not be assigned"
                if mission == BOMBARDMENT and isinstance(target, int):
                    unit.mission = BOMBARDMENT
                    unit.location = target
                elif mission == OVERWATCH:
                    unit.mission = OVERWATCH

    def sdb_activity_phase(self) -> list:
        s = self.state
        agent = self.agents[SOLOMANI]
        coming_up = agent.sdb_surface(s, SOLOMANI, self)
        surfaced = []
        for unit in list(s.naval.values()):
            if not unit.is_sdb or not unit.hidden:
                continue
            if unit.uid in coming_up:
                unit.hidden = False
                surfaced.append(unit)
        taken = {u.location for u in s.naval.values()
                 if u.is_sdb and u.hidden and isinstance(u.location, int)}
        for uid, cell in agent.sdb_relocate(s, SOLOMANI, self).items():
            unit = s.naval.get(uid)
            if unit is None or not unit.is_sdb or not unit.hidden:
                continue
            if cell in s.geometry.deep_sea and cell not in taken \
                    and hexmap.distance(unit.location, cell) == 1:
                taken.discard(unit.location)
                unit.location = cell
                taken.add(cell)
        return surfaced

    def overwatch_combat_phase(self, surfaced: list) -> None:
        """Overwatch fire, one surfaced wing at a time, both sides firing."""
        s = self.state
        watchers = [u for u in s.naval_at(CLOSE_ORBIT, IMPERIAL)
                    if u.mission == OVERWATCH]
        if not watchers:
            return
        order = self.agents[IMPERIAL].overwatch_targets(s, IMPERIAL, surfaced, self)
        for wing in order:
            if wing.uid not in s.naval:
                continue
            watchers = [u for u in watchers if u.uid in s.naval]
            if not watchers:
                break
            strength = sum(u.bombard_factor() for u in watchers)
            damage = tables.space_bombard(strength, self.d6())
            reply = tables.space_bombard(wing.bombard_factor(), self.d6())
            self._apply_damage([wing], damage)
            self._apply_damage(watchers, reply)

    def bombardment_phase(self, surfaced: list) -> None:
        """Planetary bombardment and planetary defence fire, simultaneously.

        Two rules shape this phase and both are about *totalling*:

            Each surface unit must be bombarded separately; two or more surface
            units may not be bombarded together.  Total the bombardment factors
            of all units bombarding a given surface unit to determine the
            column used on the table.

            Naval units are not attacked individually.  Instead, all naval
            units in the close orbit box defend as a single group, and all
            naval units in a hex on the map defend as a single group ... by
            totalling the bombardment factors of all PD units and SDB wings
            firing upon a group of naval units.

        So fire is pooled per target and resolved once, not once per firer.
        Resolving each firer separately would be a different and far deadlier
        game: the twenty-four Solomani planetary defence units would make
        two dozen attacks a turn on the fleet instead of one.

        "All activity in this phase is considered simultaneous; the combat
        results achieved during this phase are implemented at the end of the
        phase", so everything is gathered first and applied after.
        """
        s = self.state
        against_surface: dict[int, float] = {}
        against_naval: dict[object, float] = {}

        # Imperial squadrons on the planetary bombardment mission
        for unit in list(s.naval.values()):
            if unit.mission != BOMBARDMENT or not isinstance(unit.location, int):
                continue
            targets = [u for u in s.surface_at(unit.location)
                       if u.side != unit.side and not u.cls.guerrilla]
            if not targets:
                continue     # "planetary bombardment attacks may not be made
                             # against guerrilla units"
            target = self.agents[unit.side].bombardment_target(
                s, unit.side, unit.location, targets, self)
            if target is None or target not in targets:
                continue
            against_surface[target.uid] = (against_surface.get(target.uid, 0.0)
                                           + unit.bombard_factor())

        # PD units and surfaced SDB wings
        for uid, order in self.agents[SOLOMANI].defence_fire(
                s, SOLOMANI, self).items():
            kind, target = order
            firer = s.surface.get(uid) or s.naval.get(uid)
            if firer is None:
                continue
            strength = self._defence_strength(firer)
            if strength <= 0:
                continue
            if kind == "naval":
                group = self._naval_group(target)
                if not group:
                    continue
                # "except that the bombardment factors of the firing PD units
                # and SDB wings are halved" when the target is in close orbit
                if target == CLOSE_ORBIT:
                    strength /= 2.0
                against_naval[target] = against_naval.get(target, 0.0) + strength
            else:
                victim = s.surface.get(target)
                if victim is None or victim.side == firer.side or victim.dead:
                    continue
                if not self._in_pd_range(firer, victim.location):
                    continue
                against_surface[victim.uid] = (
                    against_surface.get(victim.uid, 0.0) + strength)

        results: dict[int, int] = {}
        for uid, strength in against_surface.items():
            victim = s.surface.get(uid)
            if victim is None:
                continue
            roll = self.d6() + tables.TECH_MODIFIER.get(victim.cls.tech, -1)
            results[uid] = tables.surface_bombard(strength, roll)
        damage: list = []
        for where, strength in against_naval.items():
            group = self._naval_group(where)
            damage.append((group, tables.space_bombard(strength, self.d6())))

        for uid, loss in results.items():
            unit = s.surface.get(uid)
            if unit is not None:
                unit.losses = min(100, unit.losses + loss)
        for group, dealt in damage:
            self._apply_damage([u for u in group if u.uid in s.naval], dealt)

    def _defence_strength(self, firer) -> float:
        if isinstance(firer, NavalUnit):
            return firer.bombard_factor() if not firer.hidden else 0.0
        if not firer.cls.planetary_defense or firer.dead:
            return 0.0
        # "the PD unit's bombardment factor always remains at full strength
        # until the PD unit is entirely eliminated"
        return float(firer.cls.bombard)

    def _in_pd_range(self, firer, cell) -> bool:
        origin = firer.location
        if not isinstance(origin, int) or not isinstance(cell, int):
            return False
        return hexmap.distance(origin, cell) <= oob.PD_RANGE

    def _naval_group(self, where) -> list:
        s = self.state
        if where == CLOSE_ORBIT:
            return s.naval_at(CLOSE_ORBIT, IMPERIAL)
        return [u for u in s.naval.values()
                if u.location == where and u.side == IMPERIAL]

    def landing_phase(self) -> None:
        s = self.state
        for side in (IMPERIAL, SOLOMANI):
            for order in self.agents[side].landings(s, side, self):
                verb, uid, where = order
                if verb == "unload":
                    self._unload(side, uid, where)
                elif verb == "load":
                    self._load(side, uid, where)

    def _unload(self, side: str, uid: int, cell) -> None:
        s = self.state
        unit = s.surface.get(uid) or s.bases.get(uid)
        if unit is None or unit.side != side or unit.carrier is None:
            return
        carrier = s.naval.get(unit.carrier)
        if carrier is None:
            return
        if carrier.location == CLOSE_ORBIT:
            if not isinstance(cell, int) or cell not in s.geometry.land:
                return          # "placed in any hex on Terra except for an
                                # all-sea hex"
            if isinstance(unit, Base) and s.geometry.terrain[cell] in (
                    terra.TUNDRA, terra.PERMANENT_ICE, terra.SEASONAL_ICE):
                return          # a base may not land on ice or tundra
            if isinstance(unit, SurfaceUnit) and \
                    self._stacking(side).get(cell, 0.0) + unit.current \
                    > oob.STACKING_LIMIT:
                return          # the stacking limit applies to a hex however
                                # the unit got there
        elif carrier.location == FAR_ORBIT:
            if cell != LUNA:
                return          # "naval units may only load from or unload
                                # onto the surface of Luna"
        elif carrier.location == OUT_SYSTEM:
            cell = OUT_SYSTEM
        else:
            return              # "for the deep space box, no loading or
                                # unloading may occur"
        carrier.cargo = [c for c in carrier.cargo if c != uid]
        unit.carrier = None
        unit.location = cell
        if isinstance(unit, SurfaceUnit):
            unit.landed_turn = s.turn
            self._assault_fire(unit, cell)

    def _load(self, side: str, uid: int, carrier_uid: int) -> None:
        s = self.state
        unit = s.surface.get(uid) or s.bases.get(uid)
        carrier = s.naval.get(carrier_uid)
        if unit is None or carrier is None or unit.carrier is not None:
            return
        if unit.side != side or carrier.side != side:
            return
        cost = (oob.BASE_TRANSPORT_COST if isinstance(unit, Base)
                else unit.current)
        used = sum(self._cargo_cost(c) for c in carrier.cargo)
        if used + cost > carrier.cls.capacity:
            return
        if carrier.location == CLOSE_ORBIT:
            if not isinstance(unit.location, int):
                return
            self._assault_fire(unit, unit.location)
            if getattr(unit, "dead", False):
                return
        elif carrier.location == FAR_ORBIT:
            if unit.location != LUNA:
                return
        elif carrier.location != OUT_SYSTEM or unit.location != OUT_SYSTEM:
            return
        carrier.cargo.append(uid)
        unit.carrier = carrier.uid
        unit.location = carrier.location

    def _cargo_cost(self, uid: int) -> float:
        s = self.state
        if uid in s.bases:
            return oob.BASE_TRANSPORT_COST
        unit = s.surface.get(uid)
        return unit.current if unit else 0.0

    def _assault_fire(self, unit, cell) -> None:
        """PD and SDB fire on units landing in or leaving a hex within three."""
        s = self.state
        if not isinstance(cell, int):
            return
        firers = []
        for other in s.surface.values():
            if (other.side == unit.side or other.dead
                    or not other.cls.planetary_defense):
                continue
            if self._in_pd_range(other, cell):
                firers.append(float(other.cls.bombard))
        for wing in s.naval.values():
            if wing.is_sdb and not wing.hidden and wing.side != unit.side \
                    and self._in_pd_range(wing, cell):
                firers.append(wing.bombard_factor())
        if not firers:
            return
        tech = 14 if isinstance(unit, Base) else unit.cls.tech
        assault = ("jump" if not isinstance(unit, Base)
                   and unit.cls.assault_class != "other" else "other")
        roll = (self.d6() + tables.TECH_MODIFIER.get(tech, -1)
                + tables.ASSAULT_MODIFIER[assault])
        loss = tables.surface_bombard(sum(firers), roll)
        unit.losses = min(100, unit.losses + loss)

    def restoration_phase(self, surfaced: list) -> None:
        """SDB wings go back under; bombarding squadrons return to close orbit."""
        s = self.state
        for wing in surfaced:
            if wing.uid in s.naval:
                wing.hidden = True
        for unit in s.naval.values():
            if unit.mission == BOMBARDMENT and isinstance(unit.location, int):
                unit.location = CLOSE_ORBIT
            unit.mission = None

    # ---- 4. surface segment ------------------------------------------
    def surface_segment(self) -> None:
        for side in (IMPERIAL, SOLOMANI):
            self.movement_phase(side)
            self.combat_phase(side)
        self._update_garrisons()
        self._destroy_stranded_bases()

    def movement_phase(self, side: str) -> None:
        s = self.state
        enemy_zoc = s.zone_of_control(enemy_of(side))
        occupied = {u.location for u in s.surface.values()
                    if u.side != side and u.carrier is None and not u.dead
                    and isinstance(u.location, int)}
        stacked = self._stacking(side)
        for uid, destination in self.agents[side].surface_moves(s, side, self).items():
            unit = s.surface.get(uid)
            if unit is None or unit.side != side or unit.dead:
                continue
            if unit.carrier is not None or not unit.cls.mobile:
                continue
            if unit.landed_turn == s.turn:
                continue        # "may not move ... on the turn it is unloaded"
            if not isinstance(unit.location, int) or not isinstance(destination, int):
                continue
            if destination not in s.geometry.land:
                continue        # "may not end its movement in an all-sea hex"
            if destination != unit.location and \
                    stacked.get(destination, 0.0) + unit.current > oob.STACKING_LIMIT:
                continue        # "no more than 1000 combat factors ... in a
                                # single hex"
            cost = self._path_cost(unit, unit.location, destination,
                                   enemy_zoc, occupied)
            if cost is not None and cost <= oob.MOVEMENT_POINTS:
                stacked[unit.location] = stacked.get(unit.location, 0.0) - unit.current
                stacked[destination] = stacked.get(destination, 0.0) + unit.current
                unit.location = destination

    def _stacking(self, side: str) -> dict:
        """How many combat factors ``side`` has in each hex.

        Rule 3's stacking limit is a limit on what a player may have in a hex,
        not on what both players may have between them, so the two sides are
        counted separately.  A move that would break it is refused rather than
        corrected, and moves are taken in the order the agent offers them: a
        doctrine that wants a particular unit in a full hex has to say so
        first, which is the same discipline the loading order needs.
        """
        totals: dict = {}
        for unit in self.state.surface.values():
            if unit.side != side or unit.carrier is not None or unit.dead:
                continue
            if isinstance(unit.location, int):
                totals[unit.location] = totals.get(unit.location, 0.0) + unit.current
        return totals

    def _path_cost(self, unit, origin, destination, enemy_zoc, occupied):
        """Cheapest legal path, charging the zone-of-control surcharges."""
        if origin == destination:
            return 0
        return self.movement_costs(unit, enemy_zoc, occupied,
                                   origin=origin).get(destination)

    def movement_costs(self, unit, enemy_zoc, occupied, origin=None) -> dict:
        """What every hex within reach costs this unit, from one search.

        "The total cost to enter a hex in an enemy zone of control is 2
        movement points ... the total cost to enter an enemy occupied hex in an
        enemy zone of control is 3."  A commando "may totally ignore the
        presence of enemy units and enemy zones of control while moving"; a
        guerrilla starting in an enemy-occupied hex "may move only one hex".

        The whole cost map is returned rather than one hex's cost, because a
        doctrine choosing where to send a corps wants to compare thirty
        destinations and the search that answers one answers all of them.
        """
        origin = unit.location if origin is None else origin
        commando = unit.cls.commando
        if unit.cls.guerrilla and origin in occupied:
            limit = 1
        else:
            limit = oob.MOVEMENT_POINTS
        best = {origin: 0}
        frontier = [origin]
        while frontier:
            nxt = []
            for cell in frontier:
                if cell in occupied and cell != origin and not commando:
                    continue     # "must cease movement ... upon entering an
                                 # enemy occupied hex"
                for step in self.state.geometry.adjacent(cell):
                    cost = 1
                    if not commando:
                        if step in enemy_zoc:
                            cost += 1
                        if step in occupied:
                            cost += 1
                    total = best[cell] + cost
                    if total > limit:
                        continue
                    if step not in best or total < best[step]:
                        best[step] = total
                        nxt.append(step)
            frontier = nxt
        return best

    def combat_phase(self, side: str) -> None:
        """Combat in every hex where the moving side meets the other."""
        s = self.state
        enemy = enemy_of(side)
        hiding = self.agents[SOLOMANI].guerrilla_hiding(s, SOLOMANI, self)
        for unit in s.surface.values():
            if unit.cls.guerrilla:
                unit.hiding = unit.uid in hiding

        contested = set()
        for unit in s.surface.values():
            if unit.carrier is None and not unit.dead and isinstance(unit.location, int):
                if s.surface_at(unit.location, side) and s.surface_at(unit.location, enemy):
                    contested.add(unit.location)
        for cell in sorted(contested):
            self._resolve_surface_combat(cell)
        self._clear_dead()

    def _resolve_surface_combat(self, cell) -> None:
        """A simultaneous exchange; results applied at the end of the phase."""
        s = self.state
        winter = s.in_winter(cell)
        pending: dict[int, list] = {}
        supply = {u.uid: self._in_supply(u) for u in s.surface_at(cell)}

        for side in (IMPERIAL, SOLOMANI):
            for firer_uid, target_uid, factors in \
                    self.agents[side].allocate_fire(s, side, cell, self):
                firer, target = s.surface.get(firer_uid), s.surface.get(target_uid)
                if firer is None or target is None or firer.side != side:
                    continue
                if firer.location != cell or target.location != cell:
                    continue
                if target.side == side or firer.dead or target.dead:
                    continue
                if not supply.get(firer.uid, True):
                    continue     # out of supply "may not fire (attack)"
                if firer.cls.guerrilla and firer.hiding:
                    continue     # hiding "may not fire during surface combat"
                pending.setdefault(target.uid, []).append((firer, float(factors)))

        results: dict[int, int] = {}
        for target_uid, attacks in pending.items():
            target = s.surface[target_uid]
            firing = sum(f for _u, f in attacks)
            if firing <= 0:
                continue
            defending = target.defence_strength(supply.get(target_uid, True))
            # "subtract the tech level of the attacked unit from the tech level
            # of the unit with the lowest tech level that is contributing any
            # factors to the attack"
            lowest = min(u.cls.tech for u, _f in attacks)
            shift = lowest - target.cls.tech
            roll = self.d66()
            if winter:
                roll += tables.WINTER_MODIFIER
            if target.cls.guerrilla and target.hiding:
                roll += tables.GUERRILLA_HIDING_MODIFIER
            result = tables.troop_combat(firing, defending, roll, shift)
            results[target_uid] = 100 if result == "d" else result

        for uid, loss in results.items():
            unit = s.surface.get(uid)
            if unit is not None:
                unit.losses = min(100, unit.losses + loss)

    def _in_supply(self, unit) -> bool:
        """Trace a supply line up to five hexes to a source.

        "A Solomani surface unit may use any starport or urban hex that is not
        garrisoned by Imperial units as a source of supply.  An Imperial
        surface unit may use any Imperial base as a source of supply.  Solomani
        guerrilla units are always in supply."  A commando is always in supply
        too, by rule 5.
        """
        s = self.state
        if unit.cls.guerrilla or unit.cls.commando:
            return True
        if not isinstance(unit.location, int):
            return True
        if unit.side == SOLOMANI:
            sources = set(s.solomani_urban()) | {
                c for c in s.geometry.starports if c not in s.garrisoned}
        else:
            sources = {b.location for b in s.bases.values()
                       if b.carrier is None and not b.dead
                       and isinstance(b.location, int)}
        if not sources:
            return False
        if unit.location in sources:
            return True
        enemy = enemy_of(unit.side)
        enemy_zoc = s.zone_of_control(enemy)
        blocked = set()
        for other in s.surface.values():
            if other.side != enemy or other.carrier is not None or other.dead:
                continue
            if isinstance(other.location, int):
                blocked.add(other.location)
        friendly = {u.location for u in s.surface.values()
                    if u.side == unit.side and u.carrier is None and not u.dead
                    and isinstance(u.location, int)}
        seen = {unit.location}
        frontier = [unit.location]
        for _ in range(oob.SUPPLY_RANGE):
            nxt = []
            for cell in frontier:
                for step in s.geometry.adjacent(cell):
                    if step in seen:
                        continue
                    if step in blocked and step not in friendly:
                        continue
                    if step in enemy_zoc and step not in friendly:
                        continue
                    seen.add(step)
                    if step in sources:
                        return True
                    nxt.append(step)
            frontier = nxt
        return False

    def _clear_dead(self) -> None:
        s = self.state
        for unit in list(s.surface.values()):
            if unit.dead:
                s.pools["dead_surface"].append(unit.cls)
                del s.surface[unit.uid]
        for base in list(s.bases.values()):
            if base.dead:
                del s.bases[base.uid]

    def _update_garrisons(self) -> None:
        """Which starports and urban hexes the Imperium holds.

        "An Imperial unit having a zone of control is able to garrison all
        hexes in its zone of control.  A hex in an Imperial unit's zone of
        control is considered to be garrisoned as long as the hex is neither
        occupied by a Solomani unit nor in the zone of control of a Solomani
        unit."  A hiding guerrilla does not stop it.
        """
        s = self.state
        imperial_zoc = s.zone_of_control(IMPERIAL)
        solomani_zoc = s.zone_of_control(SOLOMANI)
        occupied = set()
        for unit in s.surface.values():
            if unit.side != SOLOMANI or unit.carrier is not None or unit.dead:
                continue
            if unit.cls.guerrilla and unit.hiding:
                continue
            if isinstance(unit.location, int):
                occupied.add(unit.location)
        held = set()
        for cell in s.geometry.urban | s.geometry.starports:
            if cell in imperial_zoc and cell not in occupied \
                    and cell not in solomani_zoc:
                held.add(cell)
        s.garrisoned = held

    def _destroy_stranded_bases(self) -> None:
        s = self.state
        for base in list(s.bases.values()):
            if base.carrier is not None or not isinstance(base.location, int):
                continue
            here = s.surface_at(base.location)
            if any(u.side == SOLOMANI for u in here) and \
                    not any(u.side == IMPERIAL for u in here):
                del s.bases[base.uid]
                s.note("Imperial base overrun")

    # ---- 5. the quarterly special turn -------------------------------
    def quarterly_special_turn(self) -> None:
        s = self.state
        if self.agents[IMPERIAL].abandon_invasion(s, self):
            s.abandoned = True
            s.game_over = True
            s.result = "solomani major victory"
            s.note("the Imperial player abandons the invasion")
            return
        if len(s.solomani_urban()) < VICTORY_URBAN_THRESHOLD:
            self._end("Solomani resources exhausted")
            return
        self._quarterly_replacements()
        self._quarterly_guerrillas()
        self._quarterly_sdb()

    def _quarterly_replacements(self) -> None:
        s = self.state
        waves = max(0, self.agents[IMPERIAL].replacement_waves(s, self))
        if waves:
            s.waves_taken += waves
            self._rebuild_surface(IMPERIAL, waves * 100, 1.0)
            self._rebuild_naval(waves)
        self.agents[SOLOMANI].spend_replacements(s, SOLOMANI, self)
        self._rebuild_surface(SOLOMANI, s.replacement_points[SOLOMANI], 1.0)

    def _rebuild_surface(self, side: str, budget: float, rate: float) -> None:
        """Spend replacement points rebuilding eliminated units, cheapest first.

        "A unit must be replaced at full strength, with one replacement point
        being spent for each combat factor of unit, regardless of tech level,
        armor, or elite status."  Elite units may not be rebuilt.
        """
        s = self.state
        if budget <= 0:
            return
        pool = [c for c in s.pools["dead_surface"]
                if (c.name.startswith("colonial") or True)
                and not c.elite and not c.planetary_defense
                and ((side == IMPERIAL) == (c not in _solomani_classes()))]
        pool = [c for c in s.pools["dead_surface"] if not c.elite
                and not c.planetary_defense and _side_of_class(c) == side]
        pool.sort(key=lambda c: c.factor)
        spent = 0.0
        for cls in list(pool):
            cost = cls.factor * rate
            if spent + cost > budget:
                continue
            spent += cost
            s.pools["dead_surface"].remove(cls)
            where = OUT_SYSTEM if side == IMPERIAL else self._solomani_home()
            if where is None:
                break
            _add_surface(s, cls, side, where)
        s.replacement_points[side] = max(0.0, budget - spent) if side == SOLOMANI \
            else s.replacement_points[side]

    def _solomani_home(self):
        urban = self.state.solomani_urban()
        return urban[0] if urban else None

    def _rebuild_naval(self, waves: int) -> None:
        """"An Imperial replacement wave may be used to replace any three
        eliminated Imperial naval units."  Waves are shared with the surface
        rebuild above, so this replaces what is left over rather than doubling.
        """
        s = self.state
        pool = s.pools["dead_imperial_naval"]
        for _ in range(min(waves * 3, len(pool))):
            cls = pool.pop(0)
            _add_naval(s, cls, IMPERIAL, OUT_SYSTEM)

    def _quarterly_guerrillas(self) -> None:
        """Four guerrilla units at the end of each of the first two quarters."""
        s = self.state
        if s.quarters_elapsed > 2:
            return
        pool = s.pools["guerrillas"]
        count = min(oob.GUERRILLA_WAVE, len(pool))
        if not count:
            return
        cells = self.agents[SOLOMANI].place_guerrillas(s, count, self)
        used = set()
        for cell in cells:
            if not pool:
                break
            if cell in used or not isinstance(cell, int) or cell not in s.geometry.land:
                continue
            used.add(cell)
            _add_surface(s, pool.pop(), SOLOMANI, cell)

    def _quarterly_sdb(self) -> None:
        """Lay down new SDB wings, and activate the ones laid down last quarter."""
        s = self.state
        building = s.pools["sdb_building"]
        ready = []
        for cell, cls in list(building.items()):
            if cell in s.garrisoned:
                del building[cell]          # "automatically destroyed"
                continue
            ready.append((cell, cls))
            del building[cell]
        if ready:
            placement = self.agents[SOLOMANI].deploy_sdb(s, ready, self)
            taken = {u.location for u in s.naval.values()
                     if u.is_sdb and isinstance(u.location, int)}
            for cell, cls in ready:
                where = placement.get(cell, CLOSE_ORBIT)
                if isinstance(where, int) and where in s.geometry.deep_sea \
                        and where not in taken and hexmap.distance(cell, where) <= 3:
                    _add_naval(s, cls, SOLOMANI, where, hidden=True)
                    taken.add(where)
                else:
                    _add_naval(s, cls, SOLOMANI, CLOSE_ORBIT)
        for cell in self.agents[SOLOMANI].build_sdb(s, self):
            if cell in s.geometry.starports and cell not in s.garrisoned:
                building[cell] = oob.SOLOMANI_SDB[0][0]

    def _end(self, reason: str) -> None:
        s = self.state
        s.game_over = True
        s.result = s.victory_level()
        s.note("%s -- %s (%d victory points)"
               % (reason, s.result, s.victory_points()))


_SOLOMANI_CLASSES = None


def _solomani_classes():
    global _SOLOMANI_CLASSES
    if _SOLOMANI_CLASSES is None:
        _SOLOMANI_CLASSES = set()
        for spec in (oob.SOLOMANI_TROOPS, oob.SOLOMANI_PD,
                     oob.SOLOMANI_GUERRILLAS):
            for cls, _n in spec:
                _SOLOMANI_CLASSES.add(cls)
    return _SOLOMANI_CLASSES


def _side_of_class(cls) -> str:
    return SOLOMANI if cls in _solomani_classes() else IMPERIAL


# ==========================================================================
def play(state: GameState, imperial, solomani, max_turns: int = MAX_TURNS,
         on_turn=None, rng=None) -> str:
    """Play a whole game and return the level of victory."""
    engine = Engine(state, imperial, solomani, rng=rng)
    while not state.game_over and state.turn <= max_turns:
        engine.play_turn()
        if on_turn is not None:
            on_turn(state)
    if not state.game_over:
        state.game_over = True
        state.result = state.victory_level()
    return state.result

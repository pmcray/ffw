"""The Fifth Frontier War rules engine: setup, sequence of play, combat."""

from __future__ import annotations

import random
from typing import Iterable

from . import hexmap, oob, tables
from .state import (Admiral, Fleet, GameState, Squadron, TroopUnit, World,
                    WorldMap, IMPERIAL, ZHODANI, NAVIES_OF, side_of)

REINFORCEMENT_BOXES = {
    "imperial_reinforcements": IMPERIAL,
    "imperial_rimward": IMPERIAL,
    "zhodani_reinforcements": ZHODANI,
    "zhodani_additional": ZHODANI,
    "sword_worlds": ZHODANI,
}

MAX_TURNS = 60
MAX_COMBAT_ROUNDS = 12


# ==========================================================================
#  setup
# ==========================================================================
def new_game(seed: int | None = None, options: dict | None = None) -> GameState:
    """Build a game set up according to the two order of battle charts."""
    rng = random.Random(seed)
    state = GameState(world_map=WorldMap.load())
    if options:
        state.options.update(options)
    state.rng = rng                                  # type: ignore[attr-defined]

    _build_pools(state, rng)
    _deploy_zhodani_guerrillas(state, rng)
    _deploy_imperial(state, rng)
    _deploy_zhodani(state, rng)
    _choose_secret_base(state, rng)
    _build_special_group(state, rng)
    for world in state.world_map:
        state.update_control(world.hex)
    state.note("game set up; %d squadrons, %d troop units in play"
               % (len(state.squadrons), len(state.troops)))
    return state


def _build_pools(state: GameState, rng: random.Random) -> None:
    """Fill the off-map reinforcement and replacement pools."""
    p = state.pools
    p["imperial_battle"] = oob.expand(oob.IMPERIAL_BATTLE)
    p["imperial_black_globes"] = oob.expand(oob.IMPERIAL_BLACK_GLOBES)
    p["imperial_cruiser"] = oob.expand(oob.IMPERIAL_CRUISER)
    p["imperial_scout"] = oob.expand(oob.IMPERIAL_SCOUT)
    p["imperial_tanker"] = oob.expand(oob.IMPERIAL_TANKER)
    p["imperial_assault"] = oob.expand(oob.IMPERIAL_ASSAULT)
    p["imperial_colonial_sq"] = oob.expand(oob.IMPERIAL_COLONIAL_SQUADRONS)
    p["imperial_troops_5c"] = oob.expand(oob.IMPERIAL_TROOPS_5C)
    p["imperial_troops_1c"] = oob.expand(oob.IMPERIAL_TROOPS_1C)
    p["imperial_troops_20"] = oob.expand(oob.IMPERIAL_TROOPS_20)
    p["imperial_colonial_tr"] = oob.expand(oob.IMPERIAL_COLONIAL_TROOPS)
    p["imperial_marines"] = oob.expand(oob.IMPERIAL_MARINES)
    p["imperial_huscarles"] = oob.expand(oob.IMPERIAL_HUSCARLES)
    p["imperial_mercenaries"] = oob.expand(oob.IMPERIAL_MERCENARIES)
    p["zhodani_battle"] = oob.expand(oob.ZHODANI_BATTLE)
    p["zhodani_cruiser"] = oob.expand(oob.ZHODANI_CRUISER)
    p["zhodani_scout"] = oob.expand(oob.ZHODANI_SCOUT)
    p["zhodani_tanker"] = oob.expand(oob.ZHODANI_TANKER)
    p["zhodani_assault"] = oob.expand(oob.ZHODANI_ASSAULT)
    p["zhodani_colonial_sq"] = oob.expand(oob.ZHODANI_COLONIAL_SQUADRONS)
    p["zhodani_troops_5c"] = oob.expand(oob.ZHODANI_TROOPS_5C)
    p["zhodani_troops_1c"] = oob.expand(oob.ZHODANI_TROOPS_1C)
    p["zhodani_troops_20"] = oob.expand(oob.ZHODANI_TROOPS_20)
    p["zhodani_troops_10"] = oob.expand(oob.ZHODANI_TROOPS_10)
    p["zhodani_troops_5"] = oob.expand(oob.ZHODANI_TROOPS_5)
    p["zhodani_colonial_tr"] = oob.expand(oob.ZHODANI_COLONIAL_TROOPS)
    p["zhodani_guards"] = oob.expand(oob.ZHODANI_GUARDS)
    for key, value in p.items():
        rng.shuffle(value)

    state.replacement_points[IMPERIAL]["squadron"] = \
        oob.IMPERIAL_REPLACEMENTS["squadron"]["initial"]
    state.replacement_points[IMPERIAL]["troop"] = \
        oob.IMPERIAL_REPLACEMENTS["troop"]["initial"]
    state.replacement_points[ZHODANI]["squadron"] = \
        oob.ZHODANI_REPLACEMENTS["squadron"]["initial"]
    state.replacement_points[ZHODANI]["troop"] = \
        oob.ZHODANI_REPLACEMENTS["troop"]["initial"]

    # fleet markers and admirals
    # Every marker in the counter mix exists from the start, but only the
    # opening allotment is available; the rest arrive as reinforcements.
    for navy, names in (("imperial", oob.IMPERIAL_FLEETS),
                        ("zhodani", oob.ZHODANI_FLEETS),
                        ("sword_worlds", oob.SWORD_WORLDS_FLEETS),
                        ("vargr", oob.VARGR_FLEETS)):
        opening = oob.INITIAL_FLEETS.get(navy, len(names))
        for i, name in enumerate(names):
            _add_fleet(state, name, navy).available = i < opening
    p["imperial_admirals"] = [
        _make_admiral(state, "imperial", *a) for a in oob.IMPERIAL_ADMIRALS]
    p["zhodani_admirals"] = [
        _make_admiral(state, "zhodani", *a) for a in oob.ZHODANI_ADMIRALS]
    p["sword_worlds_admirals"] = [
        _make_admiral(state, "sword_worlds", *a) for a in oob.SWORD_WORLDS_ADMIRALS]
    p["vargr_admirals"] = [
        _make_admiral(state, "vargr", *a) for a in oob.VARGR_ADMIRALS]
    # loaned admirals: a Sword Worlds officer serves with the Zhodani and a
    # Vargr mercenary with the Imperium, both at precedence 15
    p["zhodani_admirals"].append(
        _make_admiral(state, "zhodani", 15, 4, 1, "Sword Worlds liaison"))
    p["imperial_admirals"].append(
        _make_admiral(state, "imperial", 15, 2, 1, "Vargr mercenary admiral"))
    for key in ("imperial_admirals", "zhodani_admirals"):
        rng.shuffle(p[key])
    p["imperial_warrant"] = 1


def _build_special_group(state: GameState, rng: random.Random) -> None:
    """Rule 6's Imperial special group: tankers, assault carriers, the warrant.

    Built after deployment so that the squadrons already on the map are not
    also sitting in the draw pile.  Entries are drawn one at a time, one per
    fleet marker that arrives as a reinforcement, so the warrant reaches play
    on a die roll rather than at a fixed turn -- which is the point of it being
    in the group at all.
    """
    p = state.pools
    group: list = [("squadron", "imperial_tanker", cls)
                   for cls in p["imperial_tanker"]]
    group += [("squadron", "imperial_assault", cls)
              for cls in p["imperial_assault"]]
    if p.get("imperial_warrant"):
        group.append(("warrant", None, None))
    rng.shuffle(group)
    p["imperial_special"] = group
    p["imperial_tanker"], p["imperial_assault"] = [], []


def _add_fleet(state: GameState, name: str, navy: str) -> Fleet:
    fleet = Fleet(uid=state.new_uid(), name=name, navy=navy, location="")
    state.fleets[fleet.uid] = fleet
    return fleet


def _make_admiral(state: GameState, navy: str, precedence: int,
                  planning: int, tactical: int, name: str) -> Admiral:
    adm = Admiral(uid=state.new_uid(), navy=navy, precedence=precedence,
                  planning=planning, tactical=tactical, name=name, location="")
    state.admirals[adm.uid] = adm
    return adm


def _place_squadron(state: GameState, cls, navy: str, where: str,
                    colonial: bool = False) -> Squadron:
    sq = Squadron(uid=state.new_uid(), cls=cls, navy=navy, location=where,
                  colonial=colonial)
    sq.black_globe = (navy == "imperial" and cls.attack == 6
                      and cls.bombard == 2 and cls.defense == 8)
    state.squadrons[sq.uid] = sq
    return sq


def _place_troop(state: GameState, cls, navy: str, where: str,
                 guerrilla: bool = False) -> TroopUnit:
    tr = TroopUnit(uid=state.new_uid(), cls=cls, navy=navy, location=where,
                   guerrilla=guerrilla)
    state.troops[tr.uid] = tr
    return tr


def _worlds_of(state: GameState, owner: str) -> list[World]:
    return sorted((w for w in state.world_map if w.owner == owner),
                  key=lambda w: (-w.defense_battalions, -w.tech_level, w.hex))


def _deploy_zhodani_guerrillas(state: GameState, rng: random.Random) -> None:
    """Rule 5: ten Ine Givar units, 3+ on Efate and 1+ on Ruie."""
    efate = _world_named(state, "Efate")
    ruie = _world_named(state, "Ruie")
    hosts = [efate.hex] * 3 + [ruie.hex]
    candidates = [w for w in state.world_map
                  if w.owner == "imperial" and w.defense_battalions > 0
                  and w.zone != "red" and w.hex not in (efate.hex, ruie.hex)]
    candidates.sort(key=lambda w: -w.defense_battalions)
    pick = candidates[:20]
    rng.shuffle(pick)
    hosts.extend(w.hex for w in pick[:6])
    for host in hosts:
        world = state.world_map.get(host)
        cls = oob.TroopClass("guerrilla", max(1, world.defense_battalions // 10),
                             world.tech_level, name="Ine Givar")
        _place_troop(state, cls, "zhodani", host, guerrilla=True)


def _world_named(state: GameState, name: str) -> World:
    for world in state.world_map:
        if world.name == name:
            return world
    raise KeyError(name)


def _deploy_imperial(state: GameState, rng: random.Random) -> None:
    p = state.pools
    imperial_worlds = _worlds_of(state, "imperial")
    naval = [w for w in imperial_worlds if w.naval_base]
    scout = [w for w in imperial_worlds if w.scout_base]
    quar, zircon = _world_named(state, "Quar"), _world_named(state, "Zircon")
    naval.append(quar)
    scout.append(zircon)

    # 16 named colonial squadrons and troops at their home worlds.  Named
    # counters are not drawn from the reinforcement pools, so the pools are
    # cycled rather than consumed.
    homes = imperial_worlds[:16]
    sq_cycle = p["imperial_colonial_sq"]
    tr_cycle = p["imperial_colonial_tr"]
    for i, world in enumerate(homes):
        _place_squadron(state, sq_cycle[i % len(sq_cycle)], "imperial",
                        world.hex, colonial=True)
        _place_troop(state, tr_cycle[i % len(tr_cycle)], "imperial", world.hex)

    # required placements from the Imperial order of battle chart
    efate, ruie = _world_named(state, "Efate"), _world_named(state, "Ruie")
    psg = next(c for c in oob.expand(oob.IMPERIAL_MERCENARIES) if c.name == "PSG")
    spl = next(c for c in oob.expand(oob.IMPERIAL_MERCENARIES) if c.name == "SPL")
    _place_troop(state, psg, "imperial", efate.hex)
    _place_troop(state, spl, "imperial", ruie.hex)
    _place_troop(state, p["imperial_troops_1c"].pop(), "imperial", efate.hex)
    _place_squadron(state, p["imperial_assault"].pop(), "imperial", efate.hex)

    # remaining mercenaries anywhere in the Imperium
    for cls in [c for c in oob.expand(oob.IMPERIAL_MERCENARIES)
                if c.name not in ("PSG", "SPL")][:7]:
        _place_troop(state, cls, "imperial", rng.choice(imperial_worlds[:30]).hex)

    for cls in p["imperial_marines"][:8]:
        _place_troop(state, cls, "imperial",
                     rng.choice(naval or imperial_worlds).hex)
    for cls in p["imperial_huscarles"][:8]:
        _place_troop(state, cls, "imperial",
                     rng.choice(imperial_worlds[:20]).hex)
    for _ in range(4):
        if p["imperial_troops_20"]:
            _place_troop(state, p["imperial_troops_20"].pop(), "imperial",
                         rng.choice(imperial_worlds[:20]).hex)

    for cls in p["imperial_scout"][:8]:
        _place_squadron(state, cls, "imperial",
                        rng.choice(scout or imperial_worlds).hex)
    for _ in range(2):
        if p["imperial_battle"]:
            _place_squadron(state, p["imperial_battle"].pop(), "imperial",
                            rng.choice(naval).hex)
    for _ in range(6):
        if p["imperial_cruiser"]:
            _place_squadron(state, p["imperial_cruiser"].pop(), "imperial",
                            rng.choice(naval).hex)

    # the Duke of Regina plus three other admirals
    regina = _world_named(state, "Regina")
    duke = p["imperial_admirals"].pop()
    duke.name, duke.location = "Duke of Regina", regina.hex
    duke.aboard = False
    for _ in range(3):
        adm = p["imperial_admirals"].pop()
        adm.location = rng.choice(naval).hex
    for adm in p["vargr_admirals"]:
        adm.navy = "imperial"
        adm.precedence = 15
        adm.location = rng.choice(naval).hex
        break
    _organise_new_fleets(state, IMPERIAL)


def _deploy_zhodani(state: GameState, rng: random.Random) -> None:
    p = state.pools
    zho = _worlds_of(state, "zhodani")
    sw = _worlds_of(state, "sword_worlds")
    vargr = _worlds_of(state, "vargr")

    sq_cycle, tr_cycle = p["zhodani_colonial_sq"], p["zhodani_colonial_tr"]
    for i, world in enumerate(zho[:16]):
        _place_squadron(state, sq_cycle[i % len(sq_cycle)], "zhodani",
                        world.hex, colonial=True)
        _place_troop(state, tr_cycle[i % len(tr_cycle)], "zhodani", world.hex)

    staging = [w for w in zho if w.base] or zho
    for _ in range(23):
        if p["zhodani_battle"]:
            _place_squadron(state, p["zhodani_battle"].pop(), "zhodani",
                            rng.choice(staging[:8]).hex)
    for _ in range(14):
        if p["zhodani_cruiser"]:
            _place_squadron(state, p["zhodani_cruiser"].pop(), "zhodani",
                            rng.choice(staging[:8]).hex)
    for _ in range(2):
        if p["zhodani_tanker"]:
            _place_squadron(state, p["zhodani_tanker"].pop(), "zhodani",
                            rng.choice(staging[:8]).hex)
    for _ in range(6):
        if p["zhodani_assault"]:
            _place_squadron(state, p["zhodani_assault"].pop(), "zhodani",
                            rng.choice(staging[:8]).hex)
    for cls in p["zhodani_scout"][:6]:
        _place_squadron(state, cls, "zhodani", rng.choice(staging).hex)

    for _ in range(7):
        if p["zhodani_troops_5c"]:
            _place_troop(state, p["zhodani_troops_5c"].pop(), "zhodani",
                         rng.choice(staging[:8]).hex)
    for _ in range(10):
        if p["zhodani_troops_1c"]:
            _place_troop(state, p["zhodani_troops_1c"].pop(), "zhodani",
                         rng.choice(staging[:8]).hex)
    for _ in range(16):
        if p["zhodani_troops_20"]:
            _place_troop(state, p["zhodani_troops_20"].pop(), "zhodani",
                         rng.choice(staging[:8]).hex)
    for _ in range(2):
        for key in ("zhodani_troops_10", "zhodani_troops_5"):
            if p[key]:
                _place_troop(state, p[key].pop(), "zhodani",
                             rng.choice(staging[:8]).hex)
    for cls in p["zhodani_guards"]:
        _place_troop(state, cls, "zhodani", rng.choice(staging[:4]).hex)

    for _ in range(7):
        adm = p["zhodani_admirals"].pop()
        adm.location = rng.choice(staging[:8]).hex
    governor = next((a for a in state.admirals.values()
                     if "Governor" in a.name), None)
    if governor is not None:
        governor.location = _world_named(state, "Chronor").hex

    # Sword Worlds initial forces
    if sw:
        for cls in oob.expand(oob.SWORD_WORLDS_BATTLE)[:2]:
            _place_squadron(state, cls, "sword_worlds", sw[0].hex)
        for cls in oob.expand(oob.SWORD_WORLDS_CRUISER)[:1]:
            _place_squadron(state, cls, "sword_worlds", sw[0].hex)
        for cls in oob.expand(oob.SWORD_WORLDS_TANKER)[:1]:
            _place_squadron(state, cls, "sword_worlds", sw[0].hex)
        for cls in oob.expand(oob.SWORD_WORLDS_TROOPS)[:1]:
            _place_troop(state, cls, "sword_worlds", sw[0].hex)
        for adm in p["sword_worlds_admirals"][:2]:
            adm.location = sw[0].hex
    # Vargr initial forces
    if vargr:
        for cls in oob.expand(oob.VARGR_BATTLE):
            _place_squadron(state, cls, "vargr", vargr[0].hex)
        for cls in oob.expand(oob.VARGR_CRUISER):
            _place_squadron(state, cls, "vargr", vargr[0].hex)
        for cls in oob.expand(oob.VARGR_TANKER) + oob.expand(oob.VARGR_ASSAULT):
            _place_squadron(state, cls, "vargr", vargr[0].hex)
        for cls in oob.expand(oob.VARGR_TROOPS):
            _place_troop(state, cls, "vargr", vargr[0].hex)
        for adm in p["vargr_admirals"][:2]:
            if adm.navy == "vargr":
                adm.location = vargr[0].hex

    _organise_new_fleets(state, ZHODANI)


def _choose_secret_base(state: GameState, rng: random.Random) -> None:
    """Rule 5: an Imperial world with no base, no defence battalions, not red."""
    candidates = [w for w in state.world_map
                  if w.owner == "imperial" and not w.naval_base
                  and not w.scout_base and w.defense_battalions == 0
                  and w.zone != "red" and w.name != "Esalin"]
    preferred = [w for w in candidates if w.name == "Fulacin"]
    state.secret_base = (preferred or candidates or list(state.world_map))[0].hex \
        if preferred else (rng.choice(candidates).hex if candidates else None)


# ==========================================================================
#  fleet organisation
# ==========================================================================
#: a fleet larger than this cannot be handled usefully; the slowest squadron
#: sets the whole fleet's jump number, so big mixed fleets crawl
MAX_FLEET_SIZE = 10


def _detach(state: GameState, sq: Squadron) -> None:
    if sq.fleet is None:
        return
    fleet = state.fleets.get(sq.fleet)
    if fleet is not None and sq.uid in fleet.squadrons:
        fleet.squadrons.remove(sq.uid)
    sq.fleet = None


def _release_marker(state: GameState, fleet: Fleet) -> None:
    fleet.active = False
    fleet.plot.clear()
    fleet.squadrons.clear()
    if fleet.admiral is not None:
        adm = state.admirals.get(fleet.admiral)
        if adm is not None:
            adm.fleet = None
            adm.aboard = False
        fleet.admiral = None


def reorganise_fleets(state: GameState, side: str,
                      max_size: int = MAX_FLEET_SIZE, reserve: int = 0) -> None:
    """Fleet adjustment step: keep fleets fast, coherent and the right size.

    Fleets are restricted by their slowest and thirstiest squadron, so the aim
    is homogeneous groups: squadrons are sorted into jump-number bands, each
    band is capped at ``max_size``, and markers freed by empty fleets are
    re-spent on the largest unattached concentrations.

    ``reserve`` holds that many markers back unspent.  Markers are scarce -- a
    player has fourteen and this step will happily consume every one on the
    main effort -- so a doctrine that wants to detach a picket has to be given
    the room to do it before the concentrations are re-formed.
    """
    # 1. prune fleets: drop squadrons that drag the fleet's jump number down,
    #    and trim any fleet over the size cap
    for fleet in list(state.fleets.values()):
        if not fleet.active or fleet.side != side:
            continue
        members = [state.squadrons[u] for u in list(fleet.squadrons)
                   if u in state.squadrons]
        if not members:
            _release_marker(state, fleet)
            continue
        jumps = sorted({sq.jump for sq in members})
        if len(jumps) > 1:
            keep_jump = max(jumps, key=lambda j: sum(1 for sq in members
                                                     if sq.jump >= j) * j)
            for sq in members:
                if sq.jump < keep_jump:
                    _detach(state, sq)
        members = [state.squadrons[u] for u in list(fleet.squadrons)
                   if u in state.squadrons]
        if len(members) > max_size:
            members.sort(key=lambda sq: (-sq.attack - sq.bombard, -sq.defense))
            for sq in members[max_size:]:
                if not sq.troops:
                    _detach(state, sq)
        if not fleet.squadrons:
            _release_marker(state, fleet)

    # 2. attach loose squadrons to a compatible fleet in the same hex
    loose: dict[str, list[Squadron]] = {}
    for sq in state.squadrons.values():
        if sq.side != side or sq.fleet is not None or sq.cls.kind == "scout":
            continue
        if sq.location:
            loose.setdefault(sq.location, []).append(sq)
    for hex_id, squadrons in list(loose.items()):
        hosts = [f for f in state.fleets.values()
                 if f.active and f.side == side and f.location == hex_id]
        for sq in list(squadrons):
            for host in sorted(hosts, key=lambda f: -len(f.squadrons)):
                if len(host.squadrons) >= max_size:
                    continue
                members = [state.squadrons[u] for u in host.squadrons
                           if u in state.squadrons]
                if members and sq.jump < min(m.jump for m in members):
                    continue
                own = sum(1 for m in members if m.navy == host.navy)
                if sq.navy != host.navy and own <= len(members) - own:
                    continue
                host.squadrons.append(sq.uid)
                sq.fleet = host.uid
                squadrons.remove(sq)
                break
        if not squadrons:
            del loose[hex_id]

    # 3. spend spare markers on the biggest remaining concentrations
    spare = [f for f in state.fleets.values()
             if f.available and not f.active and f.side == side]
    if reserve > 0:
        spare = spare[:max(0, len(spare) - reserve)]
    order = sorted(loose.items(), key=lambda kv: (
        0 if kv[0] in REINFORCEMENT_BOXES else 1, -len(kv[1])))
    for hex_id, squadrons in order:
        if hex_id not in state.world_map.worlds and hex_id not in REINFORCEMENT_BOXES:
            continue
        by_jump: dict[int, list[Squadron]] = {}
        for sq in squadrons:
            by_jump.setdefault(sq.jump, []).append(sq)
        for jump in sorted(by_jump, reverse=True):
            group = by_jump[jump]
            while group and spare:
                chunk = group[:max_size]
                navy_counts: dict[str, int] = {}
                for sq in chunk:
                    navy_counts[sq.navy] = navy_counts.get(sq.navy, 0) + 1
                navy = max(navy_counts, key=navy_counts.get)
                fleet = next((f for f in spare if f.navy == navy), None)
                if fleet is None:
                    break
                spare.remove(fleet)
                fleet.active = True
                fleet.location = hex_id
                fleet.plot.clear()
                for sq in chunk:
                    fleet.squadrons.append(sq.uid)
                    sq.fleet = fleet.uid
                _assign_admiral(state, fleet)
                group = group[max_size:]

    # 4. precedence: the senior admiral present commands the largest fleet
    for hex_id in {f.location for f in state.fleets.values()
                   if f.active and f.side == side}:
        fleets = sorted((f for f in state.fleets.values()
                         if f.active and f.side == side and f.location == hex_id),
                        key=lambda f: -len(f.squadrons))
        free = sorted((a for a in state.admirals.values()
                       if a.side == side and a.location == hex_id),
                      key=lambda a: a.effective_precedence)
        for adm in free:
            adm.fleet, adm.aboard = None, False
        for fleet in fleets:
            fleet.admiral = None
        for fleet, adm in zip(fleets, free):
            fleet.admiral = adm.uid
            adm.fleet = fleet.uid
            adm.aboard = True


def _organise_new_fleets(state: GameState, side: str) -> None:
    """Form the opening fleets out of the deployed squadrons.

    There is no cap to apply here: how many markers a side may spend is now
    carried by ``Fleet.available``, set from ``oob.INITIAL_FLEETS``, so the
    reorganiser runs into the order of battle rather than a separate limit.
    """
    reorganise_fleets(state, side)


def _assign_admiral(state: GameState, fleet: Fleet) -> None:
    """Precedence rule: the most senior eligible admiral takes the biggest fleet."""
    if fleet.admiral is not None:
        return
    here = [a for a in state.admirals.values()
            if a.location == fleet.location and a.fleet is None
            and a.side == fleet.side]
    if not here:
        return
    own = sum(1 for uid in fleet.squadrons
              if state.squadrons[uid].navy == fleet.navy)
    other = len(fleet.squadrons) - own
    eligible = [a for a in here
                if a.navy == fleet.navy or other >= own]
    if not eligible:
        return
    adm = min(eligible, key=lambda a: a.effective_precedence)
    fleet.admiral = adm.uid
    adm.fleet = fleet.uid
    adm.aboard = True


def fleet_plotting_factor(state: GameState, fleet: Fleet) -> int:
    if fleet.admiral is not None:
        return state.admirals[fleet.admiral].planning
    return tables.FLEET_PLOTTING[fleet.navy]


def fleet_jump(state: GameState, fleet: Fleet) -> int:
    jumps = [state.squadrons[uid].jump for uid in fleet.squadrons
             if uid in state.squadrons]
    return min(jumps) if jumps else 0


def fleet_ready(state: GameState, fleet: Fleet) -> bool:
    return all(state.squadrons[uid].refuelled for uid in fleet.squadrons
               if uid in state.squadrons)


# ==========================================================================
#  refuelling
# ==========================================================================
def refuel_time(state: GameState, sq: Squadron, starport_slot: bool) -> int | None:
    """Movement phases needed to refuel, or None if no fuel is available."""
    if sq.location in REINFORCEMENT_BOXES:
        return 0                                     # boxes count as bases
    world = state.world_map.get(sq.location)
    if world is None:
        return None
    friendly = world.control == sq.side
    interdicted = (world.zone == "red" and sq.side == IMPERIAL
                   and not world.zone_lifted)
    best: int | None = None
    if world.gas_giant:
        best = tables.REFUEL_TIME["gas_giant"][sq.cls.refuel]
    if friendly and not interdicted:
        if world.base:
            best = 0
        if world.water:
            t = tables.REFUEL_TIME["ocean"][sq.cls.refuel]
            best = t if best is None else min(best, t)
        if world.starport_capacity:
            t = 0 if starport_slot else 1
            best = t if best is None else min(best, t)
    return best


def can_refuel_at(state: GameState, side: str, hex_id: str,
                  refuel_classes: Iterable[str], expect_control: bool = False) -> bool:
    """Would squadrons of these streamlining classes find fuel at ``hex_id``?

    ``expect_control`` treats the world as if the mover already held it, which
    is how a fleet planning an invasion has to think about its fuel.
    """
    if hex_id in REINFORCEMENT_BOXES:
        return True
    world = state.world_map.get(hex_id)
    if world is None:
        return False
    if world.gas_giant:
        return True
    friendly = world.control == side or expect_control
    if not friendly:
        return False
    if world.zone == "red" and side == IMPERIAL and not world.zone_lifted:
        return False
    if world.base or world.starport_capacity:
        return True
    return world.water and all(c == "streamlined" for c in refuel_classes)


def resolve_refuelling(state: GameState) -> None:
    """Advance every unrefuelled squadron's refuelling at its current location."""
    by_hex: dict[str, list[Squadron]] = {}
    for sq in state.squadrons.values():
        if not sq.refuelled:
            by_hex.setdefault(sq.location, []).append(sq)
    for hex_id, squadrons in by_hex.items():
        world = state.world_map.get(hex_id)
        slots = {IMPERIAL: 0, ZHODANI: 0}
        if world is not None:
            cap = world.starport_capacity
            for side in (IMPERIAL, ZHODANI):
                slots[side] = cap if world.control == side else 0
        # tankers refuel their consorts in zero time
        tanker_capacity = {IMPERIAL: 0, ZHODANI: 0}
        for sq in state.squadrons_at(hex_id):
            if sq.cls.kind == "tanker" and sq.refuelled:
                tanker_capacity[sq.side] += sq.defense
        for sq in sorted(squadrons, key=lambda s: -s.cls.defense):
            if tanker_capacity[sq.side] > 0 and sq.cls.kind != "tanker":
                tanker_capacity[sq.side] -= 1
                sq.refuelled, sq.refuel_progress = True, 0
                continue
            slot = slots[sq.side] > 0
            needed = refuel_time(state, sq, slot)
            if needed is None:
                continue
            if slot and needed == 0:
                slots[sq.side] -= 1
            sq.refuel_progress += 1
            if needed == 0 or sq.refuel_progress > needed:
                sq.refuelled, sq.refuel_progress = True, 0


# ==========================================================================
#  the engine
# ==========================================================================
class Engine:
    """Runs the sequence of play.  Agents supply the decisions."""

    def __init__(self, state: GameState, imperial_agent, zhodani_agent,
                 rng: random.Random | None = None):
        self.state = state
        self.agents = {IMPERIAL: imperial_agent, ZHODANI: zhodani_agent}
        self.rng = rng or getattr(state, "rng", None) or random.Random()

    # -- dice --------------------------------------------------------------
    def d6(self) -> int:
        return self.rng.randint(1, 6)

    def d66(self) -> int:
        return self.d6() + self.d6()

    # -- setup -------------------------------------------------------------
    def initial_plotting(self) -> None:
        """Rule 3: the Zhodani plan the assault before the war begins.

        Zhodani fleets are plotted four turns ahead and Vargr and Sword Worlds
        fleets five.  The Imperial player, taken by surprise, plots nothing and
        so cannot move on turn 1 apart from scouts and well-led fleets.
        """
        s = self.state
        for fleet in s.fleets.values():
            if not fleet.active or fleet.side != ZHODANI:
                continue
            factor = fleet_plotting_factor(s, fleet)
            for ahead in range(1, factor + 1):
                turn = s.turn + ahead - 1
                if turn < s.turn or turn in fleet.plot:
                    continue
                fleet.plot[turn] = self.agents[ZHODANI].plot_fleet(
                    s, ZHODANI, fleet, turn, self)

    # -- the turn ----------------------------------------------------------
    def play_turn(self) -> None:
        s = self.state
        if s.game_over:
            return
        self.reinforcement_phase()
        self.movement_phase()
        self.combat_phase()
        self.reorganisation_phase()
        self.check_victory()
        s.turn += 1

    # ---- phase 1 ---------------------------------------------------------
    def reinforcement_phase(self) -> None:
        s = self.state
        self._release_reinforcements()
        for side in (IMPERIAL, ZHODANI):
            schedule = (oob.IMPERIAL_REPLACEMENTS if side == IMPERIAL
                        else oob.ZHODANI_REPLACEMENTS)
            for kind, spec in schedule.items():
                for turns, amount in spec["turns"]:
                    if s.turn in turns:
                        s.replacement_points[side][kind] += amount
            self.agents[side].spend_replacements(s, side, self)

    # -- reinforcement helpers --------------------------------------------
    def _release_fleet_markers(self, navy: str, count: int) -> int:
        """Hand ``count`` more fleet markers to ``navy``; return how many.

        Rule 3 caps a player at the markers his order of battle has released,
        so this is what actually puts a new fleet within reach of the fleet
        adjustment step.  Markers already in play or already released are
        skipped, and running out is normal rather than an error -- the counter
        mix is finite and the table keeps rolling.
        """
        released = 0
        for fleet in self.state.fleets.values():
            if released >= count:
                break
            if fleet.navy == navy and not fleet.available:
                fleet.available = True
                released += 1
        return released

    def _draw_special(self, count: int = 1) -> None:
        """Draw from the Imperial special group: tanker, carrier, or warrant."""
        p = self.state.pools
        for _ in range(count):
            if not p.get("imperial_special"):
                return
            entry = p["imperial_special"].pop()
            kind, _key, cls = entry
            if kind == "warrant":
                held = [a for a in self.state.admirals.values()
                        if a.side == IMPERIAL and a.location]
                if not held:
                    # nobody to carry it; it stays in the group rather than
                    # being lost, and will come up again on a later draw
                    p["imperial_special"].insert(0, entry)
                    continue
                # rule 5: the warrant sets its holder's precedence to 0, so it
                # belongs with the admiral who is already senior
                holder = min(held, key=lambda a: a.precedence)
                holder.warrant = True
                p["imperial_warrant"] = 0
                self.state.note("the Emperor's warrant reaches %s" % holder.name)
                continue
            _place_squadron(self.state, cls, "imperial",
                            "imperial_reinforcements")

    def _draw_admirals(self, key: str, box: str, count: int) -> None:
        p = self.state.pools
        for _ in range(count):
            if not p[key]:
                return
            adm = p[key].pop()
            if not adm.location:
                adm.location = box

    def _release_reinforcements(self) -> None:
        s, p = self.state, self.state.pools
        if s.turn == 2:
            # rule 6: three fleet markers, three admirals, three special
            # counters, all drawn at random
            n = oob.TURN_2_IMPERIAL_DRAW
            self._release_fleet_markers("imperial", n)
            self._draw_admirals("imperial_admirals", "imperial_reinforcements", n)
            self._draw_special(n)
        if s.turn == 6:
            for cls in p["imperial_colonial_sq"][:14]:
                _place_squadron(s, cls, "imperial", "imperial_rimward",
                                colonial=True)
            for cls in p["imperial_colonial_tr"][:14]:
                _place_troop(s, cls, "imperial", "imperial_rimward")
            s.note("Imperial colonial reinforcements available")
        if s.turn >= 10:
            die = self.d6()
            battle, cruiser, fleets = oob.IMPERIAL_REINFORCEMENT_TABLE[die]
            if die == 6 and p.get("imperial_black_globes"):
                # rule 9: the first 6 brings the black globe squadrons and one
                # fleet *instead of* the units a 6 usually gives, not as well
                for cls in p["imperial_black_globes"]:
                    _place_squadron(s, cls, "imperial",
                                    "imperial_reinforcements")
                p["imperial_black_globes"] = []
                battle, cruiser, fleets = 0, 0, 1
                s.note("black globe squadrons join the Imperial order of battle")
            for _ in range(battle):
                if p["imperial_battle"]:
                    _place_squadron(s, p["imperial_battle"].pop(), "imperial",
                                    "imperial_reinforcements")
            for _ in range(cruiser):
                if p["imperial_cruiser"]:
                    _place_squadron(s, p["imperial_cruiser"].pop(), "imperial",
                                    "imperial_reinforcements")
            # "whenever a fleet appears as a reinforcement, an admiral is
            # selected at random from the admirals group and one counter is
            # selected at random from the special group"
            for _ in range(fleets):
                self._release_fleet_markers("imperial", 1)
                self._draw_admirals("imperial_admirals",
                                    "imperial_reinforcements", 1)
                self._draw_special(1)
            if s.turn == 10:
                for key in ("imperial_troops_5c", "imperial_troops_1c",
                            "imperial_troops_20"):
                    for cls in p[key]:
                        _place_troop(s, cls, "imperial",
                                     "imperial_reinforcements")
                    p[key] = []
        if s.turn in (10, 20):
            counts = ((13, 7, 1) if s.turn == 10 else (10, 7, 1))
            for _ in range(counts[0]):
                if p["zhodani_battle"]:
                    _place_squadron(s, p["zhodani_battle"].pop(), "zhodani",
                                    "zhodani_reinforcements")
            for _ in range(counts[1]):
                if p["zhodani_cruiser"]:
                    _place_squadron(s, p["zhodani_cruiser"].pop(), "zhodani",
                                    "zhodani_reinforcements")
            for _ in range(counts[2]):
                if p["zhodani_assault"]:
                    _place_squadron(s, p["zhodani_assault"].pop(), "zhodani",
                                    "zhodani_reinforcements")
            for key in ("zhodani_troops_5c", "zhodani_troops_1c",
                        "zhodani_troops_20", "zhodani_troops_10"):
                if p[key]:
                    _place_troop(s, p[key].pop(), "zhodani",
                                 "zhodani_reinforcements")
            for adm in p["zhodani_admirals"][:4 if s.turn == 10 else 3]:
                if not adm.location:
                    adm.location = "zhodani_reinforcements"
            # the two Zhodani groups bring the markers the opening deployment
            # did not, so the reinforcing squadrons have something to fly in
            self._release_fleet_markers("zhodani", 2 if s.turn == 10 else 1)
            s.note("Zhodani turn %d reinforcements arrive" % s.turn)

    # ---- phase 2 ---------------------------------------------------------
    def movement_phase(self) -> None:
        s = self.state
        first = IMPERIAL if self.d6() <= 3 else ZHODANI
        order = [first, s.enemy_of(first)]
        for side in order:                               # embark troops
            self.agents[side].load_troops(s, side, self)
        for side in order:                               # B. xboat movement
            self.agents[side].move_admirals(s, side, self)
        for side in order:                               # C. scouts
            self._move_scouts(side)
        for side in order:                               # D. well-led fleets
            self._move_well_led(side)
        self._move_plotted()                             # E. plotted movement
        resolve_refuelling(s)

    def _move_scouts(self, side: str) -> None:
        s = self.state
        scouts = [sq for sq in s.squadrons.values()
                  if sq.side == side and sq.cls.kind == "scout"
                  and sq.fleet is None and sq.refuelled]
        moves = self.agents[side].move_scouts(s, side, scouts, self)
        for uid, dest in moves.items():
            sq = s.squadrons.get(uid)
            if sq is None or dest == sq.location:
                continue
            if self._legal_jump(sq.location, dest, sq.jump):
                self._jump_squadron(sq, dest)

    def _move_well_led(self, side: str) -> None:
        s = self.state
        fleets = [f for f in s.fleets.values()
                  if f.active and f.side == side
                  and fleet_plotting_factor(s, f) == 0 and fleet_ready(s, f)]
        moves = self.agents[side].move_well_led_fleets(s, side, fleets, self)
        for uid, dest in moves.items():
            fleet = s.fleets.get(uid)
            if fleet is None or dest == fleet.location:
                continue
            if self._legal_jump(fleet.location, dest, fleet_jump(s, fleet)):
                self._jump_fleet(fleet, dest)

    def _move_plotted(self) -> None:
        s = self.state
        for fleet in list(s.fleets.values()):
            if not fleet.active or fleet_plotting_factor(s, fleet) == 0:
                continue
            order = fleet.plot.pop(s.turn, None)
            if order is None:
                continue
            kind, dest = order
            if kind != "jump" or dest == fleet.location:
                continue
            if not fleet_ready(s, fleet):
                continue
            if self._legal_jump(fleet.location, dest, fleet_jump(s, fleet)):
                self._jump_fleet(fleet, dest)
            else:
                self.abort_plot(fleet)

    def _legal_jump(self, origin: str, dest: str, jump: int) -> bool:
        s = self.state
        if jump <= 0 or dest == origin:
            return False
        if dest in REINFORCEMENT_BOXES:
            entry = s.world_map.entry_hexes.get(dest)
            return entry is not None and (origin == entry
                                          or hexmap.distance(origin, entry) + 1 <= jump)
        if dest not in s.world_map.worlds:
            return False                            # may only jump to systems
        if origin in REINFORCEMENT_BOXES:
            entry = s.world_map.entry_hexes.get(origin)
            if entry is None:
                return False
            return hexmap.distance(entry, dest) + 1 <= jump
        return hexmap.distance(origin, dest) <= jump

    def _jump_squadron(self, sq: Squadron, dest: str) -> None:
        sq.location = dest
        sq.refuelled = False
        sq.refuel_progress = 0
        for uid in sq.troops:
            if uid in self.state.troops:
                self.state.troops[uid].location = dest
        self._lift_red_zone(sq, dest)

    def _jump_fleet(self, fleet: Fleet, dest: str) -> None:
        fleet.location = dest
        for uid in list(fleet.squadrons):
            sq = self.state.squadrons.get(uid)
            if sq is not None:
                self._jump_squadron(sq, dest)
        if fleet.admiral is not None:
            adm = self.state.admirals.get(fleet.admiral)
            if adm is not None:
                adm.location = dest

    def _lift_red_zone(self, sq: Squadron, dest: str) -> None:
        world = self.state.world_map.get(dest)
        if world is not None and world.zone == "red" and sq.side == ZHODANI:
            world.zone_lifted = True

    # ---- phase 3 ---------------------------------------------------------
    def combat_phase(self) -> None:
        s = self.state
        hexes = sorted({sq.location for sq in s.squadrons.values()}
                       | {t.location for t in s.troops.values()})
        for hex_id in hexes:
            if hex_id in REINFORCEMENT_BOXES:
                continue
            self.resolve_hex(hex_id)

    def resolve_hex(self, hex_id: str) -> None:
        s = self.state
        world = s.world_map.get(hex_id)
        imp = s.squadrons_at(hex_id, IMPERIAL)
        zho = s.squadrons_at(hex_id, ZHODANI)
        black_globe_used = set()
        if s.options.get("black_globes") and imp and zho:
            black_globe_used = self._black_globe_step(hex_id)
        if imp and zho:
            self.space_combat(hex_id)
        self.interface_combat(hex_id, black_globe_used)
        self.surface_combat(hex_id)
        if world is not None:
            self._check_surrender(world)
            s.update_control(hex_id)

    # -- black globes ------------------------------------------------------
    def _black_globe_step(self, hex_id: str) -> set[int]:
        s = self.state
        used = set()
        for fleet in s.fleets_at(hex_id, IMPERIAL):
            squadrons = [s.squadrons[u] for u in fleet.squadrons
                         if u in s.squadrons]
            if not squadrons or not all(sq.black_globe for sq in squadrons):
                continue
            enemy = s.squadrons_at(hex_id, ZHODANI)
            if not enemy:
                continue
            total = sum(sq.attack for sq in squadrons)
            damage = tables.space_result(self.d6(), total)
            self._apply_battle_damage(hex_id, ZHODANI, damage)
            s.note("black globe surprise attack at %s for %d" % (hex_id, damage))
            used.update(sq.uid for sq in squadrons)
        return used

    # -- space combat ------------------------------------------------------
    def space_combat(self, hex_id: str) -> None:
        s = self.state
        for _ in range(MAX_COMBAT_ROUNDS):
            imp = s.squadrons_at(hex_id, IMPERIAL)
            zho = s.squadrons_at(hex_id, ZHODANI)
            if not imp or not zho:
                return
            tac = {side: s.tactical_factor(hex_id, side)
                   for side in (IMPERIAL, ZHODANI)}
            results = {}
            for side, own, foe in ((IMPERIAL, imp, zho), (ZHODANI, zho, imp)):
                total = sum(sq.attack for sq in own)
                die = self.d6()
                if tac[side] > tac[s.enemy_of(side)]:
                    die += 1
                elif tac[side] < tac[s.enemy_of(side)]:
                    die -= 1
                shift = self._quality_shift(side, own, foe) \
                    if s.options.get("squadron_quality") else 0
                results[side] = tables.space_result(min(6, max(1, die)),
                                                    total, shift)
            for side, damage in results.items():
                if damage:
                    self._apply_battle_damage(hex_id, s.enemy_of(side), damage)
            imp = s.squadrons_at(hex_id, IMPERIAL)
            zho = s.squadrons_at(hex_id, ZHODANI)
            if not imp or not zho:
                return
            # disengagement, lower tactical ability decides first
            order = sorted((IMPERIAL, ZHODANI), key=lambda x: tac[x])
            disengaged = False
            for side in order:
                if self.agents[side].wants_to_disengage(s, side, hex_id, self):
                    if self._disengage(hex_id, side):
                        disengaged = True
            if disengaged and not (s.squadrons_at(hex_id, IMPERIAL)
                                   and s.squadrons_at(hex_id, ZHODANI)):
                return

    def _quality_shift(self, side: str, own: list[Squadron],
                       foe: list[Squadron]) -> int:
        def quality(squadrons: list[Squadron], navy: str) -> str:
            total = sum(sq.attack for sq in squadrons)
            regular = sum(sq.attack for sq in squadrons
                          if sq.navy == navy and not sq.colonial)
            return "standard" if total and regular * 2 >= total else "low"
        own_q = "%s_%s" % (side, quality(own, "imperial" if side == IMPERIAL
                                         else "zhodani"))
        foe_side = self.state.enemy_of(side)
        foe_q = "%s_%s" % (foe_side, quality(foe, "imperial" if foe_side == IMPERIAL
                                             else "zhodani"))
        return tables.QUALITY_SHIFT.get((own_q, foe_q), 0)

    def _apply_battle_damage(self, hex_id: str, side: str, damage: int) -> None:
        """Remove at least ``damage`` defence factors from ``side``."""
        s = self.state
        remaining = damage
        while remaining > 0:
            targets = s.squadrons_at(hex_id, side)
            if not targets:
                return
            sq = self.agents[side].choose_loss(s, side, targets, remaining, self)
            # every squadron is worth at least one defence factor, so this
            # always terminates even if an agent returns an odd choice
            remaining -= max(1, sq.damage_value())
            if sq.reduced or sq.cls.reduced is None:
                for uid in list(sq.troops):
                    if uid in s.troops:
                        s.troops[uid].losses = 100
                self._remove_squadron(sq)
            else:
                sq.reduced = True
                self._trim_cargo(sq)

    def _trim_cargo(self, sq: Squadron) -> None:
        s = self.state
        capacity = sq.capacity
        carried = sum(s.troops[u].current for u in sq.troops if u in s.troops)
        for uid in list(sq.troops):
            if carried <= capacity:
                break
            troop = s.troops.get(uid)
            if troop is None:
                continue
            carried -= troop.current
            troop.losses = 100
            sq.troops.remove(uid)

    def _remove_squadron(self, sq: Squadron) -> None:
        s = self.state
        if sq.fleet is not None:
            fleet = s.fleets.get(sq.fleet)
            if fleet and sq.uid in fleet.squadrons:
                fleet.squadrons.remove(sq.uid)
            if fleet and not fleet.squadrons:
                self._disband(fleet)
        s.squadrons.pop(sq.uid, None)

    def _disband(self, fleet: Fleet) -> None:
        fleet.active = False
        fleet.plot.clear()
        if fleet.admiral is not None:
            adm = self.state.admirals.get(fleet.admiral)
            if adm is not None:
                adm.fleet = None
                adm.aboard = False
            fleet.admiral = None
        fleet.squadrons.clear()

    def _disengage(self, hex_id: str, side: str) -> bool:
        """Jump every squadron that can leave; returns True if any left."""
        s = self.state
        moved = False
        for fleet in list(s.fleets_at(hex_id, side)):
            able = [s.squadrons[u] for u in fleet.squadrons
                    if u in s.squadrons and s.squadrons[u].refuelled]
            if not able:
                continue
            jump = min(sq.jump for sq in able)
            dest = self.agents[side].choose_retreat(s, side, hex_id, jump, self)
            if dest is None:
                continue
            for sq in list(s.squadrons[u] for u in fleet.squadrons
                           if u in s.squadrons):
                if not sq.refuelled:
                    fleet.squadrons.remove(sq.uid)
                    sq.fleet = None
            self._jump_fleet(fleet, dest)
            self.abort_plot(fleet)
            moved = True
        for sq in list(s.squadrons_at(hex_id, side)):
            if sq.fleet is not None or not sq.refuelled:
                continue
            if sq.cls.kind != "scout":
                continue
            dest = self.agents[side].choose_retreat(s, side, hex_id, sq.jump, self)
            if dest:
                self._jump_squadron(sq, dest)
                moved = True
        return moved

    def abort_plot(self, fleet: Fleet) -> None:
        for turn in list(fleet.plot):
            fleet.plot[turn] = ("hold", fleet.location)

    # -- interface combat --------------------------------------------------
    def interface_combat(self, hex_id: str, black_globe_used: set[int]) -> None:
        s = self.state
        world = s.world_map.get(hex_id)
        if world is None:
            return
        defender = world.control
        attackers = {side: s.squadrons_at(hex_id, side)
                     for side in (IMPERIAL, ZHODANI)}
        sdb_active = world.sdb_current > 0 and defender is not None
        engaged_out = set()

        if sdb_active:
            for side, squadrons in attackers.items():
                if side == defender or not squadrons:
                    continue
                attack = self.agents[side].attack_sdbs(s, side, hex_id, self)
                passive = self.agents[defender].sdbs_passive(s, defender,
                                                            hex_id, self)
                if not attack:
                    engaged_out.add(side)
                    if not passive:
                        self._sdb_fire(hex_id, world, defender, side)
                    continue
                rounds = 1 if passive else MAX_COMBAT_ROUNDS
                for _ in range(rounds):
                    bombard = sum(sq.bombard for sq in s.squadrons_at(hex_id, side))
                    if bombard <= 0 or world.sdb_current <= 0:
                        break
                    loss = tables.squadron_vs_sdb_result(
                        self.d6(), bombard, world.sdb, world.tech_level, passive)
                    world.sdb_losses = min(100, world.sdb_losses + loss)
                    if not passive and world.sdb_current > 0:
                        self._sdb_fire(hex_id, world, defender, side)
                    if world.sdb_current <= 0:
                        break
                    if self.agents[side].break_off_sdb_action(s, side, hex_id, self):
                        engaged_out.add(side)
                        break
                if world.sdb_current > 0 and passive:
                    world.sdb_losses = min(100, world.sdb_losses)

        # surface bombing: squadrons may not bomb while enemy SDBs are active,
        # and SDBs bomb at one tenth of their current strength
        for side in (IMPERIAL, ZHODANI):
            if side in engaged_out:
                continue
            bombard = 0
            if world.sdb_current == 0 or defender is None or defender == side:
                bombard += sum(sq.bombard for sq in s.squadrons_at(hex_id, side))
            if defender == side and world.sdb_current > 0:
                bombard += world.sdb_current // 10
            if bombard <= 0:
                continue
            targets = self.agents[side].bombing_targets(s, side, hex_id,
                                                        bombard, self)
            for target, factors in targets:
                if factors <= 0:
                    continue
                if isinstance(target, TroopUnit):
                    loss = tables.surface_bombing_result(
                        self.d6(), factors, target.tech_level(world))
                    target.losses = min(100, target.losses + loss)
                elif target == "defense" and world.defense_current > 0:
                    loss = tables.surface_bombing_result(
                        self.d6(), factors, world.tech_level)
                    world.defense_losses = min(100, world.defense_losses + loss)

        # space-surface transfers
        for side in (IMPERIAL, ZHODANI):
            if side in engaged_out:
                continue
            blocked = (world.sdb_current > 0 and defender is not None
                       and defender != side)
            self.agents[side].transfer_troops(s, side, hex_id, blocked, self)

    def _sdb_fire(self, hex_id: str, world: World, defender: str,
                  target_side: str) -> None:
        damage = tables.sdb_vs_squadron_result(self.d6(), world.sdb_current,
                                               world.tech_level)
        if damage:
            self._apply_battle_damage(hex_id, target_side, damage)

    # -- surface combat ----------------------------------------------------
    def surface_combat(self, hex_id: str) -> None:
        s = self.state
        world = s.world_map.get(hex_id)
        if world is None:
            return
        sides = {}
        for side in (IMPERIAL, ZHODANI):
            units = [t for t in s.troops_at(hex_id, side)
                     if not (t.guerrilla and not t.overt)]
            sides[side] = units
        defender = world.control
        has_defence = defender is not None and world.defense_current > 0
        if not (sides[IMPERIAL] and (sides[ZHODANI] or
                                     (has_defence and defender == ZHODANI))) \
           and not (sides[ZHODANI] and (sides[IMPERIAL] or
                                        (has_defence and defender == IMPERIAL))):
            return

        pending: list[tuple[object, int]] = []

        def resolve_fire(side: str, first_fire: bool) -> list[tuple[object, int]]:
            shooters = [t for t in sides[side] if not t.cls.psionic or first_fire]
            if first_fire:
                shooters = [t for t in sides[side] if t.cls.psionic]
            allocation = self.agents[side].surface_fire(
                s, side, hex_id, shooters, self)
            out = []
            for target, factors, tech in allocation:
                if factors <= 0:
                    continue
                if isinstance(target, TroopUnit):
                    strength = target.combat_strength(world)
                    target_tech = target.tech_level(world)
                else:
                    strength = float(world.defense_current)
                    target_tech = world.tech_level
                if strength <= 0:
                    continue
                loss = tables.troop_result(self.d66(), factors, strength,
                                           tech - target_tech, world.atmosphere)
                out.append((target, loss))
            return out

        # psionic Guards fire first and their casualties are applied at once
        for side in (IMPERIAL, ZHODANI):
            if any(t.cls.psionic for t in sides[side]):
                for target, loss in resolve_fire(side, True):
                    self._apply_surface_loss(world, target, loss)
        for side in (IMPERIAL, ZHODANI):
            sides[side] = [t for t in s.troops_at(hex_id, side)
                           if not (t.guerrilla and not t.overt)]
            pending.extend(resolve_fire(side, False))
        for target, loss in pending:
            self._apply_surface_loss(world, target, loss)

    def _apply_surface_loss(self, world: World, target, loss: int) -> None:
        if loss <= 0:
            return
        if isinstance(target, TroopUnit):
            target.losses = min(100, target.losses + loss)
            if target.losses >= 100 and not target.guerrilla:
                self.state.troops.pop(target.uid, None)
        else:
            world.defense_losses = min(100, world.defense_losses + loss)

    def _check_surrender(self, world: World) -> None:
        """Rule 5: an airless world surrenders to an unopposed warship."""
        if world.atmosphere != "vacuum" or world.control is None:
            return
        s = self.state
        holder = world.control
        if s.squadrons_at(world.hex, holder) or world.sdb_current > 0:
            return
        enemy = s.enemy_of(holder)
        if not any(sq.attack > 0 for sq in s.squadrons_at(world.hex, enemy)):
            return
        world.defense_losses = 100
        for troop in s.troops_at(world.hex, holder):
            troop.losses = 100
            if not troop.guerrilla:
                s.troops.pop(troop.uid, None)
        world.control = None
        s.note("%s surrenders" % world.name)

    # ---- phase 4 ---------------------------------------------------------
    def reorganisation_phase(self) -> None:
        s = self.state
        for side in (IMPERIAL, ZHODANI):
            self.agents[side].fleet_adjustment(s, side, self)
        self._guerrilla_step()
        for side in (IMPERIAL, ZHODANI):
            self._plot_for(side)
        for world in s.world_map:
            s.update_control(world.hex)

    def _guerrilla_step(self) -> None:
        s = self.state
        guerrillas = [t for t in s.troops.values() if t.guerrilla]
        for unit in guerrillas:
            if unit.losses >= 100:
                s.troops.pop(unit.uid, None)
                continue
            if not unit.overt:
                unit.losses = max(0, unit.losses - 10)
        self.agents[ZHODANI].set_guerrilla_modes(s, self)

    def _plot_for(self, side: str) -> None:
        s = self.state
        for fleet in s.fleets.values():
            if not fleet.active or fleet.side != side:
                continue
            factor = fleet_plotting_factor(s, fleet)
            if factor == 0:
                fleet.plot.clear()
                continue
            for ahead in range(1, factor + 1):
                turn = s.turn + ahead
                if turn in fleet.plot:
                    continue
                order = self.agents[side].plot_fleet(s, side, fleet, turn, self)
                fleet.plot[turn] = order
            for turn in [t for t in fleet.plot if t > s.turn + factor]:
                del fleet.plot[turn]

    # ---- victory ---------------------------------------------------------
    def check_victory(self) -> None:
        s = self.state
        margin = s.victory_margin()
        if margin >= 301 or margin <= -301:
            s.game_over = True
            s.result = s.victory_level(margin)
            s.note("automatic victory: %s" % s.result)
            return
        if s.turn >= 26 and self.agents[ZHODANI].declare_armistice(s, self):
            shift = 2 if s.turn <= 51 else 1
            levels = ["imperial automatic victory", "imperial decisive victory",
                      "imperial major victory", "imperial marginal victory",
                      "stalemate", "zhodani marginal victory",
                      "zhodani major victory", "zhodani decisive victory",
                      "zhodani automatic victory"]
            idx = levels.index(s.victory_level(margin))
            s.result = levels[max(0, idx - shift)]
            s.game_over = True
            s.note("Zhodani armistice: %s" % s.result)
            return
        if s.turn >= MAX_TURNS:
            s.game_over = True
            s.result = s.victory_level(margin)
            s.note("game ends at turn limit: %s" % s.result)


def play(state: GameState, imperial_agent, zhodani_agent,
         max_turns: int = MAX_TURNS, on_turn=None) -> str:
    """Play a whole game and return the victory level."""
    engine = Engine(state, imperial_agent, zhodani_agent)
    imperial_agent.begin_game(state, IMPERIAL, engine)
    zhodani_agent.begin_game(state, ZHODANI, engine)
    engine.initial_plotting()
    while not state.game_over and state.turn <= max_turns:
        engine.play_turn()
        if on_turn is not None:
            on_turn(state)
    if state.result is None:
        state.result = state.victory_level()
    return state.result

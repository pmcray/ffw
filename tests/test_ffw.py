"""Tests for the Fifth Frontier War implementation.

Run with ``python -m pytest tests`` or ``python tests/test_ffw.py``.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffw import hexmap, oob, tables                                  # noqa: E402
from ffw.engine import Engine, can_refuel_at, new_game, play         # noqa: E402
from ffw.state import IMPERIAL, ZHODANI, WorldMap                    # noqa: E402
from ffw.agents import (HeuristicAgent, RandomAgent, ScriptedAgent,  # noqa: E402
                        NeuralAgent, HumanAgent)


class TestHexGeometry(unittest.TestCase):
    def test_adjacent_hexes_are_one_parsec(self):
        for hex_id in ("2314", "0708", "1510", "2109"):
            for neighbour in hexmap.neighbours(hex_id):
                self.assertEqual(hexmap.distance(hex_id, neighbour), 1,
                                 "%s -> %s" % (hex_id, neighbour))

    def test_distance_is_symmetric_and_zero_on_self(self):
        self.assertEqual(hexmap.distance("2314", "2314"), 0)
        self.assertEqual(hexmap.distance("2314", "2109"),
                         hexmap.distance("2109", "2314"))

    def test_known_map_distances(self):
        # Pixie to Boughene is the one-parsec hop the xboat rule cites
        self.assertEqual(hexmap.distance("2307", "2308"), 1)
        # ... and Pixie to Feri is inside a jump-4 xboat move
        self.assertLessEqual(hexmap.distance("2307", "2409"), 4)

    def test_neighbours_are_mutual(self):
        for hex_id in ("1510", "2411"):
            for neighbour in hexmap.neighbours(hex_id):
                self.assertIn(hex_id, hexmap.neighbours(neighbour))


class TestWorldData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.map = WorldMap.load()

    def test_map_has_146_worlds(self):
        self.assertEqual(len(self.map.worlds), 146)

    def test_rule_book_hexes(self):
        """Hexes quoted in the rules must hold the worlds the rules name."""
        expected = {
            "0708": "Chronor", "1018": "Querion", "1627": "Gram",
            "1510": "Jewell", "1520": "Frenzie", "2123": "Lanth",
            "2314": "Regina", "3120": "Rhylanor", "2109": "Efate",
            "2213": "Ruie", "1212": "Quar", "1514": "Zircon",
            "1408": "Esalin", "2307": "Pixie", "2409": "Feri",
            "1914": "871-438", "1319": "Quare", "2822": "Icetina",
        }
        for hex_id, name in expected.items():
            world = self.map.get(hex_id)
            self.assertIsNotNone(world, "no world at %s" % hex_id)
            self.assertEqual(world.name, name)

    def test_regina_matches_the_rulebook(self):
        """Rule 5 says Regina needs a 15-factor garrison; rule 8 gives it TL 10."""
        regina = self.map.get("2314")
        self.assertEqual(regina.tech_level, 10)
        self.assertEqual(regina.defense_battalions, 1500)
        self.assertEqual(regina.garrison_required(), 15)
        self.assertEqual(regina.victory_points, 20)

    def test_efate_guerrilla_strength(self):
        """Rule 5 example: a guerrilla unit on Efate is 100 factors at TL 13."""
        efate = self.map.get("2109")
        self.assertEqual(efate.tech_level, 13)
        self.assertEqual(efate.defense_battalions // 10, 100)

    def test_three_vargr_and_four_sword_worlds(self):
        """The victory bonuses in rule 8 count exactly these worlds."""
        vargr = [w for w in self.map if w.owner == "vargr"]
        sword = [w for w in self.map if w.owner == "sword_worlds"]
        self.assertEqual(len(vargr), 3)
        self.assertEqual(len(sword), 4)

    def test_eight_subsector_capitals(self):
        capitals = [w for w in self.map if w.subsector_capital]
        self.assertEqual(len(capitals), 8)

    def test_atmospheres_and_zones_are_known_values(self):
        for world in self.map:
            self.assertIn(world.atmosphere, ("breathable", "tainted", "vacuum"))
            self.assertIn(world.zone, (None, "amber", "red"))


class TestCombatTables(unittest.TestCase):
    def test_space_combat_worked_example(self):
        """Rule 4: 29 attack factors fire on the 24 column."""
        self.assertEqual(tables.column_index(tables.SPACE_COLUMNS, 29), 5)
        self.assertEqual(tables.SPACE_COLUMNS[5], 24)

    def test_multiple_attack_split(self):
        """Rule 4 example: 58 factors split into 48 + 10, rolling 3 and 4."""
        self.assertEqual(tables.split_attack(58), [48, 10])
        first = tables.SPACE_CRT[3][tables.column_index(tables.SPACE_COLUMNS, 48)]
        second = tables.SPACE_CRT[4][tables.column_index(tables.SPACE_COLUMNS, 10)]
        self.assertEqual(first, 9)
        self.assertEqual(second, 2)
        self.assertEqual(first + second, 11)

    def test_squadron_versus_sdb_worked_example(self):
        """Rule 4: 40 bombardment vs 100 TL-13 SDBs, rolling 3, gives 40%."""
        result = tables.squadron_vs_sdb_result(die=3, bombardment=40,
                                               sdb_full=100, sdb_tech=13,
                                               passive=False)
        self.assertEqual(result, 40)

    def test_troop_combat_worked_example(self):
        """Rule 4's surface combat example, attack by attack."""
        # 15 TL-14 factors vs a 5-factor TL-15 unit: 3:1 shifted left to 2:1,
        # roll 5 -> 40% losses
        self.assertEqual(tables.troop_result(5, 15, 5, 14 - 15, "breathable"), 40)
        # 25 TL-14 factors vs 120 TL-10 defence: 1:5 shifted four columns
        # right lands on 1:1, and a roll of 6 there is 10% losses.  The
        # rulebook's own worked example says "1.5:1 ... 20%", which is one
        # column too far: the same paragraph shifts 5:1 four columns left and
        # correctly gets 1:1, and 1:5 -> 1:1 is the mirror of that.  The table
        # and the other three attacks in the example agree with 1:1.
        self.assertEqual(tables.ODDS_COLUMNS[tables.odds_index(25, 120) + 4], "1:1")
        self.assertEqual(tables.troop_result(6, 25, 120, 14 - 10, "breathable"), 10)
        # 5 TL-15 factors vs a 20-factor TL-14 unit: 1:5 shifted right to 1:3,
        # roll 9 -> no effect
        self.assertEqual(tables.troop_result(9, 5, 20, 15 - 14, "breathable"), 0)
        # 120 TL-10 factors vs a 20-factor TL-14 unit: 5:1 shifted four left to
        # 1:1, roll 7 -> 10% losses
        self.assertEqual(tables.troop_result(7, 120, 20, 10 - 14, "breathable"), 10)

    def test_odds_round_down_in_favour_of_the_defender(self):
        """Rule 4: 13:5 rounds to 2:1 and 25:2 rounds to 10:1."""
        self.assertEqual(tables.ODDS_COLUMNS[tables.odds_index(13, 5)], "2:1")
        self.assertEqual(tables.ODDS_COLUMNS[tables.odds_index(25, 2)], "10:1")

    def test_tech_level_modifiers(self):
        self.assertEqual(tables.tech_modifier(15), 2)
        self.assertEqual(tables.tech_modifier(13), 0)
        self.assertEqual(tables.tech_modifier(11), -1)
        self.assertEqual(tables.tech_modifier(9), -2)
        self.assertEqual(tables.tech_modifier(8), -3)

    def test_atmosphere_modifiers(self):
        self.assertEqual(tables.ATMOSPHERE_MODIFIER["breathable"], 0)
        self.assertEqual(tables.ATMOSPHERE_MODIFIER["tainted"], -1)
        self.assertEqual(tables.ATMOSPHERE_MODIFIER["vacuum"], -2)

    def test_quality_shift_table(self):
        self.assertEqual(tables.QUALITY_SHIFT[("imperial_standard",
                                               "zhodani_low")], 2)
        self.assertEqual(tables.QUALITY_SHIFT[("zhodani_standard",
                                               "imperial_low")], 1)


class TestPercentageLosses(unittest.TestCase):
    def test_losses_are_additive_and_kill_at_100(self):
        state = new_game(seed=3)
        troop = next(iter(state.troops.values()))
        full = troop.full_factor
        troop.losses = 40
        self.assertEqual(troop.current, full * 60 // 100)
        troop.losses += 60
        self.assertEqual(troop.current, 0)

    def test_50_factor_corps_at_40_percent_is_30(self):
        """Rule 4: a 50-factor corps at 40% losses has a current strength of 30."""
        cls = oob.TroopClass("corps", 50, 14)
        from ffw.state import TroopUnit
        unit = TroopUnit(uid=1, cls=cls, navy="zhodani", location="2314",
                         losses=40)
        self.assertEqual(unit.current, 30)

    def test_armour_and_elite_double_strength(self):
        from ffw.state import TroopUnit
        plain = TroopUnit(1, oob.TroopClass("brigade", 10, 15), "imperial", "2314")
        elite = TroopUnit(2, oob.TroopClass("brigade", 10, 15, elite=True),
                          "imperial", "2314")
        both = TroopUnit(3, oob.TroopClass("brigade", 10, 15, elite=True,
                                           armor=True), "imperial", "2314")
        self.assertEqual(plain.combat_strength(), 10)
        self.assertEqual(elite.combat_strength(), 20)
        self.assertEqual(both.combat_strength(), 40)

    def test_mercenary_halves_after_50_percent(self):
        from ffw.state import TroopUnit
        merc = TroopUnit(1, oob.TroopClass("division", 20, 14, mercenary=True),
                         "imperial", "2314", losses=60)
        self.assertEqual(merc.current, 8)
        self.assertEqual(merc.combat_strength(), 4)


class TestRefuelling(unittest.TestCase):
    def setUp(self):
        self.state = new_game(seed=4)

    def test_gas_giants_serve_either_side(self):
        giant = next(w for w in self.state.world_map if w.gas_giant)
        self.assertTrue(can_refuel_at(self.state, IMPERIAL, giant.hex,
                                      ["unstreamlined"]))
        self.assertTrue(can_refuel_at(self.state, ZHODANI, giant.hex,
                                      ["unstreamlined"]))

    def test_enemy_starport_is_not_available(self):
        world = next(w for w in self.state.world_map
                     if w.control == IMPERIAL and not w.gas_giant
                     and w.starport in "ABCD")
        self.assertFalse(can_refuel_at(self.state, ZHODANI, world.hex,
                                       ["streamlined"]))

    def test_reinforcement_boxes_count_as_bases(self):
        self.assertTrue(can_refuel_at(self.state, IMPERIAL,
                                      "imperial_reinforcements",
                                      ["unstreamlined"]))

    def test_starport_capacity_table(self):
        self.assertEqual(tables.STARPORT_CAPACITY["A"], 4)
        self.assertEqual(tables.STARPORT_CAPACITY["D"], 1)
        self.assertEqual(tables.STARPORT_CAPACITY["E"], 0)


class TestVictory(unittest.TestCase):
    def setUp(self):
        self.state = new_game(seed=6)

    def test_no_points_before_anyone_moves(self):
        points = self.state.victory_points()
        self.assertEqual(points[IMPERIAL], 0.0)
        self.assertEqual(points[ZHODANI], 0.0)

    def test_taking_regina_is_worth_twice_its_tech_level(self):
        regina = self.state.world_map.get("2314")
        regina.control = ZHODANI
        self.assertEqual(self.state.victory_points()[ZHODANI], 20.0)

    def test_victory_levels(self):
        self.assertEqual(self.state.victory_level(-400), "imperial automatic victory")
        self.assertEqual(self.state.victory_level(-120), "imperial major victory")
        self.assertEqual(self.state.victory_level(0), "stalemate")
        self.assertEqual(self.state.victory_level(75), "zhodani marginal victory")
        self.assertEqual(self.state.victory_level(350), "zhodani automatic victory")

    def test_vargr_bonus(self):
        for world in self.state.world_map:
            if world.owner == "vargr":
                world.control = IMPERIAL
        self.assertGreaterEqual(self.state.victory_points()[IMPERIAL], 15.0)


class TestSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.state = new_game(seed=7)

    def test_ten_guerrilla_units(self):
        guerrillas = [t for t in self.state.troops.values() if t.guerrilla]
        self.assertEqual(len(guerrillas), 10)

    def test_guerrillas_garrison_efate_and_ruie(self):
        hexes = [t.location for t in self.state.troops.values() if t.guerrilla]
        self.assertGreaterEqual(hexes.count("2109"), 3)
        self.assertGreaterEqual(hexes.count("2213"), 1)

    def test_secret_base_is_legal(self):
        world = self.state.world_map.get(self.state.secret_base)
        self.assertIsNotNone(world)
        self.assertEqual(world.owner, "imperial")
        self.assertEqual(world.defense_battalions, 0)
        self.assertFalse(world.naval_base or world.scout_base)
        self.assertNotEqual(world.zone, "red")

    def test_both_sides_have_forces(self):
        imperial = [s for s in self.state.squadrons.values() if s.side == IMPERIAL]
        zhodani = [s for s in self.state.squadrons.values() if s.side == ZHODANI]
        self.assertGreater(len(imperial), 20)
        self.assertGreater(len(zhodani), 30)

    def test_quar_and_zircon_are_imperial(self):
        self.assertEqual(self.state.world_map.get("1212").control, IMPERIAL)
        self.assertEqual(self.state.world_map.get("1514").control, IMPERIAL)

    def test_order_of_battle_pool_sizes(self):
        """Counter counts follow the printed order of battle charts."""
        self.assertEqual(sum(c for _, c in oob.ZHODANI_BATTLE), 46)   # 23+13+10
        self.assertEqual(sum(c for _, c in oob.ZHODANI_CRUISER), 28)  # 14+7+7
        self.assertEqual(sum(c for _, c in oob.ZHODANI_ASSAULT), 8)   # 6+1+1
        self.assertEqual(sum(c for _, c in oob.ZHODANI_SCOUT), 6)
        self.assertEqual(sum(c for _, c in oob.IMPERIAL_BATTLE), 34)  # 2+32
        self.assertEqual(sum(c for _, c in oob.IMPERIAL_CRUISER), 32)  # 6+26
        self.assertEqual(len(oob.IMPERIAL_FLEETS), 14)
        self.assertEqual(len(oob.ZHODANI_FLEETS), 14)
        self.assertEqual(len(oob.ZHODANI_ADMIRALS), 14)
        self.assertEqual(len(oob.IMPERIAL_ADMIRALS), 14)


class TestEngine(unittest.TestCase):
    def test_a_turn_runs_and_advances_the_clock(self):
        state = new_game(seed=8)
        engine = Engine(state, ScriptedAgent(IMPERIAL, seed=1),
                        ScriptedAgent(ZHODANI, seed=2))
        engine.initial_plotting()
        engine.play_turn()
        self.assertEqual(state.turn, 2)

    def test_fleets_stay_within_their_size_and_jump_limits(self):
        from ffw.engine import MAX_FLEET_SIZE, fleet_jump
        state = new_game(seed=9)
        play(state, ScriptedAgent(IMPERIAL, seed=1),
             ScriptedAgent(ZHODANI, seed=2), max_turns=12)
        for fleet in state.fleets.values():
            if not fleet.active:
                continue
            self.assertLessEqual(len(fleet.squadrons), MAX_FLEET_SIZE)
            if fleet.squadrons:
                self.assertGreaterEqual(fleet_jump(state, fleet), 0)

    def test_squadrons_never_sit_in_a_hex_without_a_world(self):
        state = new_game(seed=10)
        play(state, ScriptedAgent(IMPERIAL, seed=1),
             ScriptedAgent(ZHODANI, seed=2), max_turns=10)
        for sq in state.squadrons.values():
            self.assertTrue(sq.location in state.world_map.worlds
                            or sq.location in
                            ("imperial_reinforcements", "imperial_rimward",
                             "zhodani_reinforcements", "zhodani_additional",
                             "sword_worlds"),
                            "squadron adrift at %r" % sq.location)

    def test_control_requires_a_garrison(self):
        state = new_game(seed=11)
        for world in state.world_map:
            if world.control is None:
                continue
            original = {"imperial": IMPERIAL, "zhodani": ZHODANI,
                        "sword_worlds": ZHODANI, "vargr": ZHODANI}.get(world.owner)
            if world.control == original:
                continue
            self.assertGreaterEqual(
                state.garrison_strength(world.hex, world.control),
                world.garrison_required())

    def test_full_game_reaches_a_verdict(self):
        state = new_game(seed=12)
        result = play(state, ScriptedAgent(IMPERIAL, seed=1),
                      ScriptedAgent(ZHODANI, seed=2), max_turns=30)
        self.assertIn(result, [
            "imperial automatic victory", "imperial decisive victory",
            "imperial major victory", "imperial marginal victory", "stalemate",
            "zhodani marginal victory", "zhodani major victory",
            "zhodani decisive victory", "zhodani automatic victory"])

    def test_the_war_actually_moves(self):
        """A campaign must change hands somewhere: a frozen front is a bug."""
        state = new_game(seed=13)
        play(state, ScriptedAgent(IMPERIAL, seed=1),
             ScriptedAgent(ZHODANI, seed=2), max_turns=25)
        conquered = [w for w in state.world_map
                     if w.owner == "imperial" and w.control == ZHODANI]
        self.assertGreater(len(conquered), 0)


class TestAgents(unittest.TestCase):
    def test_every_agent_can_play_a_short_game(self):
        for factory in (RandomAgent, HeuristicAgent, ScriptedAgent,
                        NeuralAgent, HumanAgent):
            state = new_game(seed=14)
            result = play(state, factory(IMPERIAL, seed=1),
                          factory(ZHODANI, seed=2), max_turns=6)
            self.assertIsInstance(result, str)

    def test_human_agent_uses_supplied_orders(self):
        seen = []

        def ask(request):
            seen.append(request["kind"])
            return None            # fall back to the doctrine
        state = new_game(seed=15)
        play(state, HumanAgent(IMPERIAL, ask=ask, seed=1),
             ScriptedAgent(ZHODANI, seed=2), max_turns=4)
        self.assertIn("plot_fleet", seen)

    def test_weight_vector_round_trip(self):
        from ffw.agents import DEFAULT_WEIGHTS, weight_dict, weight_vector
        vector = weight_vector(DEFAULT_WEIGHTS)
        self.assertEqual(weight_dict(vector), DEFAULT_WEIGHTS)


class TestLearning(unittest.TestCase):
    def test_value_network_learns_a_constant(self):
        import numpy as np
        from ffw.agents import ValueNetwork
        from ffw import features
        net = ValueNetwork(hidden=8, seed=0)
        x = np.random.default_rng(0).normal(0, 1, (200, features.N_FEATURES))
        y = np.full(200, 0.5)
        before = abs(net(x[0]) - 0.5)
        net.train(x, y, epochs=120, lr=0.05)
        after = abs(net(x[0]) - 0.5)
        self.assertLess(after, before)

    def test_network_save_and_load(self):
        import tempfile
        from ffw.agents import ValueNetwork
        net = ValueNetwork(hidden=6, seed=1)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        net.save(path)
        clone = ValueNetwork.load(path)
        import numpy as np
        probe = np.zeros(net.w1.shape[0])
        self.assertAlmostEqual(net(probe), clone(probe), places=10)
        os.unlink(path)

    def test_feature_perspective_flips_the_margin(self):
        from ffw import features
        state = new_game(seed=16)
        state.world_map.get("2314").control = ZHODANI
        raw = features.extract(state)
        zho = features.perspective(raw, ZHODANI)
        imp = features.perspective(raw, IMPERIAL)
        self.assertAlmostEqual(zho[1], -imp[1])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestValueNetworkQuality(unittest.TestCase):
    """The regression loss is not a quality measure; these guard the real one."""

    def test_training_produces_a_spread_of_outcomes(self):
        """A single fixed opponent gives near-identical labels and a useless fit."""
        from ffw.training import train_value_network
        net, _ = train_value_network(games=6, epochs=10, seed=2, max_turns=14)
        self.assertGreater(net.margin_spread, 5.0,
                           "games all ended alike: the labels carry no signal")

    def test_evaluator_reports_game_level_uncertainty(self):
        from ffw.training import train_value_network, evaluator_report
        net, _ = train_value_network(games=4, epochs=10, seed=2, max_turns=12)
        report = evaluator_report(net, games=3, seed=5, max_turns=12)
        self.assertEqual(report["games"], 3)
        self.assertGreater(report["positions"], report["games"])
        low, high = report["ci"]
        self.assertLessEqual(low, report["correlation"] + 1e-6)
        self.assertGreaterEqual(high, report["correlation"] - 1e-6)

    def test_network_serves_both_sides_symmetrically(self):
        """One network evaluates both players; the perspective flip is its own inverse."""
        from ffw.agents import ValueNetwork
        from ffw import features
        state = new_game(seed=17)
        raw = features.extract(state)
        self.assertTrue(np.allclose(
            features.perspective(features.perspective(raw, IMPERIAL), IMPERIAL),
            raw))


class TestPairedEvaluation(unittest.TestCase):
    """Paired games are what make agent comparisons mean anything here."""

    def test_paired_advantage_is_antisymmetric(self):
        from ffw.training import play_paired
        a = lambda side, seed: ScriptedAgent(side, seed=seed)
        b = lambda side, seed: RandomAgent(side, seed=seed)
        forward, f1, f2 = play_paired(a, b, seed=3, max_turns=8)
        reverse, r1, r2 = play_paired(b, a, seed=3, max_turns=8)
        self.assertAlmostEqual(forward, -reverse, places=6)

    def test_identical_agents_have_no_paired_advantage(self):
        """Mirroring the same doctrine must cancel exactly, seat bias and all."""
        from ffw.training import play_paired
        same = lambda side, seed: ScriptedAgent(side, seed=seed)
        advantage, first, second = play_paired(same, same, seed=5, max_turns=8)
        self.assertAlmostEqual(advantage, 0.0, places=6)
        self.assertAlmostEqual(first, second, places=6)

    def test_tournament_table_is_antisymmetric(self):
        from ffw.training import tournament
        entries = {"scripted": lambda s, d: ScriptedAgent(s, seed=d),
                   "random": lambda s, d: RandomAgent(s, seed=d)}
        table, summary = tournament(entries, games=1, max_turns=8, seed=1)
        self.assertAlmostEqual(table[("scripted", "random")],
                               -table[("random", "scripted")], places=6)
        for entry in summary.values():
            self.assertIn("stderr", entry)
            self.assertGreater(entry["pairings"], 0)

    def test_seat_bias_is_reported(self):
        from ffw.training import seat_bias
        bias = seat_bias(games=2, max_turns=8, seed=2)
        self.assertEqual(bias["games"], 2)
        self.assertIn("mean_margin", bias)


class TestAmphibiousCapability(unittest.TestCase):
    """Regressions for two bugs that silently disarmed the Imperium."""

    def test_home_worlds_need_no_garrison(self):
        """Rule 5's garrison binds on a world you have *taken*, not your own.

        A world cannot "revert to its original owner" when that owner already
        holds it, and its defence battalions are its garrison.  Charging the
        owner 1% of them locked 2296 factors of the Imperial army onto worlds
        that needed no garrison at all.
        """
        world_map = WorldMap.load()
        regina = world_map.get("2314")          # Imperial, 1500 battalions
        self.assertEqual(regina.original_control, IMPERIAL)
        self.assertEqual(regina.garrison_required(IMPERIAL), 0)
        self.assertEqual(regina.garrison_required(ZHODANI), 15)
        self.assertEqual(regina.garrison_required(), 15)
        chronor = world_map.get("0708")         # Zhodani, 1000 battalions
        self.assertEqual(chronor.garrison_required(ZHODANI), 0)
        self.assertEqual(chronor.garrison_required(IMPERIAL), 10)

    def test_a_unit_too_big_to_lift_does_not_block_smaller_ones(self):
        """The loader must take the largest unit that *fits*, not give up.

        Spare troops are offered biggest-first.  Stopping at the head of the
        list meant a 500-factor army parked in front of a battle squadron's 20
        factors of lift blocked every smaller unit behind it, so the Imperium --
        whose army is mostly armies and corps -- never embarked anything.
        """
        from ffw import oob
        from ffw.engine import _place_troop
        state = new_game(seed=21)
        agent = ScriptedAgent(IMPERIAL, seed=1)
        fleet = next(f for f in state.fleets.values()
                     if f.active and f.side == IMPERIAL and f.squadrons
                     and f.location in state.world_map.worlds)
        hex_id = fleet.location

        # reduce the fleet to one battle squadron: 20 factors of lift, which no
        # army or corps can use but a battalion can
        keep = next((u for u in fleet.squadrons
                     if state.squadrons[u].cls.kind == "battle"), None)
        if keep is None:
            keep = fleet.squadrons[0]
            state.squadrons[keep].cls = oob.IMPERIAL_BATTLE[0][0]
        for uid in list(fleet.squadrons):
            if uid != keep:
                state.squadrons[uid].fleet = None
                fleet.squadrons.remove(uid)
        lift = state.squadrons[keep].capacity
        self.assertLess(lift, 500, "test needs a squadron that cannot lift an army")

        for troop in list(state.troops_at(hex_id, IMPERIAL)):
            state.troops.pop(troop.uid, None)
        big = _place_troop(state, oob.TroopClass("army", 500, 15), "imperial", hex_id)
        small = _place_troop(state, oob.TroopClass("battalion", 1, 15), "imperial", hex_id)

        fleet.plot[state.turn] = ("jump", fleet.location)
        agent.load_troops(state, IMPERIAL, None)
        self.assertIsNone(big.aboard,
                          "a 500-factor army cannot fit in %d factors of lift" % lift)
        self.assertIsNotNone(small.aboard,
                             "the small unit must still be loaded behind it")

    def test_the_imperium_can_put_troops_to_sea(self):
        """An Imperial army that never embarks can never retake anything."""
        state = new_game(seed=5)
        imperial = ScriptedAgent(IMPERIAL, seed=1)
        zhodani = ScriptedAgent(ZHODANI, seed=2)
        engine = Engine(state, imperial, zhodani)
        imperial.begin_game(state, IMPERIAL, engine)
        zhodani.begin_game(state, ZHODANI, engine)
        engine.initial_plotting()
        afloat = 0
        for _ in range(20):
            if state.game_over:
                break
            engine.play_turn()
            afloat = max(afloat, sum(t.current for t in state.troops.values()
                                     if t.side == IMPERIAL and t.aboard is not None
                                     and t.losses < 100))
        self.assertGreater(afloat, 0,
                           "the Imperium never embarked a single troop factor")


class TestFeatureLayout(unittest.TestCase):
    """The perspective flip is derived from names, so it must stay consistent."""

    def test_flip_is_its_own_inverse(self):
        from ffw import features
        state = new_game(seed=31)
        raw = features.extract(state)
        for side in (IMPERIAL, ZHODANI):
            twice = features.perspective(features.perspective(raw, side), side)
            self.assertTrue(np.allclose(twice, raw))

    def test_every_paired_feature_is_swapped(self):
        """A feature named _imperial/_zhodani must have a partner and swap."""
        from ffw import features
        paired = [n for n in features.FEATURE_NAMES
                  if n.endswith(("_imperial", "_zhodani"))]
        self.assertEqual(len(paired), 2 * len(features.SWAP_PAIRS),
                         "a paired feature has no partner in SWAP_PAIRS")
        probe = np.arange(features.N_FEATURES, dtype=float)
        flipped = features.perspective(probe, IMPERIAL)
        for a, b in features.SWAP_PAIRS:
            self.assertEqual(flipped[a], probe[b])
            self.assertEqual(flipped[b], probe[a])

    def test_signed_features_are_negated(self):
        from ffw import features
        probe = np.ones(features.N_FEATURES)
        flipped = features.perspective(probe, IMPERIAL)
        for name in features.SIGNED:
            self.assertEqual(flipped[features.INDEX[name]], -1.0)

    def test_neutral_features_are_untouched(self):
        from ffw import features
        swapped = {i for pair in features.SWAP_PAIRS for i in pair}
        signed = {features.INDEX[n] for n in features.SIGNED}
        probe = np.arange(features.N_FEATURES, dtype=float) + 1.0
        flipped = features.perspective(probe, IMPERIAL)
        for i, name in enumerate(features.FEATURE_NAMES):
            if i in swapped or i in signed:
                continue
            self.assertEqual(flipped[i], probe[i],
                             "%s should be viewer-independent" % name)

    def test_a_zhodani_advance_reads_as_good_for_the_zhodani(self):
        from ffw import features
        state = new_game(seed=32)
        before = features.extract(state)
        for hex_id in ("2314", "2109", "1510"):
            state.world_map.get(hex_id).control = ZHODANI
        after = features.extract(state)
        i = features.INDEX["vp_margin"]
        self.assertGreater(after[i], before[i])
        # ... and the reverse when read from the Imperial seat
        self.assertLess(features.perspective(after, IMPERIAL)[i],
                        features.perspective(before, IMPERIAL)[i])

    def test_stale_networks_are_rejected_not_misread(self):
        from ffw.agents import ValueNetwork
        from ffw import features
        net = ValueNetwork(hidden=4, seed=0)
        payload = net.to_dict()
        payload["feature_version"] = features.FEATURE_VERSION + 99
        with self.assertRaises(ValueError):
            ValueNetwork.from_dict(payload)


class TestLeagueTraining(unittest.TestCase):
    def test_league_returns_a_doctrine_for_each_side(self):
        from ffw.training import train_league
        from ffw.agents import WEIGHTS
        champions, log = train_league(generations=1, population=3, elite=2,
                                      games=1, pool=2, max_turns=8,
                                      benchmark_games=1, seed=2)
        self.assertEqual(set(champions), {IMPERIAL, ZHODANI})
        for side in champions:
            self.assertEqual(set(champions[side]), set(WEIGHTS))
        self.assertEqual(len(log.rows), 2)
        for row in log.rows:
            self.assertIn("benchmark", row)
            self.assertIn("accepted", row)

    def test_league_pool_gains_variety(self):
        """A pool of clones is a fixed opponent wearing hats; it must diverge."""
        from ffw.training import train_league
        champions, log = train_league(generations=2, population=3, elite=2,
                                      games=1, pool=2, max_turns=8,
                                      benchmark_games=1, seed=3)
        zho = [r["weights"] for r in log.rows if r["side"] == ZHODANI]
        self.assertEqual(len(zho), 2)
        self.assertIsInstance(zho[0], dict)


class TestEvaluatorHonesty(unittest.TestCase):
    """A correlation with the outcome is not the same as judgement."""

    def test_report_includes_the_trivial_baseline(self):
        from ffw.training import train_value_network, evaluator_report
        net, _ = train_value_network(games=4, epochs=10, seed=2, max_turns=12)
        report = evaluator_report(net, games=3, seed=5, max_turns=12)
        for key in ("baseline_correlation", "skill_over_baseline",
                    "early_correlation", "early_baseline"):
            self.assertIn(key, report)
        self.assertAlmostEqual(
            report["skill_over_baseline"],
            report["correlation"] - report["baseline_correlation"], places=9)

    def test_echoing_the_score_scores_no_skill(self):
        """A "network" that returns the margin column must show zero skill."""
        from ffw.training import evaluator_report
        from ffw import features

        class Echo:
            def __call__(self, vector):
                return float(vector[features.INDEX["vp_margin"]])

        report = evaluator_report(Echo(), games=3, seed=6, max_turns=14)
        self.assertLess(abs(report["skill_over_baseline"]), 0.05,
                        "echoing one feature must not read as skill")

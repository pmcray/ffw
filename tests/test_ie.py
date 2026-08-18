"""Tests for the Invasion: Earth implementation.

Run with ``python -m unittest tests.test_ie`` or with the rest of the suite.

The rulebook is unusually testable: it prints four combat tables, a counter
inventory, an order of battle chart with exact counts, and several worked
examples with the dice rolls given.  Wherever it states a number, the number is
asserted here rather than paraphrased.
"""

from __future__ import annotations

import math
import os
import random
import sys
import unittest
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ie import hexmap, oob, tables, terra                            # noqa: E402
from ie.engine import Engine, new_game, play                         # noqa: E402
from ie.state import (BOMBARDMENT, CLOSE_ORBIT, DEEP_SPACE,          # noqa: E402
                      FAR_ORBIT, IMPERIAL, OUT_SYSTEM, OVERWATCH,
                      SOLOMANI, VICTORY_URBAN_THRESHOLD, Base, GameState,
                      enemy_of)
from ie.agents import (Agent, BeachheadAgent, HeuristicAgent,        # noqa: E402
                       HumanAgent, RandomAgent, ScriptedAgent)


class TestGeodesicGrid(unittest.TestCase):
    """The map is a sphere, and the rules say so if you read them closely.

    "A hex divided between two or more of these triangles is considered to be a
    single hex for all purposes" and "the eastern and western edges of the map
    are considered to be adjacent" exist only to undo the tearing the paper
    projection causes.  Applied, they describe a hex grid on a sphere.
    """

    def test_the_cell_count_is_the_goldberg_formula(self):
        self.assertEqual(hexmap.cell_count(), 10 * hexmap.FREQUENCY ** 2 + 2)

    def test_there_are_exactly_twelve_pentagons(self):
        """Euler's formula, not a modelling choice: a sphere cannot be all hexes."""
        self.assertEqual(len(hexmap.PENTAGONS), 12)
        degrees = Counter(len(hexmap.neighbours(c)) for c in hexmap.CELLS)
        self.assertEqual(degrees[5], 12)
        self.assertEqual(degrees[6], hexmap.cell_count() - 12)

    def test_adjacency_is_symmetric(self):
        for cell in hexmap.CELLS:
            for other in hexmap.neighbours(cell):
                self.assertIn(cell, hexmap.neighbours(other))

    def test_no_cell_is_its_own_neighbour(self):
        for cell in hexmap.CELLS:
            self.assertNotIn(cell, hexmap.neighbours(cell))

    def test_the_hexes_are_about_the_size_the_rulebook_prints(self):
        """1148 km across; the grid quantises to 10n^2+2 cells and lands near it."""
        width = hexmap.cell_width_km()
        self.assertLess(abs(width - hexmap.NOMINAL_HEX_KM) / hexmap.NOMINAL_HEX_KM,
                        0.10)

    def test_the_grid_wraps(self):
        """A hex near the antimeridian has neighbours on the other side of it."""
        cell = hexmap.nearest(0, 179)
        longitudes = [hexmap.lat_lon(n)[1] for n in hexmap.neighbours(cell)]
        self.assertTrue(any(lon < -150 for lon in longitudes),
                        "the map does not wrap east to west")

    def test_the_poles_are_ordinary_cells(self):
        for lat in (90, -90):
            cell = hexmap.nearest(lat, 0)
            self.assertGreaterEqual(len(hexmap.neighbours(cell)), 5)
            self.assertAlmostEqual(abs(hexmap.lat_lon(cell)[0]), 90, places=6)

    def test_every_cell_has_an_outline_and_it_matches_its_neighbours(self):
        """A cell has one corner per neighbour: five for the twelve, six else.

        The corners are the centres of the geodesic triangles meeting at the
        cell, so a disagreement between corner count and neighbour count would
        mean the dual had been built two different ways.
        """
        counts = Counter()
        for cell in hexmap.CELLS:
            corners = hexmap.outline(cell)
            counts[len(corners)] += 1
            self.assertEqual(len(corners), len(hexmap.neighbours(cell)), cell)
            for corner in corners:
                self.assertAlmostEqual(
                    math.sqrt(sum(v * v for v in corner)), 1.0, places=9)
        self.assertEqual(counts[5], 12)
        self.assertEqual(counts[6], hexmap.cell_count() - 12)

    def test_an_outline_is_wound_around_its_own_cell(self):
        """Corners in the wrong order draw a bow tie rather than a hexagon.

        Two checks: every step around the outline turns the same way in the
        tangent plane, and the corners average back to the cell's own centre.
        """
        for cell in (0, 100, 250, 491, min(hexmap.PENTAGONS)):
            centre = hexmap.unit_vector(cell)
            corners = hexmap.outline(cell)
            angles = []
            ref = (0.0, 0.0, 1.0) if abs(centre[2]) < 0.9 else (1.0, 0.0, 0.0)
            east = hexmap._normalise(hexmap._cross(ref, centre))
            north = hexmap._cross(centre, east)
            for corner in corners:
                angles.append(math.atan2(hexmap._dot(corner, north),
                                         hexmap._dot(corner, east)))
            steps = [(angles[(i + 1) % len(angles)] - a) % (2 * math.pi)
                     for i, a in enumerate(angles)]
            self.assertAlmostEqual(sum(steps), 2 * math.pi, places=6, msg=cell)
            middle = hexmap._normalise(tuple(
                sum(c[d] for c in corners) / len(corners) for d in range(3)))
            self.assertGreater(hexmap._dot(middle, centre), 0.999, cell)

    def test_distance_is_symmetric_and_zero_on_self(self):
        pairs = [(0, 100), (17, 300), (250, 251)]
        for a, b in pairs:
            self.assertEqual(hexmap.distance(a, b), hexmap.distance(b, a))
            self.assertEqual(hexmap.distance(a, a), 0)

    def test_adjacent_cells_are_one_hex_apart(self):
        for cell in (0, 100, 250, 400):
            for other in hexmap.neighbours(cell):
                self.assertEqual(hexmap.distance(cell, other), 1)

    def test_pole_to_pole_is_half_the_circumference(self):
        north, south = hexmap.nearest(90, 0), hexmap.nearest(-90, 0)
        steps = hexmap.distance(north, south)
        km_per_step = hexmap.great_circle_km(north, south) / steps
        self.assertLess(abs(km_per_step - hexmap.cell_width_km())
                        / hexmap.cell_width_km(), 0.15)

    def test_within_matches_distance(self):
        origin = 200
        band = set(hexmap.within(origin, 2))
        self.assertNotIn(origin, band)
        for cell in band:
            self.assertLessEqual(hexmap.distance(origin, cell), 2)


class TestTerra(unittest.TestCase):
    """The map, against the counts the rulebook states."""

    def setUp(self):
        self.world = terra.load()
        self.terrain = self.world["terrain"]

    def test_there_are_exactly_sixty_one_urban_hexes(self):
        """Rule 6: "The Solomani player has 61 urban hexes at the start"."""
        count = sum(1 for k in self.terrain.values() if k == terra.URBAN)
        self.assertEqual(count, terra.URBAN_HEXES)
        self.assertEqual(count, 61)

    def test_there_are_three_starports(self):
        ports = [c for c, k in self.terrain.items() if k == terra.STARPORT]
        self.assertEqual(len(ports), 3)
        self.assertEqual(len(self.world["starports"]), 3)

    def test_the_starports_are_where_the_map_prints_them(self):
        names = set(self.world["starports"].values())
        self.assertEqual(names, {"Starport Phoenix", "AECO Starport",
                                 "LaGrange Starport"})
        for cell, name in self.world["starports"].items():
            lat, lon = hexmap.lat_lon(cell)
            continent = terra.continent(lat, lon)
            if name == "Starport Phoenix":
                self.assertEqual(continent, "North America")
            elif name == "AECO Starport":
                self.assertEqual(continent, "Africa")
            else:
                self.assertEqual(continent, "Australia")

    def test_every_cell_has_a_terrain(self):
        self.assertEqual(len(self.terrain), hexmap.cell_count())
        for kind in self.terrain.values():
            self.assertIn(kind, (terra.LAND | terra.DEEP_SEA
                                 | {terra.PERMANENT_ICE, terra.SEASONAL_ICE}))

    def test_there_is_room_for_every_sdb_wing_to_hide(self):
        """"Only one SDB wing may hide in a single deep sea hex" -- and there
        are thirty-four wings plus two dummies."""
        deep = sum(1 for k in self.terrain.values() if k in terra.DEEP_SEA)
        self.assertGreaterEqual(deep, oob.total(oob.SOLOMANI_SDB) + oob.DUMMY_SDB)

    def test_the_four_deployment_continents_all_have_cities(self):
        """The Solomani armies deploy one to each; they have to have somewhere."""
        by_continent = Counter()
        for cell, kind in self.terrain.items():
            if kind == terra.URBAN:
                by_continent[terra.continent(*hexmap.lat_lon(cell))] += 1
        for name in ("Africa", "Asia", "North America", "South America"):
            self.assertGreater(by_continent[name], 0, name)

    def test_land_and_sea_are_both_substantial(self):
        land = sum(1 for k in self.terrain.values() if k in terra.LAND)
        fraction = land / len(self.terrain)
        self.assertGreater(fraction, 0.2)
        self.assertLess(fraction, 0.5)

    def test_ice_and_tundra_count_as_winter_always(self):
        """Rule 5's climate paragraph."""
        for kind in (terra.TUNDRA, terra.PERMANENT_ICE, terra.SEASONAL_ICE):
            self.assertIn(kind, terra.ALWAYS_WINTER)
        self.assertNotIn(terra.CLEAR, terra.ALWAYS_WINTER)

    def test_guerrilla_cover_is_the_terrain_the_rule_names(self):
        """"urban, desert, and any type of wilderness"."""
        self.assertEqual(
            terra.GUERRILLA_COVER,
            frozenset({terra.URBAN, terra.STARPORT, terra.DESERT,
                       terra.TUNDRA, terra.FOREST, terra.MOUNTAIN}))
        self.assertNotIn(terra.CLEAR, terra.GUERRILLA_COVER)

    def test_the_cached_map_matches_a_fresh_build(self):
        self.assertEqual(terra.load()["terrain"], terra.build()["terrain"])


class TestCombatTables(unittest.TestCase):
    """The chart, and the rulebook's own worked examples."""

    def test_the_column_rule(self):
        """"the column that most closely approximates without exceeding"."""
        columns = tables.SPACE_ATTACK_COLUMNS
        self.assertEqual(columns[tables.column_index(columns, 29)], 24)
        self.assertEqual(columns[tables.column_index(columns, 24)], 24)
        self.assertEqual(columns[tables.column_index(columns, 1000)], 90)
        self.assertEqual(tables.column_index(columns, 0), -1)

    def test_every_table_row_is_the_width_of_its_columns(self):
        for table, columns in (
                (tables.SPACE_ATTACK, tables.SPACE_ATTACK_COLUMNS),
                (tables.SPACE_BOMBARD, tables.SPACE_BOMBARD_COLUMNS),
                (tables.SURFACE_BOMBARD, tables.SURFACE_BOMBARD_COLUMNS),
                (tables.TROOP_COMBAT, tables.TROOP_ODDS)):
            for roll, row in table.items():
                self.assertEqual(len(row), len(columns), "%r roll %s" % (table, roll))

    def test_the_odds_examples_from_the_rulebook(self):
        """"13:5 ... rounds down to 2:1.  25:2 ... rounds down to 10:1"."""
        self.assertEqual(tables.TROOP_ODDS_LABELS[tables.odds_index(13, 5)], "2:1")
        self.assertEqual(tables.TROOP_ODDS_LABELS[tables.odds_index(25, 2)], "10:1")
        self.assertEqual(tables.TROOP_ODDS_LABELS[tables.odds_index(100, 100)], "1:1")
        self.assertEqual(tables.TROOP_ODDS_LABELS[tables.odds_index(100, 20)], "5:1")

    def test_odds_below_one_to_a_hundred_are_treated_as_one_to_a_hundred(self):
        self.assertEqual(tables.odds_index(1, 100000), 0)

    def test_the_rulebooks_four_attack_worked_example(self):
        """Page 7's example, all four attacks, with the dice it gives.

        100 TL14 on a full 1C-14 at 1:1, roll 6, is 10%.  100 TL12 on a
        20-strength 1C-14 at 5:1 shifted two left to 2:1, roll 7, is 20%.  100
        TL14 on a 1C-12 at 1:1 shifted two right to 2:1, roll 5, is 40%.  20 on
        a full 1C-14 at 1:5, roll 9, is no effect.
        """
        self.assertEqual(tables.troop_combat(100, 100, 6, 0), 10)
        self.assertEqual(tables.troop_combat(100, 20, 7, -2), 20)
        self.assertEqual(tables.troop_combat(100, 100, 5, 2), 40)
        self.assertEqual(tables.troop_combat(20, 100, 9, 0), 0)

    def test_the_percentage_loss_examples(self):
        """"a 20 factor division that has suffered 60% losses would have a
        current strength of 8", and "a 100 factor corps that has suffered 80%
        losses would have a current strength of 20"."""
        self.assertEqual(tables.current_strength(20, 60), 8)
        self.assertEqual(tables.current_strength(100, 80), 20)
        self.assertEqual(tables.current_strength(500, 50), 250)
        self.assertEqual(tables.current_strength(100, 100), 0)

    def test_losses_are_additive_on_the_printed_strength(self):
        """"A troop unit receives 10% losses ... then 40% ... total 50%"."""
        self.assertEqual(tables.current_strength(100, 10 + 40), 50)

    def test_a_modified_roll_is_held_inside_the_table(self):
        """The chart's closing note, with its own example: a 2 in winter stays 2."""
        self.assertEqual(tables.troop_combat(100, 100, 1, 0),
                         tables.troop_combat(100, 100, 2, 0))
        self.assertEqual(tables.troop_combat(100, 100, 99, 0),
                         tables.troop_combat(100, 100, 12, 0))

    def test_a_hidden_sdb_is_attacked_three_columns_left(self):
        """"if the 18 column ... would normally be used, then the 3 column"."""
        columns = tables.SPACE_BOMBARD_COLUMNS
        index = tables.column_index(columns, 18)
        self.assertEqual(columns[index], 18)
        self.assertEqual(columns[max(0, index + tables.HIDDEN_SDB_COLUMN_SHIFT)], 3)

    def test_an_attack_shifted_below_zero_uses_the_zero_column(self):
        """"Any attack shifted below the 0 column is resolved using the 0 column"."""
        low = tables.space_bombard(1, 6, column_shift=-9)
        self.assertEqual(low, tables.space_bombard(0, 6))

    def test_the_space_attack_table_has_no_column_below_one(self):
        self.assertEqual(tables.space_attack(0.5, 6), 0)

    def test_the_bombardment_table_does_have_a_zero_column(self):
        self.assertEqual(tables.SPACE_BOMBARD_COLUMNS[0], 0)

    def test_the_modifier_tables(self):
        self.assertEqual(tables.TECH_MODIFIER[14], 0)
        self.assertEqual(tables.TECH_MODIFIER[13], 0)
        self.assertEqual(tables.TECH_MODIFIER[12], -1)
        self.assertEqual(tables.TECH_MODIFIER[11], -1)
        self.assertEqual(tables.ASSAULT_MODIFIER["marine"], 0)
        self.assertEqual(tables.ASSAULT_MODIFIER["jump"], 0)
        self.assertEqual(tables.ASSAULT_MODIFIER["other"], -3)
        self.assertEqual(tables.WINTER_MODIFIER, -1)
        self.assertEqual(tables.GUERRILLA_HIDING_MODIFIER, 3)

    def test_more_fire_never_gives_a_worse_result(self):
        """A sanity property of every column: the table is monotone."""
        for roll in tables.SURFACE_BOMBARD:
            row = [v or 0 for v in tables.SURFACE_BOMBARD[roll]]
            self.assertEqual(row, sorted(row), "roll %d" % roll)


class TestOrderOfBattle(unittest.TestCase):
    """Counts against the order of battle chart, line by line."""

    def _sizes(self, spec):
        counts = Counter()
        for cls, n in spec:
            counts[cls.size] += n
        return counts

    def test_solomani_initial_forces(self):
        self.assertEqual(oob.total(oob.SOLOMANI_SDB), 34)
        self.assertEqual(oob.total(oob.SOLOMANI_NAVAL), 8)
        pd = self._sizes(oob.SOLOMANI_PD)
        self.assertEqual((pd["corps"], pd["division"], pd["regiment"]), (3, 17, 4))
        troops = self._sizes(oob.SOLOMANI_TROOPS)
        self.assertEqual((troops["army"], troops["corps"], troops["division"],
                          troops["regiment"]), (4, 12, 16, 8))
        self.assertEqual(oob.total(oob.SOLOMANI_GUERRILLAS), 8)

    def test_imperial_initial_forces(self):
        by_kind = Counter()
        for cls, n in oob.IMPERIAL_NAVAL:
            by_kind[(cls.kind, cls.colonial)] += n
        self.assertEqual(by_kind[("BR", False)], 14)
        self.assertEqual(by_kind[("CR", False)], 7)
        self.assertEqual(by_kind[("SR", False)], 3)
        self.assertEqual(by_kind[("AR", False)], 4)
        self.assertEqual(by_kind[("BR", True)], 6)
        self.assertEqual(by_kind[("CR", True)], 8)

        troops = self._sizes(oob.IMPERIAL_TROOPS)
        self.assertEqual((troops["army"], troops["corps"], troops["division"],
                          troops["brigade"]), (2, 16, 8, 6))
        colonial = self._sizes(oob.IMPERIAL_COLONIAL)
        self.assertEqual((colonial["corps"], colonial["division"],
                          colonial["brigade"]), (10, 4, 2))
        marines = self._sizes(oob.IMPERIAL_MARINES)
        self.assertEqual((marines["division"], marines["regiment"]), (1, 5))
        mercs = self._sizes(oob.IMPERIAL_MERCENARIES)
        self.assertEqual((mercs["division"], mercs["brigade"],
                          mercs["regiment"]), (2, 3, 1))
        self.assertEqual(oob.IMPERIAL_BASES, 10)

    def test_only_imperial_squadrons_can_jump(self):
        """"The Solomani squadrons in the game are incapable of making
        interstellar jumps", and SDB wings are "inherently incapable"."""
        for cls, _n in oob.IMPERIAL_NAVAL:
            self.assertTrue(cls.jump, cls.name)
        for spec in (oob.SOLOMANI_NAVAL, oob.SOLOMANI_SDB):
            for cls, _n in spec:
                self.assertFalse(cls.jump, cls.name)

    def test_the_transport_capability_table(self):
        """"a battle squadron may carry no more than one 20-factor surface unit"."""
        self.assertEqual(oob.TRANSPORT_CAPACITY["AR"], 600)
        self.assertEqual(oob.TRANSPORT_CAPACITY["BR"], 20)
        self.assertEqual(oob.TRANSPORT_CAPACITY["CR"], 5)
        self.assertEqual(oob.TRANSPORT_CAPACITY["SR"], 0)
        self.assertEqual(oob.TRANSPORT_CAPACITY["SDB"], 0)

    def test_pd_units_are_static_and_armed(self):
        for cls, _n in oob.SOLOMANI_PD:
            self.assertTrue(cls.planetary_defense)
            self.assertFalse(cls.mobile)
            self.assertGreater(cls.bombard, 0)

    def test_the_imperium_cannot_lift_its_own_army_in_one_go(self):
        """The fact the whole Imperial problem turns on."""
        lift = sum(cls.capacity * n for cls, n in oob.IMPERIAL_NAVAL)
        army = sum(oob.factors(spec) for spec in
                   (oob.IMPERIAL_TROOPS, oob.IMPERIAL_COLONIAL,
                    oob.IMPERIAL_MARINES, oob.IMPERIAL_MERCENARIES))
        self.assertLess(lift, army)


class TestSetup(unittest.TestCase):
    def setUp(self):
        self.state = new_game(seed=11)

    def test_every_imperial_unit_starts_out_system(self):
        """"All forces are placed in the out-system box"."""
        for unit in self.state.naval.values():
            if unit.side == IMPERIAL:
                self.assertEqual(unit.location, OUT_SYSTEM)
        for unit in self.state.surface.values():
            if unit.side == IMPERIAL:
                self.assertEqual(unit.location, OUT_SYSTEM)
        for base in self.state.bases.values():
            self.assertEqual(base.location, OUT_SYSTEM)

    def test_the_solomani_start_with_all_sixty_one_cities(self):
        self.assertEqual(len(self.state.solomani_urban()), 61)

    def test_solomani_naval_units_start_in_orbit(self):
        """"Naval units are placed in far orbit or in near orbit"."""
        for unit in self.state.naval.values():
            if unit.side == SOLOMANI and not unit.is_sdb:
                self.assertIn(unit.location, (FAR_ORBIT, CLOSE_ORBIT))

    def test_the_sdb_wings_start_hidden_in_distinct_ocean_hexes(self):
        wings = [u for u in self.state.naval.values() if u.is_sdb]
        self.assertEqual(len(wings), 34 + oob.DUMMY_SDB)
        cells = [u.location for u in wings]
        self.assertEqual(len(set(cells)), len(cells), "two wings share a hex")
        for wing in wings:
            self.assertTrue(wing.hidden)
            self.assertIn(wing.location, self.state.geometry.deep_sea)

    def test_a_pd_corps_sits_on_each_starport(self):
        """"One PD corps is placed at each starport"."""
        for cell in self.state.geometry.starports:
            here = self.state.surface_at(cell, SOLOMANI)
            self.assertTrue(any(u.cls.planetary_defense and u.cls.size == "corps"
                                for u in here), "no PD corps at %d" % cell)

    def test_no_two_pd_divisions_share_a_city(self):
        """"All PD divisions are placed in urban hexes, no more than one per hex"."""
        cells = [u.location for u in self.state.surface.values()
                 if u.cls.planetary_defense and u.cls.size == "division"]
        self.assertEqual(len(set(cells)), len(cells))

    def test_one_solomani_army_per_named_continent(self):
        """"One army is placed in any urban hex on each of the following
        continents: Africa, Asia, North America, and South America"."""
        continents = []
        for unit in self.state.surface.values():
            if unit.side == SOLOMANI and unit.cls.size == "army":
                lat, lon = hexmap.lat_lon(unit.location)
                continents.append(terra.continent(lat, lon))
        self.assertEqual(sorted(continents),
                         ["Africa", "Asia", "North America", "South America"])

    def test_the_guerrillas_are_not_on_the_map_yet(self):
        """They arrive at the end of the first two quarters, four at a time."""
        self.assertFalse([u for u in self.state.surface.values()
                          if u.cls.guerrilla])
        self.assertEqual(len(self.state.pools["guerrillas"]), 8)

    def test_setup_is_reproducible(self):
        other = new_game(seed=11)
        self.assertEqual(sorted((u.cls.name, str(u.location))
                                for u in self.state.surface.values()),
                         sorted((u.cls.name, str(u.location))
                                for u in other.surface.values()))


class TestVictory(unittest.TestCase):
    """Rule 8's arithmetic and the level of victory table."""

    def test_the_level_of_victory_table(self):
        state = new_game(seed=2)
        for points, level in ((10, "imperial decisive victory"),
                              (7, "imperial decisive victory"),
                              (6, "imperial major victory"),
                              (4, "imperial major victory"),
                              (3, "imperial marginal victory"),
                              (1, "imperial marginal victory"),
                              (0, "draw"), (-1, "draw"),
                              (-2, "solomani marginal victory"),
                              (-3, "solomani marginal victory"),
                              (-4, "solomani major victory"),
                              (-9, "solomani major victory")):
            self.assertEqual(state.victory_level(points), level, points)

    def test_the_ten_points_are_only_awarded_for_taking_the_cities(self):
        """Rule 8's award is conditional, and the condition is the whole game.

        Awarded unconditionally it swamps everything else: the total becomes
        ten minus the clock, every game ends in the same level of victory
        whatever happened on the ground, and an invasion that never landed
        scores an imperial major victory.
        """
        state = new_game(seed=2)
        state.turn = 13                       # two complete quarters elapsed
        self.assertFalse(state.objective_met())
        self.assertEqual(state.victory_points(), -2)
        self.assertEqual(state.victory_level(), "solomani marginal victory")

        cities = sorted(state.geometry.urban)
        state.garrisoned = set(cities[:len(cities) - 5])
        self.assertTrue(state.objective_met())
        self.assertEqual(state.victory_points(), 8)
        self.assertEqual(state.victory_level(), "imperial decisive victory")

    def test_a_quarter_and_a_wave_each_cost_a_victory_point(self):
        state = new_game(seed=2)
        state.turn = 13                       # two complete quarters elapsed
        base = state.victory_points()
        state.waves_taken = 3
        self.assertEqual(state.victory_points(), base - 3)

    def test_wiping_out_the_defenders_is_worth_a_point_each(self):
        state = new_game(seed=2)
        before = state.victory_points()
        for unit in list(state.surface.values()):
            if unit.side == SOLOMANI and not unit.cls.guerrilla:
                del state.surface[unit.uid]
        self.assertEqual(state.victory_points(), before + 1)
        for unit in list(state.naval.values()):
            if unit.side == SOLOMANI:
                del state.naval[unit.uid]
        self.assertEqual(state.victory_points(), before + 2)

    def test_the_game_ends_when_fewer_than_ten_cities_remain(self):
        state = new_game(seed=4)
        engine = Engine(state, HeuristicAgent(IMPERIAL, seed=1),
                        HeuristicAgent(SOLOMANI, seed=2), rng=random.Random(4))
        cities = sorted(state.geometry.urban)
        state.garrisoned = set(cities[:len(cities) - 5])
        self.assertLess(len(state.solomani_urban()), VICTORY_URBAN_THRESHOLD)
        state.turn = 6
        engine.quarterly_special_turn()
        self.assertTrue(state.game_over)

    def test_abandoning_the_invasion_is_a_solomani_major_victory(self):
        class Quitter(HeuristicAgent):
            def abandon_invasion(self, state, engine):
                return True

        state = new_game(seed=5)
        engine = Engine(state, Quitter(IMPERIAL, seed=1),
                        HeuristicAgent(SOLOMANI, seed=2), rng=random.Random(5))
        state.turn = 6
        engine.quarterly_special_turn()
        self.assertTrue(state.game_over)
        self.assertEqual(state.result, "solomani major victory")
        self.assertTrue(state.abandoned)

    def test_the_quarter_boundaries_are_every_six_turns(self):
        state = new_game(seed=6)
        for turn, is_end in ((1, False), (5, False), (6, True), (7, False),
                             (12, True), (18, True)):
            state.turn = turn
            self.assertEqual(state.is_quarter_end(), is_end, turn)


class TestMovement(unittest.TestCase):
    """Rule 3, in space and on the surface."""

    def setUp(self):
        self.state = new_game(seed=21)
        self.engine = Engine(self.state, Agent(IMPERIAL, seed=1),
                             Agent(SOLOMANI, seed=2), rng=random.Random(21))

    def _imperial_squadron(self):
        return next(u for u in self.state.naval.values()
                    if u.side == IMPERIAL and u.cls.kind == "BR")

    def test_a_jump_may_not_start_or_end_in_orbit(self):
        """"a naval unit may not jump from or to the close orbit or far orbit"."""
        unit = self._imperial_squadron()
        unit.location = FAR_ORBIT
        self.assertFalse(self.engine._move_naval(unit, OUT_SYSTEM))
        unit.location = CLOSE_ORBIT
        self.assertFalse(self.engine._move_naval(unit, OUT_SYSTEM))
        unit.location = DEEP_SPACE
        self.assertTrue(self.engine._move_naval(unit, OUT_SYSTEM))

    def test_only_one_jump_a_turn(self):
        unit = self._imperial_squadron()
        unit.location, unit.jumped = DEEP_SPACE, False
        self.assertTrue(self.engine._move_naval(unit, OUT_SYSTEM))
        self.assertFalse(self.engine._move_naval(unit, DEEP_SPACE))

    def test_the_solomani_can_never_jump(self):
        unit = next(u for u in self.state.naval.values()
                    if u.side == SOLOMANI and not u.is_sdb)
        unit.location = DEEP_SPACE
        self.assertFalse(self.engine._move_naval(unit, OUT_SYSTEM))

    def test_in_system_movement_passes_through_far_orbit(self):
        """"A unit moving between the close orbit and deep space boxes must
        first enter the far orbit box as part of its move"."""
        unit = self._imperial_squadron()
        unit.location = DEEP_SPACE
        for other in list(self.state.naval.values()):
            if other.side == SOLOMANI and other.location == FAR_ORBIT:
                del self.state.naval[other.uid]
        self.assertTrue(self.engine._move_naval(unit, CLOSE_ORBIT))
        self.assertEqual(unit.location, CLOSE_ORBIT)

    def test_movement_stops_on_contact(self):
        """"must cease its in-system movement immediately upon entering a space
        box containing enemy naval units"."""
        unit = self._imperial_squadron()
        unit.location = DEEP_SPACE
        blocker = next(u for u in self.state.naval.values()
                       if u.side == SOLOMANI and not u.is_sdb)
        blocker.location, blocker.hidden = FAR_ORBIT, False
        self.engine._move_naval(unit, CLOSE_ORBIT)
        self.assertEqual(unit.location, FAR_ORBIT)

    def test_hidden_units_do_not_block(self):
        """"Enemy naval units hiding in the deep space box are ignored for all
        purposes of in-system movement"."""
        unit = self._imperial_squadron()
        unit.location = DEEP_SPACE
        for other in self.state.naval.values():
            if other.side == SOLOMANI and not other.is_sdb:
                other.location, other.hidden = FAR_ORBIT, True
        self.engine._move_naval(unit, CLOSE_ORBIT)
        self.assertEqual(unit.location, CLOSE_ORBIT)

    def test_a_grav_unit_has_ten_movement_points(self):
        self.assertEqual(oob.MOVEMENT_POINTS, 10)
        unit = next(u for u in self.state.surface.values()
                    if u.side == SOLOMANI and u.cls.mobile)
        origin = unit.location
        far = [c for c in self.state.geometry.land
               if hexmap.distance(origin, c) == 9]
        if far:
            cost = self.engine._path_cost(unit, origin, far[0], set(), set())
            self.assertIsNotNone(cost)
            self.assertLessEqual(cost, oob.MOVEMENT_POINTS)

    def test_a_zone_of_control_costs_an_extra_movement_point(self):
        """"the total cost to enter a hex in an enemy zone of control is 2"."""
        unit = next(u for u in self.state.surface.values()
                    if u.side == SOLOMANI and u.cls.mobile)
        origin = unit.location
        step = self.state.geometry.adjacent(origin)[0]
        plain = self.engine._path_cost(unit, origin, step, set(), set())
        zoc = self.engine._path_cost(unit, origin, step, {step}, set())
        self.assertEqual(plain, 1)
        self.assertEqual(zoc, 2)

    def test_an_occupied_hex_in_a_zone_of_control_costs_three(self):
        unit = next(u for u in self.state.surface.values()
                    if u.side == SOLOMANI and u.cls.mobile)
        origin = unit.location
        step = self.state.geometry.adjacent(origin)[0]
        self.assertEqual(
            self.engine._path_cost(unit, origin, step, {step}, {step}), 3)

    def test_a_commando_ignores_zones_of_control(self):
        """"A commando unit may totally ignore the presence of enemy units and
        enemy zones of control while moving"."""
        commando = next(u for u in self.state.surface.values()
                        if u.cls.commando)
        origin = commando.location
        step = self.state.geometry.adjacent(origin)[0]
        self.assertEqual(
            self.engine._path_cost(commando, origin, step, {step}, {step}), 1)

    def test_only_corps_army_and_pd_units_have_a_zone_of_control(self):
        state = self.state
        zoc = state.zone_of_control(SOLOMANI)
        for unit in state.surface.values():
            if unit.side != SOLOMANI or not isinstance(unit.location, int):
                continue
            if unit.cls.size in ("corps", "army") or unit.cls.planetary_defense:
                self.assertIn(unit.location, zoc)

    def test_a_thousand_combat_factors_is_all_a_hex_will_hold(self):
        """"A player may have no more than 1000 combat factors ... in a hex."

        Unenforced, this is the rule that decides the game: four Solomani
        field armies are two thousand factors between them, they are grav-
        mobile with ten movement points, and nothing else stops all four
        arriving on the same beachhead in the same phase.
        """
        state, engine = self.state, self.engine
        armies = [u for u in state.surface.values()
                  if u.side == SOLOMANI and u.cls.size == "army"]
        self.assertGreaterEqual(len(armies), 3)
        target = next(c for c in sorted(state.geometry.land)
                      if not state.surface_at(c))
        movers = armies[:3]

        class Mover(Agent):
            def surface_moves(self, s, side, e):
                return {u.uid: target for u in movers}

        engine.agents[SOLOMANI] = Mover(SOLOMANI, seed=1)
        for unit in movers:                # put them in reach of the target
            unit.location = state.geometry.adjacent(target)[0]
        engine.movement_phase(SOLOMANI)
        stacked = sum(u.current for u in state.surface_at(target, SOLOMANI))
        self.assertEqual(stacked, oob.STACKING_LIMIT)      # two armies, not three
        self.assertEqual(sum(1 for u in movers if u.location == target), 2)

    def test_the_stacking_limit_applies_to_landings_too(self):
        state, engine = self.state, self.engine
        cell = next(iter(sorted(state.geometry.land)))
        carrier = next(u for u in state.naval.values()
                       if u.side == IMPERIAL and u.cls.capacity >= 600)
        carrier.location = CLOSE_ORBIT
        army = next(u for u in state.surface.values()
                    if u.side == IMPERIAL and u.cls.size == "army")
        army.carrier, army.location = carrier.uid, CLOSE_ORBIT
        carrier.cargo = [army.uid]
        filler = [u for u in state.surface.values()
                  if u.side == IMPERIAL and u.cls.size == "corps"][:6]
        for unit in filler:                 # 600 factors already ashore
            unit.location, unit.carrier = cell, None
        engine._unload(IMPERIAL, army.uid, cell)
        self.assertEqual(army.location, CLOSE_ORBIT)
        self.assertEqual(sum(u.current for u in state.surface_at(cell, IMPERIAL)),
                         600)

    def test_a_unit_may_not_end_movement_in_an_all_sea_hex(self):
        state, engine = self.state, self.engine
        unit = next(u for u in state.surface.values()
                    if u.side == SOLOMANI and u.cls.mobile)
        sea = next(c for c in state.geometry.deep_sea
                   if hexmap.distance(unit.location, c) <= 3)
        origin = unit.location

        class Mover(Agent):
            def surface_moves(self, s, side, e):
                return {unit.uid: sea}

        engine.agents[SOLOMANI] = Mover(SOLOMANI, seed=1)
        engine.movement_phase(SOLOMANI)
        self.assertEqual(unit.location, origin)


class TestSpaceCombat(unittest.TestCase):
    def setUp(self):
        self.state = new_game(seed=31)
        self.engine = Engine(self.state, Agent(IMPERIAL, seed=1),
                             Agent(SOLOMANI, seed=2), rng=random.Random(31))

    def test_damage_removes_at_least_the_defence_factors_dealt(self):
        """"the (minimum) number of defense factors that must be eliminated"."""
        units = [u for u in self.state.naval.values() if u.side == IMPERIAL][:4]
        for unit in units:
            unit.location = CLOSE_ORBIT
        before = sum(u.cls.defense for u in units)
        self.engine._apply_damage(units, 5)
        alive = [u for u in units if u.uid in self.state.naval]
        removed = before - sum(u.cls.defense for u in alive)
        self.assertGreaterEqual(removed, 5)

    def test_a_destroyed_carrier_takes_its_cargo_with_it(self):
        """"If a naval unit is eliminated for any reason, any surface unit it is
        transporting is also eliminated"."""
        state = self.state
        carrier = next(u for u in state.naval.values()
                       if u.side == IMPERIAL and u.cls.capacity > 0)
        passenger = next(u for u in state.surface.values()
                         if u.side == IMPERIAL)
        carrier.cargo.append(passenger.uid)
        passenger.carrier = carrier.uid
        self.engine._destroy_naval(carrier)
        self.assertTrue(passenger.dead)

    def test_naval_units_never_take_partial_damage(self):
        """"A naval unit is totally eliminated upon receiving battle damage"."""
        unit = next(iter(self.state.naval.values()))
        self.assertFalse(hasattr(unit, "losses"))

    def test_bombardment_is_halved_for_a_unit_that_manoeuvred(self):
        unit = next(u for u in self.state.naval.values() if u.cls.bombard >= 4)
        unit.manoeuvred = False
        full = unit.bombard_factor()
        unit.manoeuvred = True
        self.assertEqual(unit.bombard_factor(), full / 2.0)

    def test_halving_retains_fractions(self):
        """"When halving factors, retain all fractions (thus, half of 7 is 3 1/2)"."""
        unit = next(u for u in self.state.naval.values() if u.cls.bombard == 3)
        unit.manoeuvred = True
        self.assertEqual(unit.bombard_factor(), 1.5)


class TestBombardmentPooling(unittest.TestCase):
    """Fire is totalled per target, which is the difference between a game and a rout.

    Both bombardment rules say so: "Total the bombardment factors of all units
    bombarding a given surface unit to determine the column used on the table",
    and for naval targets "all naval units in the close orbit box defend as a
    single group ... by totalling the bombardment factors of all PD units and
    SDB wings firing upon a group".  Resolving one attack per firer instead
    would let the twenty-four Solomani planetary defence units make two dozen
    attacks a turn on the fleet rather than one.
    """

    def test_two_dozen_pd_units_make_one_attack_on_the_fleet(self):
        state = new_game(seed=41)
        fleet = [u for u in state.naval.values() if u.side == IMPERIAL][:12]
        for unit in fleet:
            unit.location = CLOSE_ORBIT
        pd = [u for u in state.surface.values() if u.cls.planetary_defense]
        self.assertGreater(len(pd), 15)

        class Defender(HeuristicAgent):
            def defence_fire(self, s, side, engine):
                return {u.uid: ("naval", CLOSE_ORBIT) for u in pd}

        engine = Engine(state, Agent(IMPERIAL, seed=1),
                        Defender(SOLOMANI, seed=2), rng=random.Random(41))
        before = len(state.naval)
        engine.bombardment_phase([])
        lost = before - len(state.naval)
        # the pooled attack is one row of the space bombardment table, whose
        # largest single result is 9 defence factors
        self.assertLessEqual(lost, 4, "fire was resolved per firer, not pooled")

    def test_the_bombarding_player_chooses_which_unit_is_hit(self):
        """"Each surface unit must be bombarded separately", so someone picks.

        The engine used to pick the biggest, which hands the decision to the
        defender: park a field army on a battery and the battery is safe for
        the rest of the war.
        """
        state = new_game(seed=61)
        cell = next(iter(sorted(state.geometry.land)))
        army = next(u for u in state.surface.values()
                    if u.side == SOLOMANI and u.cls.size == "army")
        battery = next(u for u in state.surface.values()
                       if u.side == SOLOMANI and u.cls.planetary_defense)
        army.location, battery.location = cell, cell
        # twenty-four bombardment factors, which is the lowest column the
        # table cannot return a dash on
        pooled = 0
        for squadron in state.naval.values():
            if squadron.side != IMPERIAL or squadron.cls.bombard == 0:
                continue
            if pooled >= 24:
                break
            squadron.location, squadron.mission = cell, BOMBARDMENT
            pooled += squadron.cls.bombard

        class PicksTheBattery(HeuristicAgent):
            def bombardment_target(self, s, side, where, candidates, engine):
                return battery

        engine = Engine(state, PicksTheBattery(IMPERIAL, seed=1),
                        HeuristicAgent(SOLOMANI, seed=2), rng=random.Random(61))
        engine.bombardment_phase([])
        self.assertGreater(battery.losses, 0)
        self.assertEqual(army.losses, 0)

    def test_the_guns_prefer_a_target_that_cannot_answer(self):
        """A bombarding squadron stands in the hex it is bombarding.

        Every battery within three hexes then fires on it at *full* factors,
        where fire at the close orbit box is halved -- so a target outside every
        battery's reach costs the fleet nothing, and one inside costs it a
        squadron a turn.  Given two targets of similar worth, the doctrine takes
        the quiet one.
        """
        state = new_game(seed=63)
        agent = HeuristicAgent(IMPERIAL, seed=1)
        agent._beachhead = None
        geo = state.geometry
        guns = agent._gun_map(state)
        covered = next(c for c in sorted(geo.land) if guns.get(c, 0) > 0)
        quiet = next(c for c in sorted(geo.land) if guns.get(c, 0) == 0)

        # one identical corps on each, so only the return fire separates them
        for cell in (covered, quiet):
            for unit in list(state.surface.values()):
                if unit.location == cell and unit.side == SOLOMANI:
                    del state.surface[unit.uid]
        from ie.engine import _add_surface
        corps = next(c for c, _n in oob.SOLOMANI_TROOPS if c.size == "corps")
        for cell in (covered, quiet):
            _add_surface(state, corps, SOLOMANI, cell)

        targets = agent._bombardment_targets(state)
        self.assertIn(quiet, targets)
        self.assertNotIn(covered, targets)

    def test_a_battery_is_worth_more_to_the_doctrine_than_its_combat_factor(self):
        """Twenty factors that shoot at the fleet every turn and freeze a city.

        The doctrine prices a battery against a field army in factors so the
        two can be compared at all, and the price rises as the battery nears
        elimination, because it fires at full strength until it dies.
        """
        state = new_game(seed=62)
        agent = HeuristicAgent(IMPERIAL, seed=1)
        battery = next(u for u in state.surface.values()
                       if u.side == SOLOMANI and u.cls.planetary_defense
                       and u.cls.bombard >= 5)
        fresh = agent._target_value(state, battery)
        battery.losses = 80
        self.assertGreater(agent._target_value(state, battery), fresh)

    def test_guerrillas_cannot_be_bombarded_from_orbit(self):
        """"Planetary bombardment attacks may not be made against guerrilla units"."""
        state = new_game(seed=42)
        engine = Engine(state, Agent(IMPERIAL, seed=1), Agent(SOLOMANI, seed=2),
                        rng=random.Random(42))
        cell = sorted(state.geometry.land)[0]
        guerrilla = next(cls for cls, _n in oob.SOLOMANI_GUERRILLAS)
        from ie.engine import _add_surface
        unit = _add_surface(state, guerrilla, SOLOMANI, cell)
        bomber = next(u for u in state.naval.values()
                      if u.side == IMPERIAL and u.cls.bombard > 0)
        bomber.location, bomber.mission = cell, BOMBARDMENT
        engine.bombardment_phase([])
        self.assertEqual(unit.losses, 0)


class TestLanding(unittest.TestCase):
    """The landing phase, and the price of arriving from orbit."""

    def setUp(self):
        self.state = new_game(seed=51)
        self.engine = Engine(self.state, Agent(IMPERIAL, seed=1),
                             Agent(SOLOMANI, seed=2), rng=random.Random(51))

    def _carrier_with(self, unit):
        carrier = next(u for u in self.state.naval.values()
                       if u.side == IMPERIAL and u.cls.capacity >= unit.current)
        carrier.location = CLOSE_ORBIT
        carrier.cargo.append(unit.uid)
        unit.carrier = carrier.uid
        unit.location = CLOSE_ORBIT
        return carrier

    def test_a_unit_may_not_be_unloaded_into_the_sea(self):
        unit = next(u for u in self.state.surface.values()
                    if u.side == IMPERIAL and u.current <= 20)
        self._carrier_with(unit)
        sea = sorted(self.state.geometry.deep_sea)[0]
        self.engine._unload(IMPERIAL, unit.uid, sea)
        self.assertIsNotNone(unit.carrier)

    def test_a_landed_unit_may_not_move_that_turn(self):
        """"A mobile surface unit may not move during its movement phase on the
        turn it is unloaded from a naval unit"."""
        unit = next(u for u in self.state.surface.values()
                    if u.side == IMPERIAL and u.current <= 20)
        self._carrier_with(unit)
        cell = sorted(self.state.geometry.land)[0]
        self.engine._unload(IMPERIAL, unit.uid, cell)
        self.assertEqual(unit.landed_turn, self.state.turn)

    def test_no_loading_or_unloading_in_deep_space(self):
        """"For the deep space box, no loading or unloading may occur"."""
        unit = next(u for u in self.state.surface.values()
                    if u.side == IMPERIAL and u.current <= 20)
        carrier = self._carrier_with(unit)
        carrier.location = DEEP_SPACE
        cell = sorted(self.state.geometry.land)[0]
        self.engine._unload(IMPERIAL, unit.uid, cell)
        self.assertIsNotNone(unit.carrier)

    def test_transport_capacity_is_enforced(self):
        state = self.state
        cruiser = next(u for u in state.naval.values()
                       if u.side == IMPERIAL and u.cls.kind == "CR")
        big = next(u for u in state.surface.values()
                   if u.side == IMPERIAL and u.cls.factor == 100)
        self.engine._load(IMPERIAL, big.uid, cruiser.uid)
        self.assertIsNone(big.carrier, "a cruiser lifts five factors, not a corps")

    def test_a_damaged_unit_is_lighter(self):
        """"The current strength of a surface unit is used for transport purposes."""
        state = self.state
        carrier = next(u for u in state.naval.values()
                       if u.side == IMPERIAL and u.cls.kind == "BR")
        corps = next(u for u in state.surface.values()
                     if u.side == IMPERIAL and u.cls.factor == 100)
        corps.losses = 80                      # current strength 20
        self.engine._load(IMPERIAL, corps.uid, carrier.uid)
        self.assertEqual(corps.carrier, carrier.uid)

    def test_marines_and_jump_troops_land_without_the_assault_penalty(self):
        for cls, _n in oob.IMPERIAL_MARINES:
            self.assertEqual(tables.ASSAULT_MODIFIER[cls.assault_class], 0)
        ordinary = next(cls for cls, _n in oob.IMPERIAL_TROOPS
                        if not cls.jump and not cls.marine)
        self.assertEqual(tables.ASSAULT_MODIFIER[ordinary.assault_class], -3)


class TestSurfaceCombat(unittest.TestCase):
    def setUp(self):
        self.state = new_game(seed=61)
        self.engine = Engine(self.state, HeuristicAgent(IMPERIAL, seed=1),
                             HeuristicAgent(SOLOMANI, seed=2),
                             rng=random.Random(61))

    def test_armour_doubles_and_elite_doubles_again(self):
        """"An elite armor unit would have its current strength doubled twice"."""
        from ie.state import SurfaceUnit
        plain = SurfaceUnit(1, oob.SurfaceClass("x", "corps", 100, 14),
                            IMPERIAL, 0)
        armoured = SurfaceUnit(2, oob.SurfaceClass("x", "corps", 100, 14,
                                                   armor=True), IMPERIAL, 0)
        both = SurfaceUnit(3, oob.SurfaceClass("x", "corps", 100, 14,
                                               armor=True, elite=True),
                           IMPERIAL, 0)
        self.assertEqual(plain.combat_strength(), 100)
        self.assertEqual(armoured.combat_strength(), 200)
        self.assertEqual(both.combat_strength(), 400)

    def test_a_mercenary_past_half_strength_fires_at_half(self):
        """"Any mercenary unit suffering over 50% losses has its current
        strength halved when firing during surface combat"."""
        from ie.state import SurfaceUnit
        cls = oob.SurfaceClass("m", "division", 20, 14, mercenary=True)
        unit = SurfaceUnit(1, cls, IMPERIAL, 0, losses=60)
        self.assertEqual(unit.current, 8)
        self.assertEqual(unit.combat_strength(), 4)

    def test_out_of_supply_halves_a_unit_when_fired_upon(self):
        from ie.state import SurfaceUnit
        cls = oob.SurfaceClass("x", "corps", 100, 14)
        unit = SurfaceUnit(1, cls, IMPERIAL, 0)
        self.assertEqual(unit.defence_strength(supplied=True), 100)
        self.assertEqual(unit.defence_strength(supplied=False), 50)

    def test_guerrillas_and_commandos_are_always_in_supply(self):
        state, engine = self.state, self.engine
        from ie.engine import _add_surface
        cell = sorted(state.geometry.land)[0]
        guerrilla = _add_surface(
            state, next(c for c, _n in oob.SOLOMANI_GUERRILLAS), SOLOMANI, cell)
        self.assertTrue(engine._in_supply(guerrilla))
        commando = next(u for u in state.surface.values() if u.cls.commando)
        self.assertTrue(engine._in_supply(commando))

    def test_an_imperial_unit_with_no_base_is_out_of_supply(self):
        state, engine = self.state, self.engine
        from ie.engine import _add_surface
        cell = sorted(state.geometry.land)[0]
        unit = _add_surface(state, next(c for c, _n in oob.IMPERIAL_TROOPS),
                            IMPERIAL, cell)
        state.bases.clear()
        self.assertFalse(engine._in_supply(unit))

    def test_a_base_within_five_hexes_supplies(self):
        state, engine = self.state, self.engine
        from ie.engine import _add_surface
        cell = sorted(state.geometry.land)[0]
        unit = _add_surface(state, next(c for c, _n in oob.IMPERIAL_TROOPS),
                            IMPERIAL, cell)
        state.bases.clear()
        base = Base(uid=state.new_uid(), location=cell)
        state.bases[base.uid] = base
        self.assertTrue(engine._in_supply(unit))

    def test_winter_shifts_the_die_roll_down(self):
        state = self.state
        cold = None
        for cell in state.geometry.land:
            lat, _lon = state.geometry.lat_lon[cell]
            if lat > 40 and state.geometry.terrain[cell] not in (
                    terra.CLEAR, terra.URBAN, terra.STARPORT):
                cold = cell
                break
        self.assertIsNotNone(cold)
        state.turn = 1 + 2 * 8                  # December
        self.assertEqual(state.month(), 12)
        self.assertTrue(state.in_winter(cold))

    def test_a_hiding_guerrilla_cannot_fire_and_is_harder_to_hit(self):
        self.assertEqual(tables.GUERRILLA_HIDING_MODIFIER, 3)
        state, engine = self.state, self.engine
        from ie.engine import _add_surface
        cell = next(c for c in sorted(state.geometry.land)
                    if state.geometry.terrain[c] in terra.GUERRILLA_COVER)
        guerrilla = _add_surface(
            state, next(c for c, _n in oob.SOLOMANI_GUERRILLAS), SOLOMANI, cell)
        _add_surface(state, next(c for c, _n in oob.IMPERIAL_TROOPS),
                     IMPERIAL, cell)
        hiding = engine.agents[SOLOMANI].guerrilla_hiding(state, SOLOMANI, engine)
        self.assertIn(guerrilla.uid, hiding)


class TestGarrisons(unittest.TestCase):
    """Rule 5's garrison rule, which is how the Imperium actually scores."""

    def test_a_corps_garrisons_its_whole_zone_of_control(self):
        state = new_game(seed=71)
        engine = Engine(state, Agent(IMPERIAL, seed=1), Agent(SOLOMANI, seed=2),
                        rng=random.Random(71))
        for unit in list(state.surface.values()):
            if unit.side == SOLOMANI:
                del state.surface[unit.uid]
        city = sorted(state.geometry.urban)[0]
        from ie.engine import _add_surface
        corps = next(c for c, _n in oob.IMPERIAL_TROOPS if c.size == "corps")
        _add_surface(state, corps, IMPERIAL, city)
        engine._update_garrisons()
        self.assertIn(city, state.garrisoned)

    def test_a_solomani_unit_prevents_garrisoning(self):
        state = new_game(seed=72)
        engine = Engine(state, Agent(IMPERIAL, seed=1), Agent(SOLOMANI, seed=2),
                        rng=random.Random(72))
        city = sorted(state.geometry.urban)[0]
        from ie.engine import _add_surface
        corps = next(c for c, _n in oob.IMPERIAL_TROOPS if c.size == "corps")
        _add_surface(state, corps, IMPERIAL, city)
        defender = next(c for c, _n in oob.SOLOMANI_TROOPS if c.size == "corps")
        _add_surface(state, defender, SOLOMANI, city)
        engine._update_garrisons()
        self.assertNotIn(city, state.garrisoned)

    def test_garrisoning_costs_the_solomani_a_replacement_point(self):
        """"The Solomani player receives one replacement point from each
        ungarrisoned urban hex"."""
        state = new_game(seed=73)
        engine = Engine(state, Agent(IMPERIAL, seed=1), Agent(SOLOMANI, seed=2),
                        rng=random.Random(73))
        state.replacement_points[SOLOMANI] = 0
        engine.initial_segment()
        self.assertEqual(state.replacement_points[SOLOMANI], 61)
        state.replacement_points[SOLOMANI] = 0
        state.garrisoned = set(sorted(state.geometry.urban)[:11])
        engine.initial_segment()
        self.assertEqual(state.replacement_points[SOLOMANI], 50)


class TestWithdrawal(unittest.TestCase):
    """Rule 6: two transport squadrons leave on turn 2, whatever it costs."""

    def test_two_transports_are_withdrawn_on_turn_two(self):
        state = new_game(seed=81)
        engine = Engine(state, Agent(IMPERIAL, seed=1), Agent(SOLOMANI, seed=2),
                        rng=random.Random(81))
        before = sum(1 for u in state.naval.values()
                     if u.side == IMPERIAL and u.cls.kind == "AR")
        self.assertEqual(before, 4)
        state.turn = 2
        engine._withdrawals()
        after = sum(1 for u in state.naval.values()
                    if u.side == IMPERIAL and u.cls.kind == "AR")
        self.assertEqual(after, 2)

    def test_the_empty_transports_go_first(self):
        """The player picks which two, and anything aboard leaves with them."""
        state = new_game(seed=82)
        engine = Engine(state, Agent(IMPERIAL, seed=1), Agent(SOLOMANI, seed=2),
                        rng=random.Random(82))
        transports = [u for u in state.naval.values()
                      if u.side == IMPERIAL and u.cls.kind == "AR"]
        passenger = next(u for u in state.surface.values()
                         if u.side == IMPERIAL and u.current <= 100)
        transports[0].cargo.append(passenger.uid)
        passenger.carrier = transports[0].uid
        state.turn = 2
        engine._withdrawals()
        self.assertIn(transports[0].uid, state.naval)
        self.assertFalse(passenger.dead)

    def test_the_doctrine_does_not_load_the_doomed_transports(self):
        state = new_game(seed=83)
        agent = HeuristicAgent(IMPERIAL, seed=1)
        engine = Engine(state, agent, HeuristicAgent(SOLOMANI, seed=2),
                        rng=random.Random(83))
        engine.play_turn()                        # turn 1
        loaded = [u for u in state.naval.values()
                  if u.cls.kind == "AR" and u.cargo]
        self.assertLessEqual(len(loaded), 2,
                             "loaded transports that are about to be withdrawn")


class TestReplacements(unittest.TestCase):
    def test_a_replacement_wave_costs_a_victory_point(self):
        state = new_game(seed=91)

        class Buyer(HeuristicAgent):
            def replacement_waves(self, s, engine):
                return 2

        engine = Engine(state, Buyer(IMPERIAL, seed=1),
                        HeuristicAgent(SOLOMANI, seed=2), rng=random.Random(91))
        before = state.victory_points()
        state.turn = 6
        engine._quarterly_replacements()
        self.assertEqual(state.waves_taken, 2)
        self.assertEqual(state.victory_points(), before - 2)

    def test_elite_units_are_never_rebuilt(self):
        """"Elite units may not be replaced or rebuilt"."""
        state = new_game(seed=92)
        engine = Engine(state, HeuristicAgent(IMPERIAL, seed=1),
                        HeuristicAgent(SOLOMANI, seed=2), rng=random.Random(92))
        elite = next(c for c, _n in oob.IMPERIAL_TROOPS if c.elite)
        state.pools["dead_surface"] = [elite]
        engine._rebuild_surface(IMPERIAL, 1000, 1.0)
        self.assertIn(elite, state.pools["dead_surface"])

    def test_pd_units_are_never_rebuilt(self):
        state = new_game(seed=93)
        engine = Engine(state, HeuristicAgent(IMPERIAL, seed=1),
                        HeuristicAgent(SOLOMANI, seed=2), rng=random.Random(93))
        pd = next(c for c, _n in oob.SOLOMANI_PD)
        state.pools["dead_surface"] = [pd]
        engine._rebuild_surface(SOLOMANI, 1000, 1.0)
        self.assertIn(pd, state.pools["dead_surface"])

    def test_guerrillas_arrive_four_at_a_time_in_the_first_two_quarters(self):
        state = new_game(seed=94)
        engine = Engine(state, HeuristicAgent(IMPERIAL, seed=1),
                        HeuristicAgent(SOLOMANI, seed=2), rng=random.Random(94))
        state.turn = 6
        engine._quarterly_guerrillas()
        self.assertEqual(sum(1 for u in state.surface.values()
                             if u.cls.guerrilla), 4)
        state.turn = 12
        engine._quarterly_guerrillas()
        self.assertEqual(sum(1 for u in state.surface.values()
                             if u.cls.guerrilla), 8)
        state.turn = 30
        engine._quarterly_guerrillas()
        self.assertEqual(sum(1 for u in state.surface.values()
                             if u.cls.guerrilla), 8)


class TestCloning(unittest.TestCase):
    """A clone has to be cheap and has to be a copy, or search is off the table."""

    def test_a_clone_is_independent(self):
        state = new_game(seed=101)
        copy = state.clone()
        unit = next(iter(copy.surface.values()))
        unit.losses = 55
        self.assertEqual(state.surface[unit.uid].losses, 0)
        copy.garrisoned.add(1)
        self.assertNotIn(1, state.garrisoned)

    def test_a_clone_shares_the_frozen_parts(self):
        state = new_game(seed=102)
        copy = state.clone()
        self.assertIs(copy.geometry, state.geometry)
        uid = next(iter(state.surface))
        self.assertIs(copy.surface[uid].cls, state.surface[uid].cls)

    def test_a_clone_plays_the_same_game(self):
        state = new_game(seed=103)
        copy = state.clone()
        for board in (state, copy):
            engine = Engine(board, HeuristicAgent(IMPERIAL, seed=1),
                            HeuristicAgent(SOLOMANI, seed=2),
                            rng=random.Random(7))
            for _ in range(3):
                engine.play_turn()
        self.assertEqual(len(state.surface), len(copy.surface))
        self.assertEqual(state.victory_points(), copy.victory_points())


class TestPlaythrough(unittest.TestCase):
    """Whole games, for the properties that only show up over a campaign."""

    def test_a_game_runs_to_a_result(self):
        state = new_game(seed=111)
        result = play(state, HeuristicAgent(IMPERIAL, seed=1),
                      HeuristicAgent(SOLOMANI, seed=2), max_turns=18,
                      rng=random.Random(111))
        self.assertIn(result, [
            "solomani major victory", "solomani marginal victory", "draw",
            "imperial marginal victory", "imperial major victory",
            "imperial decisive victory"])

    def test_the_random_agent_plays_legally(self):
        state = new_game(seed=112)
        play(state, RandomAgent(IMPERIAL, seed=1), RandomAgent(SOLOMANI, seed=2),
             max_turns=8, rng=random.Random(112))
        for unit in state.surface.values():
            if unit.carrier is None and isinstance(unit.location, int):
                self.assertIn(unit.location, state.geometry.land)

    def test_the_doctrine_gets_an_army_ashore(self):
        """The one thing an Imperial doctrine has to be able to do at all."""
        state = new_game(seed=113)
        engine = Engine(state, HeuristicAgent(IMPERIAL, seed=1),
                        HeuristicAgent(SOLOMANI, seed=2), rng=random.Random(113))
        peak = 0
        for _ in range(14):
            engine.play_turn()
            peak = max(peak, sum(u.current for u in state.surface.values()
                                 if u.side == IMPERIAL
                                 and isinstance(u.location, int)))
        self.assertGreater(peak, 100, "no army reached the surface")

    def test_the_rewrite_beats_the_doctrine_it_replaced(self):
        """The baseline is kept so this can be a measurement, not a memory.

        Both doctrines play the same defence on the same seeds.  The claim
        under test is the one the rewrite was for: the army reaches Terra.  The
        margin is large enough that four seeds settle it -- the old doctrine
        landed about six hundred factors and left two thousand out-system in
        every game measured.
        """
        def peak_and_stranded(agent_class):
            peak = stranded = 0.0
            for seed in (201, 202, 203, 204):
                state = new_game(seed=seed)
                engine = Engine(state, agent_class(IMPERIAL, seed=1),
                                BeachheadAgent(SOLOMANI, seed=2),
                                rng=random.Random(seed))
                for _ in range(18):
                    engine.play_turn()
                    peak = max(peak, sum(
                        u.current for u in state.surface.values()
                        if u.side == IMPERIAL and u.carrier is None
                        and isinstance(u.location, int)))
                stranded += sum(u.current for u in state.surface.values()
                                if u.side == IMPERIAL and not u.dead
                                and u.location == OUT_SYSTEM)
            return peak, stranded / 4

        new_peak, new_stranded = peak_and_stranded(HeuristicAgent)
        old_peak, old_stranded = peak_and_stranded(BeachheadAgent)
        self.assertGreater(new_peak, old_peak * 1.5)
        self.assertLess(new_stranded, old_stranded)

    def test_the_scripted_doctrine_goes_for_the_starports(self):
        """It should land nearer a starport than the open doctrine does.

        Not *on* one: a starport carries a planetary defence corps with a
        bombardment factor of nine, and landing inside its three-hex reach
        costs more than the starport is worth.  What the scripted plan buys is
        proximity -- the Solomani lay down a new SDB wing at every ungarrisoned
        starport each quarter, so a fleet that ignores them fights a navy that
        keeps growing.
        """
        def beachhead(agent_class):
            state = new_game(seed=114)
            agent = agent_class(IMPERIAL, seed=1)
            engine = Engine(state, agent, HeuristicAgent(SOLOMANI, seed=2),
                            rng=random.Random(114))
            for _ in range(6):
                engine.play_turn()
            self.assertIsNotNone(agent._beachhead)
            return min(state.geometry.distance(agent._beachhead, c)
                       for c in state.geometry.starports)

        self.assertLess(beachhead(ScriptedAgent), beachhead(HeuristicAgent),
                        "the scripted plan ignored the starports")

    def test_the_doctrine_lands_the_army_and_not_a_fraction_of_it(self):
        """The invasion's first job, and the one the first doctrine failed.

        The Imperium owns 4040 combat factors and has to put them on Terra
        through two surviving transport squadrons.  Landing six hundred of them
        and leaving the rest out-system is not a beachhead, it is a hostage.
        """
        state = new_game(seed=141)
        engine = Engine(state, HeuristicAgent(IMPERIAL, seed=1),
                        HeuristicAgent(SOLOMANI, seed=2), rng=random.Random(141))
        peak = 0.0
        for _ in range(20):
            engine.play_turn()
            peak = max(peak, sum(u.current for u in state.surface.values()
                                 if u.side == IMPERIAL and u.carrier is None
                                 and isinstance(u.location, int)))
        stranded = sum(u.current for u in state.surface.values()
                       if u.side == IMPERIAL and not u.dead
                       and u.location == OUT_SYSTEM)
        self.assertGreater(peak, 1200, "most of the army never landed")
        self.assertLess(stranded, 1200, "most of the army never sailed")

    def test_the_doctrine_lands_where_a_base_can_follow(self):
        """A lodgement that cannot hold a base is a lodgement out of supply.

        "A base may not be landed in a tundra or ice hex", an Imperial unit
        needs one within five hexes to be in supply, and a unit out of supply
        may not fire.  The score refuses those hexes outright rather than
        pricing them, which is also what keeps the invasion off the islands: a
        hex with no room around it cannot hold four thousand factors at a
        thousand to the hex.
        """
        state = new_game(seed=142)
        agent = HeuristicAgent(IMPERIAL, seed=1)
        engine = Engine(state, agent, HeuristicAgent(SOLOMANI, seed=2),
                        rng=random.Random(142))
        for _ in range(6):
            engine.play_turn()
        geo = state.geometry
        self.assertIsNotNone(agent._beachhead)
        self.assertNotIn(geo.terrain[agent._beachhead],
                         (terra.TUNDRA, terra.PERMANENT_ICE, terra.SEASONAL_ICE))
        room = sum(1 for c in geo.within(agent._beachhead, 2) if c in geo.land)
        self.assertGreaterEqual(room, 5, "landed on a rock")

    def test_the_supply_net_is_not_one_base_landed_six_times(self):
        state = new_game(seed=143)
        engine = Engine(state, HeuristicAgent(IMPERIAL, seed=1),
                        HeuristicAgent(SOLOMANI, seed=2), rng=random.Random(143))
        for _ in range(16):
            engine.play_turn()
        sites = [b.location for b in state.bases.values()
                 if isinstance(b.location, int)]
        if len(sites) > 1:
            self.assertGreater(len(set(sites)), 1,
                               "every base landed in the same hex")

    def test_a_game_is_reproducible_from_its_seed(self):
        def run():
            state = new_game(seed=115)
            play(state, HeuristicAgent(IMPERIAL, seed=1),
                 HeuristicAgent(SOLOMANI, seed=2), max_turns=10,
                 rng=random.Random(5))
            return state.victory_points(), len(state.surface), len(state.naval)
        self.assertEqual(run(), run())


class TestHumanInterface(unittest.TestCase):
    """The decisions a human is offered, and what happens to the answers.

    The contract is the one ``ffw.agents.human`` set: a callback that returns
    ``None`` defers to the doctrine, so a player can take one decision a turn
    and leave the rest to the staff.
    """

    def _play(self, ask, seed=81, turns=8, side=IMPERIAL):
        state = new_game(seed=seed)
        human = HumanAgent(side, ask=ask, seed=1)
        other = HeuristicAgent(SOLOMANI if side == IMPERIAL else IMPERIAL,
                               seed=2)
        pair = ((human, other) if side == IMPERIAL else (other, human))
        engine = Engine(state, pair[0], pair[1], rng=random.Random(seed))
        for _ in range(turns):
            engine.play_turn()
        return state, human, engine

    def test_a_silent_human_plays_exactly_the_doctrine(self):
        """The fallback has to be exact or nothing measured with it means much."""
        def played_by(agent_class, **kwargs):
            state = new_game(seed=82)
            engine = Engine(state, agent_class(IMPERIAL, seed=1, **kwargs),
                            HeuristicAgent(SOLOMANI, seed=2),
                            rng=random.Random(82))
            for _ in range(10):
                engine.play_turn()
            return (state.victory_points(), len(state.garrisoned),
                    sorted((u.location, round(u.current))
                           for u in state.surface.values()
                           if u.side == IMPERIAL and not u.dead
                           and isinstance(u.location, int)))

        self.assertEqual(played_by(HumanAgent, ask=lambda request: None),
                         played_by(HeuristicAgent))

    def test_the_human_is_asked_the_decisions_that_decide_the_campaign(self):
        seen = []
        self._play(lambda request: seen.append(request["kind"]))
        self.assertIn("lodgement", seen)
        self.assertIn("bombardment", seen)
        self.assertIn("move", seen)

    def test_the_human_chooses_the_lodgement(self):
        chosen = {}

        def ask(request):
            if request["kind"] != "lodgement":
                return None
            self.assertIn(request["recommended"], request["options"])
            # take the option the staff did not recommend
            pick = next(c for c in request["options"]
                        if c != request["recommended"])
            chosen.setdefault("cell", pick)
            return chosen["cell"]

        state, human, _engine = self._play(ask, seed=83, turns=5)
        self.assertTrue(chosen)
        self.assertEqual(human._beachhead, chosen["cell"])

    def test_the_human_moves_a_stack_and_can_hold_it(self):
        moved = {}

        def ask(request):
            if request["kind"] != "move" or moved:
                return None
            moved["origin"] = request["origin"]
            moved["target"] = request["options"][0]
            return moved["target"]

        state, _human, _engine = self._play(ask, seed=84, turns=9)
        self.assertTrue(moved, "no stack was ever offered a move")
        self.assertNotEqual(moved["origin"], moved["target"])

        held = {}

        def hold(request):
            if request["kind"] != "move":
                return None
            held.setdefault("origin", request["origin"])
            return "hold" if request["origin"] == held["origin"] else None

        state, _human, _engine = self._play(hold, seed=84, turns=9)
        self.assertTrue(held)

    def test_the_human_picks_the_bombardment_target(self):
        picked = {}

        def ask(request):
            if request["kind"] != "bombardment":
                return None
            picked["cell"] = request["options"][-1]
            return [picked["cell"]]

        state = new_game(seed=85)
        human = HumanAgent(IMPERIAL, ask=ask, seed=1)
        engine = Engine(state, human, HeuristicAgent(SOLOMANI, seed=2),
                        rng=random.Random(85))
        for _ in range(6):
            engine.play_turn()
        self.assertTrue(picked)
        self.assertEqual(human._bombardment_targets(state), [picked["cell"]])

    def test_the_defence_can_be_played_too(self):
        seen = []

        def ask(request):
            seen.append(request["kind"])
            return None

        state, _human, _engine = self._play(ask, seed=86, turns=8,
                                            side=SOLOMANI)
        self.assertIn("move", seen)


class TestDrawing(unittest.TestCase):
    """The renderer, where it can be checked without looking at it.

    Skipped rather than failed without matplotlib: drawing is not a dependency
    of playing, and the rules tests have to run anywhere.
    """

    @classmethod
    def setUpClass(cls):
        try:
            import matplotlib
            matplotlib.use("Agg")
            from ie import viz
        except ImportError as exc:                         # pragma: no cover
            raise unittest.SkipTest("matplotlib not installed: %s" % exc)
        cls.viz = viz

    def test_a_snapshot_records_what_is_on_the_board(self):
        state = new_game(seed=71)
        engine = Engine(state, HeuristicAgent(IMPERIAL, seed=1),
                        HeuristicAgent(SOLOMANI, seed=2), rng=random.Random(71))
        for _ in range(8):
            engine.play_turn()
        frame = self.viz.snapshot(state)
        self.assertEqual(frame.garrisoned, state.garrisoned)
        self.assertEqual(frame.turn, state.turn)
        for side in (IMPERIAL, SOLOMANI):
            drawn = sum(sides.get(side, 0.0) for sides in frame.surface.values())
            actual = sum(u.current for u in state.surface.values()
                         if u.side == side and u.carrier is None and not u.dead
                         and isinstance(u.location, int))
            self.assertAlmostEqual(drawn, actual, places=6, msg=side)

    def test_the_two_views_between_them_show_every_cell(self):
        """Two hemispheres, chosen back to back, so nothing is off the page."""
        state = new_game(seed=72)
        near, far = self.viz.focus(state)
        for a, b in zip(near, far):
            self.assertAlmostEqual(a, -b, places=9)
        seen = set()
        for centre in (near, far):
            for cell in hexmap.CELLS:
                if hexmap._dot(hexmap.unit_vector(cell), centre) > 0:
                    seen.add(cell)
        # only cells exactly on the terminator can fall between the two
        self.assertGreater(len(seen), hexmap.cell_count() - 12)

    def test_a_hemisphere_hides_what_is_behind_the_planet(self):
        state = new_game(seed=73)
        centre = hexmap.unit_vector(0)
        east, north = self.viz._basis(centre)
        x, y, depth = self.viz._project(centre, centre, east, north)
        self.assertAlmostEqual(x, 0.0, places=9)
        self.assertAlmostEqual(y, 0.0, places=9)
        self.assertAlmostEqual(depth, 1.0, places=9)
        antipode = tuple(-v for v in centre)
        self.assertLess(self.viz._project(antipode, centre, east, north)[2], 0)
        # a cell over the limb has its corners pulled back onto it
        for cell in hexmap.CELLS:
            if -0.05 < hexmap._dot(hexmap.unit_vector(cell), centre) < 0.05:
                for x, y in self.viz._clipped_outline(cell, centre, east, north):
                    self.assertLessEqual(math.hypot(x, y), 1.0 + 1e-9)

    def test_the_map_draws(self):
        state = new_game(seed=74)
        engine = Engine(state, HeuristicAgent(IMPERIAL, seed=1),
                        HeuristicAgent(SOLOMANI, seed=2), rng=random.Random(74))
        recorder = self.viz.GameRecorder()
        for _ in range(4):
            engine.play_turn()
            recorder(state)
        figure = self.viz.draw_map(state, recorder[-1])
        self.assertEqual(len(figure.axes), 3)     # two globes and the panel
        self.assertEqual(len(recorder), 4)
        self.viz.plt.close("all")


if __name__ == "__main__":
    unittest.main()

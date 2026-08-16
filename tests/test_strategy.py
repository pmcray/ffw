"""Tests for the shared layer: one feature vector, two games.

The claim this layer makes is narrow and testable: a position from either game
becomes a vector of the same length, each slot means the same thing in both,
and the meaning is the one the slot's name says.  These tests check that claim
rather than assuming it -- an encoding that silently means different things in
the two games would make transfer look like it works while teaching a network
nothing.
"""

from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                                   # noqa: E402

from strategy import (DIMENSIONS, FEATURE_NAMES, adapter_for,        # noqa: E402
                      adapters, describe, encode)


class TestAdapters(unittest.TestCase):
    def test_both_games_are_described(self):
        self.assertEqual(sorted(adapters()), ["ffw", "ie"])

    def test_each_adapter_names_two_opposed_sides(self):
        for name in ("ffw", "ie"):
            adapter = adapter_for(name)
            self.assertEqual(len(adapter.sides), 2)
            a, b = adapter.sides
            self.assertEqual(adapter.enemy_of(a), b)
            self.assertEqual(adapter.enemy_of(b), a)

    def test_adapters_are_cached_so_the_opening_is_measured_once(self):
        self.assertIs(adapter_for("ffw"), adapter_for("ffw"))

    def test_each_adapter_can_start_a_game(self):
        for name in ("ffw", "ie"):
            state = adapter_for(name).new_game(seed=1)
            self.assertEqual(state.turn, 1)

    def test_the_objectives_are_the_things_worth_holding(self):
        ffw = adapter_for("ffw")
        state = ffw.new_game(seed=1)
        self.assertGreater(len(ffw.objectives(state)), 30)

        ie = adapter_for("ie")
        state = ie.new_game(seed=1)
        self.assertEqual(len(ie.objectives(state)), 61)

    def test_the_imperium_starts_invasion_earth_holding_nothing(self):
        ie = adapter_for("ie")
        state = ie.new_game(seed=1)
        imperial, solomani = ie.sides
        self.assertEqual(ie.objectives_held(state, imperial), [])
        self.assertEqual(len(ie.objectives_held(state, solomani)), 61)

    def test_both_sides_hold_something_at_the_start_of_the_frontier_war(self):
        ffw = adapter_for("ffw")
        state = ffw.new_game(seed=1)
        for side in ffw.sides:
            self.assertGreater(len(ffw.objectives_held(state, side)), 5)

    def test_force_posture_adds_up(self):
        for name in ("ffw", "ie"):
            adapter = adapter_for(name)
            state = adapter.new_game(seed=2)
            for side in adapter.sides:
                forward, reserve, largest = adapter.force_posture(state, side)
                self.assertGreaterEqual(forward, 0.0)
                self.assertGreaterEqual(reserve, 0.0)
                self.assertLessEqual(largest, forward + 1e-6,
                                     "%s: a stack bigger than the whole force"
                                     % name)

    def test_the_whole_imperial_force_starts_in_reserve_in_invasion_earth(self):
        """"All forces are placed in the out-system box"."""
        ie = adapter_for("ie")
        state = ie.new_game(seed=3)
        forward, reserve, _largest = ie.force_posture(state, ie.sides[0])
        self.assertEqual(forward, 0.0)
        self.assertGreater(reserve, 0.0)

    def test_the_margin_is_antisymmetric(self):
        for name in ("ffw", "ie"):
            adapter = adapter_for(name)
            state = adapter.new_game(seed=4)
            a, b = adapter.sides
            self.assertAlmostEqual(adapter.normalised_margin(state, a),
                                   -adapter.normalised_margin(state, b),
                                   places=9)
            self.assertAlmostEqual(adapter.normalised_level(state, a),
                                   -adapter.normalised_level(state, b),
                                   places=9)

    def test_supply_health_is_a_fraction(self):
        for name in ("ffw", "ie"):
            adapter = adapter_for(name)
            state = adapter.new_game(seed=5)
            for side in adapter.sides:
                value = adapter.supply_health(state, side)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


class TestSharedEncoding(unittest.TestCase):
    """The vector, and what makes it the *same* vector in both games."""

    def test_the_names_and_the_width_agree(self):
        self.assertEqual(DIMENSIONS, len(FEATURE_NAMES))
        self.assertEqual(len(set(FEATURE_NAMES)), len(FEATURE_NAMES))

    def test_both_games_produce_the_same_width(self):
        vectors = []
        for name in ("ffw", "ie"):
            adapter = adapter_for(name)
            state = adapter.new_game(seed=6)
            for side in adapter.sides:
                vector = encode(adapter, state, side)
                self.assertEqual(len(vector), DIMENSIONS, name)
                vectors.append(vector)
        self.assertEqual(len({v.shape for v in vectors}), 1)

    def test_everything_is_finite_and_in_range(self):
        """A network fed an inf learns nothing; a network fed a 900 learns noise."""
        for name in ("ffw", "ie"):
            adapter = adapter_for(name)
            for seed in (7, 8):
                state = adapter.new_game(seed=seed)
                for side in adapter.sides:
                    vector = encode(adapter, state, side)
                    self.assertTrue(np.all(np.isfinite(vector)), name)
                    self.assertTrue(np.all(np.abs(vector) <= 1.0 + 1e-9),
                                    "%s out of range: %s" % (name, vector))

    def test_the_scoreboard_slots_are_antisymmetric_between_the_seats(self):
        """The whole point of a perspective encoding: one network, both seats."""
        margin = FEATURE_NAMES.index("score_margin")
        level = FEATURE_NAMES.index("score_level")
        for name in ("ffw", "ie"):
            adapter = adapter_for(name)
            state = adapter.new_game(seed=9)
            a, b = (encode(adapter, state, s) for s in adapter.sides)
            self.assertAlmostEqual(a[margin], -b[margin], places=9, msg=name)
            self.assertAlmostEqual(a[level], -b[level], places=9, msg=name)

    def test_holding_more_objectives_raises_the_share_slot(self):
        """The slot has to mean what it says, in the game it is read from."""
        index = FEATURE_NAMES.index("objective_share")
        ie = adapter_for("ie")
        state = ie.new_game(seed=10)
        imperial = ie.sides[0]
        before = encode(ie, state, imperial)[index]
        state.garrisoned = set(sorted(state.geometry.urban)[:30])
        after = encode(ie, state, imperial)[index]
        self.assertGreater(after, before)

    def test_the_elapsed_slot_rises_with_the_turn(self):
        index = FEATURE_NAMES.index("elapsed")
        for name in ("ffw", "ie"):
            adapter = adapter_for(name)
            state = adapter.new_game(seed=11)
            early = encode(adapter, state, adapter.sides[0])[index]
            state.turn = adapter.scheduled_length
            late = encode(adapter, state, adapter.sides[0])[index]
            self.assertGreater(late, early, name)

    def test_losing_force_lowers_the_remaining_slot(self):
        index = FEATURE_NAMES.index("force_remaining")
        ie = adapter_for("ie")
        state = ie.new_game(seed=12)
        imperial = ie.sides[0]
        before = encode(ie, state, imperial)[index]
        for unit in list(state.naval.values()):
            if unit.side == imperial:
                del state.naval[unit.uid]
        after = encode(ie, state, imperial)[index]
        self.assertLess(after, before)

    def test_committing_troops_moves_the_forward_slot(self):
        index = FEATURE_NAMES.index("force_forward")
        ie = adapter_for("ie")
        state = ie.new_game(seed=13)
        imperial = ie.sides[0]
        before = encode(ie, state, imperial)[index]
        cell = sorted(state.geometry.land)[0]
        moved = 0
        for unit in state.surface.values():
            if unit.side == imperial and moved < 6:
                unit.location, unit.carrier = cell, None
                moved += 1
        after = encode(ie, state, imperial)[index]
        self.assertGreater(after, before)

    def test_the_two_games_do_not_encode_to_the_same_point(self):
        """If they did, the encoding would be describing nothing."""
        opening = {}
        for name in ("ffw", "ie"):
            adapter = adapter_for(name)
            state = adapter.new_game(seed=14)
            opening[name] = encode(adapter, state, adapter.sides[0])
        self.assertGreater(float(np.linalg.norm(opening["ffw"] - opening["ie"])),
                           0.5)

    def test_describe_prints_every_slot(self):
        adapter = adapter_for("ie")
        state = adapter.new_game(seed=15)
        text = describe(encode(adapter, state, adapter.sides[0]))
        for name in FEATURE_NAMES:
            self.assertIn(name, text)


class TestEncodingUnderPlay(unittest.TestCase):
    """The encoding has to keep working once the board stops being the opening."""

    def test_it_survives_a_played_out_frontier_war(self):
        from ffw.agents import ScriptedAgent
        from ffw.state import IMPERIAL, ZHODANI
        adapter = adapter_for("ffw")
        state = adapter.new_game(seed=16)
        adapter.play(state, {IMPERIAL: ScriptedAgent(IMPERIAL, seed=1),
                             ZHODANI: ScriptedAgent(ZHODANI, seed=2)},
                     max_turns=12)
        for side in adapter.sides:
            vector = encode(adapter, state, side)
            self.assertTrue(np.all(np.isfinite(vector)))
            self.assertTrue(np.all(np.abs(vector) <= 1.0 + 1e-9))

    def test_it_survives_a_played_out_invasion(self):
        from ie.agents import HeuristicAgent
        from ie.state import IMPERIAL, SOLOMANI
        adapter = adapter_for("ie")
        state = adapter.new_game(seed=17)
        adapter.play(state, {IMPERIAL: HeuristicAgent(IMPERIAL, seed=1),
                             SOLOMANI: HeuristicAgent(SOLOMANI, seed=2)},
                     max_turns=10, rng=random.Random(17))
        for side in adapter.sides:
            vector = encode(adapter, state, side)
            self.assertTrue(np.all(np.isfinite(vector)))
            self.assertTrue(np.all(np.abs(vector) <= 1.0 + 1e-9))

    def test_the_encoding_actually_moves_as_a_game_is_played(self):
        """A feature vector that never changes is not describing the game."""
        from ie.agents import HeuristicAgent
        from ie.engine import Engine
        from ie.state import IMPERIAL, SOLOMANI
        adapter = adapter_for("ie")
        state = adapter.new_game(seed=18)
        engine = Engine(state, HeuristicAgent(IMPERIAL, seed=1),
                        HeuristicAgent(SOLOMANI, seed=2), rng=random.Random(18))
        opening = encode(adapter, state, IMPERIAL)
        for _ in range(8):
            engine.play_turn()
        later = encode(adapter, state, IMPERIAL)
        self.assertGreater(float(np.linalg.norm(later - opening)), 0.2)


if __name__ == "__main__":
    unittest.main()

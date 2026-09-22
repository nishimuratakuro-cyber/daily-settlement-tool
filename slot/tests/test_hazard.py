"""ゾーンを考慮した天井期待値（ハザード関数版）のテスト。

ハザード一定の閉じた式（ceiling_ev）は、天国やゾーンのある機種では大きくずれる。
その厳密版が正しく、かつ一定ハザードでは閉じた式と一致することを確かめる。
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slot_core import ev  # noqa: E402

CEILING = 800
BASE = 1 / 400
LOSS = 2.0
AVERAGE_PAYOUT = 600.0
CEILING_PAYOUT = 900.0


def flat_model():
    return ev.CeilingModel(
        ceiling_games=CEILING, hit_probability=BASE,
        average_payout=AVERAGE_PAYOUT, ceiling_payout=CEILING_PAYOUT,
        net_loss_per_game=LOSS,
    )


def front_loaded(target):
    return ev.calibrate_hazard(
        lambda base: ev.zone_hazard(base, heaven_games=32, heaven_multiplier=20,
                                    zone_every=100, zone_multiplier=10),
        target, CEILING,
    )


class TestZoneHazard(unittest.TestCase):
    def test_multipliers_apply_where_expected(self):
        hazard = ev.zone_hazard(0.001, heaven_games=32, heaven_multiplier=20,
                                zone_every=100, zone_multiplier=10)
        self.assertAlmostEqual(hazard(1), 0.020)     # 天国区間
        self.assertAlmostEqual(hazard(32), 0.020)    # 天国の端
        self.assertAlmostEqual(hazard(33), 0.001)    # 天国を抜けた
        self.assertAlmostEqual(hazard(100), 0.010)   # ゾーン
        self.assertAlmostEqual(hazard(101), 0.001)   # ゾーン外

    def test_probability_is_capped_at_one(self):
        hazard = ev.zone_hazard(0.2, heaven_games=10, heaven_multiplier=100)
        self.assertEqual(hazard(5), 1.0)

    def test_without_zones_it_is_constant(self):
        hazard = ev.zone_hazard(0.005)
        for game in (1, 50, 100, 777):
            self.assertAlmostEqual(hazard(game), 0.005)

    def test_invalid_arguments_rejected(self):
        with self.assertRaises(ValueError):
            ev.zone_hazard(0.0)
        with self.assertRaises(ValueError):
            ev.zone_hazard(0.001, heaven_games=-1)
        with self.assertRaises(ValueError):
            ev.zone_hazard(0.001, heaven_multiplier=0)


class TestTotalHitProbability(unittest.TestCase):
    def test_matches_geometric_formula(self):
        for games in (1, 10, 100, 800):
            expected = 1 - (1 - BASE) ** games
            self.assertAlmostEqual(
                ev.total_hit_probability(lambda g: BASE, games), expected, places=12
            )

    def test_respects_starting_point(self):
        full = ev.total_hit_probability(lambda g: BASE, CEILING)
        partial = ev.total_hit_probability(lambda g: BASE, CEILING, current_games=400)
        self.assertLess(partial, full)
        self.assertAlmostEqual(partial, 1 - (1 - BASE) ** 400, places=12)

    def test_accepts_a_sequence(self):
        self.assertAlmostEqual(
            ev.total_hit_probability([0.5, 0.5, 0.5], 3), 1 - 0.5**3
        )


class TestCalibrateHazard(unittest.TestCase):
    def test_hits_the_target_rate(self):
        target = ev.total_hit_probability(lambda g: BASE, CEILING)
        hazard = front_loaded(target)
        self.assertAlmostEqual(
            ev.total_hit_probability(hazard, CEILING), target, places=6
        )

    def test_shape_is_preserved(self):
        hazard = front_loaded(ev.total_hit_probability(lambda g: BASE, CEILING))
        self.assertAlmostEqual(hazard(1) / hazard(50), 20.0, places=6)
        self.assertAlmostEqual(hazard(100) / hazard(50), 10.0, places=6)

    def test_invalid_target_rejected(self):
        with self.assertRaises(ValueError):
            ev.calibrate_hazard(lambda base: ev.zone_hazard(base), 1.0, CEILING)


class TestHazardCeilingEv(unittest.TestCase):
    def test_constant_hazard_matches_the_closed_form(self):
        model = flat_model()
        for start in (0, 1, 100, 400, 799):
            closed = ev.ceiling_ev(model, start)
            exact = ev.hazard_ceiling_ev(
                lambda g: BASE, CEILING, start,
                average_payout=AVERAGE_PAYOUT, ceiling_payout=CEILING_PAYOUT,
                net_loss_per_game=LOSS,
            )
            for key in ("自力当選率", "天井到達率", "期待消化ゲーム数", "期待差枚"):
                self.assertAlmostEqual(closed[key], exact[key], places=9, msg=f"{key}@{start}G")

    def test_front_loading_flips_the_sign_of_the_error(self):
        model = flat_model()
        hazard = front_loaded(ev.total_hit_probability(lambda g: BASE, CEILING))

        def error(start):
            exact = ev.hazard_ceiling_ev(
                hazard, CEILING, start, average_payout=AVERAGE_PAYOUT,
                ceiling_payout=CEILING_PAYOUT, net_loss_per_game=LOSS,
            )["期待差枚"]
            return exact - ev.ceiling_ev(model, start)["期待差枚"]

        # 0G では当選が前方に寄るぶん近似は過小評価、ゾーンを抜けた直後は過大評価
        self.assertGreater(error(0), 100)
        self.assertLess(error(100), -50)

    def test_front_loading_shortens_expected_games(self):
        hazard = front_loaded(ev.total_hit_probability(lambda g: BASE, CEILING))
        exact = ev.hazard_ceiling_ev(hazard, CEILING, 0)["期待消化ゲーム数"]
        flat = ev.ceiling_ev(flat_model(), 0)["期待消化ゲーム数"]
        self.assertLess(exact, flat)

    def test_probabilities_sum_to_one(self):
        hazard = front_loaded(ev.total_hit_probability(lambda g: BASE, CEILING))
        result = ev.hazard_ceiling_ev(hazard, CEILING, 0)
        self.assertAlmostEqual(result["自力当選率"] + result["天井到達率"], 1.0, places=12)

    def test_accepts_a_sequence_of_probabilities(self):
        result = ev.hazard_ceiling_ev([0.5, 0.5], 2, average_payout=100, ceiling_payout=100)
        self.assertAlmostEqual(result["自力当選率"], 0.75)
        self.assertAlmostEqual(result["天井到達率"], 0.25)

    def test_invalid_arguments_rejected(self):
        with self.assertRaises(ValueError):
            ev.hazard_ceiling_ev(lambda g: BASE, 0)
        with self.assertRaises(ValueError):
            ev.hazard_ceiling_ev(lambda g: BASE, 100, current_games=101)


class TestHazardBreakeven(unittest.TestCase):
    def test_zones_push_the_line_much_deeper(self):
        hazard = front_loaded(ev.total_hit_probability(lambda g: BASE, CEILING))
        flat_line = ev.breakeven_start_games(flat_model())
        zoned_line = ev.hazard_breakeven_start_games(
            hazard, CEILING, AVERAGE_PAYOUT, CEILING_PAYOUT, LOSS
        )
        self.assertIsNotNone(zoned_line)
        # 一定近似を信じると、実際より 100G 以上浅い地点から打ち始めてしまう
        self.assertGreater(zoned_line - flat_line, 100)

    def test_line_never_dips_negative_afterwards(self):
        hazard = front_loaded(ev.total_hit_probability(lambda g: BASE, CEILING))
        line = ev.hazard_breakeven_start_games(
            hazard, CEILING, AVERAGE_PAYOUT, CEILING_PAYOUT, LOSS
        )
        for games in range(line, CEILING, 10):
            value = ev.hazard_ceiling_ev(
                hazard, CEILING, games, average_payout=AVERAGE_PAYOUT,
                ceiling_payout=CEILING_PAYOUT, net_loss_per_game=LOSS,
            )["期待差枚"]
            self.assertGreaterEqual(value, 0, f"{games}G でマイナスに戻っている")

    def test_hopeless_model_returns_none(self):
        self.assertIsNone(
            ev.hazard_breakeven_start_games(lambda g: BASE, CEILING, 1.0, 1.0, LOSS)
        )

    def test_invalid_step_rejected(self):
        with self.assertRaises(ValueError):
            ev.hazard_breakeven_start_games(lambda g: BASE, CEILING, 600, 900, LOSS, step=0)

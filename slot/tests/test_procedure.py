"""立ち回り情報の「機械割○%」を実額へ落とす換算のテスト。"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slot_core import ev  # noqa: E402

NON_EQUAL = ev.Exchange.from_medals_per_100yen(5.6)


class TestHourlyFromPayoutRate(unittest.TestCase):
    def test_equal_exchange_is_36000_times(self):
        for rate in (1.02, 1.05, 1.07, 1.10):
            self.assertAlmostEqual(
                ev.hourly_from_payout_rate(rate), 36_000 * (rate - 1), places=6
            )

    def test_known_values(self):
        self.assertAlmostEqual(ev.hourly_from_payout_rate(1.02), 720.0)
        self.assertAlmostEqual(ev.hourly_from_payout_rate(1.05), 1800.0)
        self.assertAlmostEqual(ev.hourly_from_payout_rate(1.07), 2520.0)

    def test_percentage_and_ratio_agree(self):
        self.assertAlmostEqual(
            ev.hourly_from_payout_rate(105), ev.hourly_from_payout_rate(1.05)
        )

    def test_non_equal_exchange_scales_down(self):
        ratio = NON_EQUAL.payout_yen / NON_EQUAL.rental_yen
        self.assertAlmostEqual(
            ev.hourly_from_payout_rate(1.05, exchange=NON_EQUAL),
            1800.0 * ratio,
            places=6,
        )

    def test_below_hundred_percent_is_negative(self):
        self.assertLess(ev.hourly_from_payout_rate(0.97), 0.0)

    def test_invalid_rate_rejected(self):
        with self.assertRaises(ValueError):
            ev.hourly_from_payout_rate(0)


class TestBreakevenPayoutRate(unittest.TestCase):
    def test_equal_exchange_breaks_even_at_one(self):
        self.assertAlmostEqual(ev.breakeven_payout_rate(ev.Exchange(), 2.0), 1.0)

    def test_non_equal_exchange_raises_the_bar(self):
        # 差 2.14円 / 換金 17.86円 = 0.12。純減2枚/G なら 1 + 0.12×2/3 = 1.08
        self.assertAlmostEqual(
            ev.breakeven_payout_rate(NON_EQUAL, 2.0), 1.08, places=3
        )
        self.assertAlmostEqual(
            ev.breakeven_payout_rate(NON_EQUAL, 1.5), 1.06, places=3
        )
        self.assertAlmostEqual(
            ev.breakeven_payout_rate(NON_EQUAL, 3.0), 1.12, places=3
        )

    def test_monotonic_in_loss_rate(self):
        rates = [ev.breakeven_payout_rate(NON_EQUAL, loss) for loss in (1.0, 1.5, 2.0, 2.5)]
        self.assertEqual(rates, sorted(rates))

    def test_zero_loss_needs_only_hundred_percent(self):
        self.assertAlmostEqual(ev.breakeven_payout_rate(NON_EQUAL, 0.0), 1.0)

    def test_negative_loss_rejected(self):
        with self.assertRaises(ValueError):
            ev.breakeven_payout_rate(NON_EQUAL, -1.0)


class TestProcedureValue(unittest.TestCase):
    def test_short_segment_has_high_rate_but_small_yen(self):
        # 機械割107%を41Gだけ消化する手順
        result = ev.procedure_value(1.07, 41)
        self.assertAlmostEqual(result["期待差枚"], 3 * 41 * 0.07)
        self.assertAlmostEqual(result["期待収支(円)"], 3 * 41 * 0.07 * 20)
        self.assertLess(result["期待収支(円)"], 200)      # 実額は 200円 未満
        self.assertAlmostEqual(result["名目時給(円)"], 2520.0)

    def test_nominal_hourly_is_independent_of_segment_length(self):
        short = ev.procedure_value(1.05, 41)
        long = ev.procedure_value(1.05, 400)
        self.assertAlmostEqual(short["名目時給(円)"], long["名目時給(円)"])
        self.assertLess(short["期待収支(円)"], long["期待収支(円)"])

    def test_overhead_reduces_effective_hourly(self):
        without = ev.procedure_value(1.07, 41, overhead_minutes=0)
        with_overhead = ev.procedure_value(1.07, 41, overhead_minutes=6)
        self.assertAlmostEqual(without["実効時給(円)"], without["名目時給(円)"], places=6)
        self.assertLess(with_overhead["実効時給(円)"], without["実効時給(円)"] / 2)

    def test_breakeven_comparison_is_added_on_request(self):
        result = ev.procedure_value(1.07, 41, exchange=NON_EQUAL, loss_medals_per_game=2.0)
        self.assertAlmostEqual(result["現金投資の損益分岐機械割"], 1.08, places=3)
        self.assertLess(result["損益分岐との差"], 0)  # 107% では 108% に届かない

    def test_breakeven_keys_absent_by_default(self):
        self.assertNotIn("現金投資の損益分岐機械割", ev.procedure_value(1.05, 200))

    def test_invalid_arguments_rejected(self):
        with self.assertRaises(ValueError):
            ev.procedure_value(1.05, 0)
        with self.assertRaises(ValueError):
            ev.procedure_value(1.05, 100, overhead_minutes=-1)


if __name__ == "__main__":
    unittest.main()

"""slot_core の単体テスト。

標準ライブラリのみで動くので、pytest が無い環境でも
``python3 -m unittest discover -s slot/tests`` で実行できる。
"""

from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slot_core import bayes, ev, ledger  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
MACHINES_CSV = os.path.join(DATA_DIR, "machines_sample.csv")
LEDGER_CSV = os.path.join(DATA_DIR, "ledger_sample.csv")


class TestParsing(unittest.TestCase):
    def test_probability_formats_agree(self):
        self.assertAlmostEqual(bayes.parse_probability("1/149.3"), 1 / 149.3)
        self.assertAlmostEqual(bayes.parse_probability("149.3"), 1 / 149.3)
        self.assertAlmostEqual(bayes.parse_probability(0.0067), 0.0067)
        self.assertAlmostEqual(bayes.parse_probability(" 1 / 6.35 "), 1 / 6.35)

    def test_blank_becomes_nan(self):
        self.assertTrue(math.isnan(bayes.parse_probability("")))
        self.assertTrue(math.isnan(bayes.parse_probability("-")))

    def test_invalid_probability_rejected(self):
        with self.assertRaises(ValueError):
            bayes.parse_probability("0")
        with self.assertRaises(ValueError):
            bayes.parse_probability(True)

    def test_payout_rate_formats_agree(self):
        self.assertAlmostEqual(bayes.parse_payout_rate("105.5"), 1.055)
        self.assertAlmostEqual(bayes.parse_payout_rate("105.5%"), 1.055)
        self.assertAlmostEqual(bayes.parse_payout_rate(1.055), 1.055)


class TestMachineTable(unittest.TestCase):
    def setUp(self):
        self.machines = bayes.load_machines(MACHINES_CSV)

    def test_loads_both_sample_machines(self):
        self.assertEqual(len(self.machines), 2)
        spec = self.machines["サンプルAタイプ（ダミー値）"]
        self.assertEqual(spec.settings, ("設定1", "設定2", "設定3", "設定4", "設定5", "設定6"))
        self.assertIn("ぶどう", spec.events)

    def test_payout_rates_are_ratios(self):
        spec = self.machines["サンプルAタイプ（ダミー値）"]
        self.assertIsNotNone(spec.payout_rates)
        self.assertAlmostEqual(spec.payout_rates[0], 0.97)
        self.assertAlmostEqual(spec.payout_rates[5], 1.06)

    def test_unknown_event_raises(self):
        spec = self.machines["サンプルAタイプ（ダミー値）"]
        with self.assertRaises(KeyError):
            spec.probabilities("存在しない役")


class TestPosterior(unittest.TestCase):
    def setUp(self):
        self.spec = bayes.load_machines(MACHINES_CSV)["サンプルAタイプ（ダミー値）"]

    def test_posterior_sums_to_one(self):
        result = bayes.posterior(self.spec, {"BIG": 20, "REG": 20, "ぶどう": 1300}, 8000)
        self.assertAlmostEqual(sum(result.posterior), 1.0)

    def test_high_setting_data_favours_high_setting(self):
        # 設定6 相当の出現率をそのまま観測したケース
        counts = {"BIG": round(8000 / 219.9), "REG": round(8000 / 255.0), "ぶどう": round(8000 / 5.92)}
        result = bayes.posterior(self.spec, counts, 8000)
        best_setting, _ = result.best()
        self.assertEqual(best_setting, "設定6")
        self.assertGreater(bayes.high_setting_probability(result), 0.9)

    def test_low_setting_data_favours_low_setting(self):
        counts = {"BIG": round(8000 / 273.1), "REG": round(8000 / 439.8), "ぶどう": round(8000 / 6.35)}
        result = bayes.posterior(self.spec, counts, 8000)
        self.assertIn(result.best()[0], {"設定1", "設定2"})
        # 8000G では設定4以上を 完全には否定できないが、低設定側に大きく寄る
        self.assertLess(bayes.high_setting_probability(result), 0.2)
        self.assertGreater(result.posterior[0] + result.posterior[1], 0.5)

    def test_zero_prior_excludes_setting(self):
        prior = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0]  # 1 と 6 しか使わないホール
        result = bayes.posterior(self.spec, {"BIG": 30, "REG": 30}, 8000, prior=prior)
        self.assertEqual(result.posterior[1], 0.0)
        self.assertAlmostEqual(result.posterior[0] + result.posterior[5], 1.0)

    def test_events_without_analysis_values_are_skipped(self):
        spec = bayes.MachineSpec(
            name="欠損あり",
            settings=("設定1", "設定6"),
            events={"BIG": (1 / 300, 1 / 200), "未入力": (math.nan, math.nan)},
        )
        result = bayes.posterior(spec, {"BIG": 30, "未入力": 10}, 6000)
        self.assertEqual(result.used_events, ("BIG",))

    def test_invalid_counts_rejected(self):
        with self.assertRaises(ValueError):
            bayes.posterior(self.spec, {"BIG": 10}, 0)
        with self.assertRaises(ValueError):
            bayes.posterior(self.spec, {"BIG": 9000}, 8000)
        with self.assertRaises(KeyError):
            bayes.posterior(self.spec, {"存在しない役": 1}, 8000)

    def test_expected_payout_rate_between_extremes(self):
        result = bayes.posterior(self.spec, {"BIG": 30, "REG": 25, "ぶどう": 1300}, 8000)
        rate = bayes.expected_payout_rate(result, self.spec)
        self.assertGreater(rate, 0.97)
        self.assertLess(rate, 1.06)

    def test_observed_rates(self):
        rows = bayes.observed_rates({"BIG": 20}, 8000)
        self.assertAlmostEqual(rows[0]["実測1/N"], 400.0)


class TestExchange(unittest.TestCase):
    def test_equal_exchange_is_symmetric(self):
        exchange = ev.Exchange()
        self.assertTrue(exchange.is_even)
        self.assertAlmostEqual(exchange.cash(100), 2000)
        self.assertAlmostEqual(exchange.cash(-100), -2000)

    def test_non_equal_exchange_is_asymmetric(self):
        exchange = ev.Exchange.from_medals_per_100yen(5.6)
        self.assertAlmostEqual(exchange.payout_yen, 100 / 5.6)
        self.assertAlmostEqual(exchange.cash(100), 100 * 100 / 5.6)
        self.assertAlmostEqual(exchange.cash(-100), -2000)  # 負け分は貸出単価

    def test_invalid_exchange_rejected(self):
        with self.assertRaises(ValueError):
            ev.Exchange(rental_yen=20.0, payout_yen=21.0)
        with self.assertRaises(ValueError):
            ev.Exchange.from_medals_per_100yen(0)


class TestHourlyBalance(unittest.TestCase):
    def test_percentage_and_ratio_agree(self):
        as_ratio = ev.hourly_balance(1.05, 800)
        as_percent = ev.hourly_balance(105, 800)
        self.assertAlmostEqual(as_ratio["差枚/時"], as_percent["差枚/時"])
        self.assertAlmostEqual(as_ratio["差枚/時"], 120.0)
        self.assertAlmostEqual(as_ratio["収支/時(換金単価評価)"], 2400.0)

    def test_non_equal_exchange_lowers_upside(self):
        exchange = ev.Exchange.from_medals_per_100yen(5.6)
        result = ev.hourly_balance(1.05, 800, exchange=exchange)
        self.assertLess(result["収支/時(換金単価評価)"], result["収支/時(貸出単価評価)"])


class TestSessionModel(unittest.TestCase):
    def setUp(self):
        self.model = ev.SessionModel(
            hit_probability=1 / 300, average_payout=500, payout_sd=200, net_loss_per_game=2.0
        )

    def test_coin_persistence_conversion(self):
        model = ev.SessionModel.from_coin_persistence(25, hit_probability=1 / 300, average_payout=500)
        self.assertAlmostEqual(model.net_loss_per_game, 2.0)

    def test_simulation_mean_matches_analytic(self):
        analytic = self.model.expected_diff(3000)
        diffs = ev.simulate_diffs(self.model, 3000, trials=5000, seed=42)
        self.assertAlmostEqual(analytic, -1000.0)
        self.assertLess(abs(sum(diffs) / len(diffs) - analytic), 150)

    def test_simulation_is_reproducible(self):
        first = ev.simulate_diffs(self.model, 1000, trials=200, seed=7)
        second = ev.simulate_diffs(self.model, 1000, trials=200, seed=7)
        self.assertEqual(first, second)

    def test_zero_sd_is_deterministic_per_hit(self):
        model = ev.SessionModel(hit_probability=1 / 300, average_payout=500, payout_sd=0.0)
        diffs = ev.simulate_diffs(model, 300, trials=50, seed=1)
        for value in diffs:
            remainder = (value + 300 * model.net_loss_per_game) % 500
            self.assertAlmostEqual(remainder, 0.0)

    def test_invalid_model_rejected(self):
        with self.assertRaises(ValueError):
            ev.SessionModel(hit_probability=0, average_payout=500)
        with self.assertRaises(ValueError):
            ev.SessionModel(hit_probability=1 / 300, average_payout=-1)


class TestSessionStats(unittest.TestCase):
    def test_quantiles_and_win_rate(self):
        diffs = [-1000.0, -200.0, 0.0, 300.0, 2000.0]
        stats = ev.session_stats(diffs)
        self.assertAlmostEqual(stats["中央値差枚"], 0.0)
        self.assertAlmostEqual(stats["勝率"], 0.4)
        self.assertAlmostEqual(stats["期待差枚"], 220.0)

    def test_non_equal_exchange_reduces_ev(self):
        diffs = [-1000.0, 1000.0]
        exchange = ev.Exchange.from_medals_per_100yen(5.6)
        stats = ev.session_stats(diffs, exchange=exchange)
        self.assertLess(stats["期待収支(円)"], 0.0)  # 差枚±0 でも非等価なら負ける
        self.assertLess(stats["交換率による目減り(円)"], 0.0)

    def test_bankroll_shortfall_probability(self):
        diffs = [-3000.0, -100.0, 500.0, 900.0]
        stats = ev.session_stats(diffs, bankroll_yen=40000)
        self.assertAlmostEqual(stats["資金ショート確率"], 0.25)

    def test_empty_sample_rejected(self):
        with self.assertRaises(ValueError):
            ev.session_stats([])


class TestCeiling(unittest.TestCase):
    def setUp(self):
        self.model = ev.CeilingModel(
            ceiling_games=1000,
            hit_probability=0.01,
            average_payout=500,
            ceiling_payout=800,
            net_loss_per_game=2.0,
        )

    def test_expected_games_matches_direct_sum(self):
        result = ev.ceiling_ev(self.model, current_games=900)
        remaining = 100
        probability = self.model.hit_probability
        miss = 1 - probability
        direct = sum(
            k * probability * miss ** (k - 1) for k in range(1, remaining)
        ) + remaining * miss ** (remaining - 1)
        self.assertAlmostEqual(result["期待消化ゲーム数"], direct, places=6)

    def test_deeper_start_is_better(self):
        shallow = ev.ceiling_ev(self.model, 0)["期待差枚"]
        deep = ev.ceiling_ev(self.model, 900)["期待差枚"]
        self.assertLess(shallow, deep)

    def test_breakeven_line_is_a_crossing_point(self):
        line = ev.breakeven_start_games(self.model)
        self.assertIsNotNone(line)
        self.assertGreaterEqual(ev.ceiling_ev(self.model, line)["期待差枚"], 0)
        if line > 0:
            self.assertLess(ev.ceiling_ev(self.model, line - 1)["期待差枚"], 0)

    def test_hopeless_model_has_no_breakeven(self):
        model = ev.CeilingModel(
            ceiling_games=1000,
            hit_probability=0.01,
            average_payout=1,
            ceiling_payout=1,  # 1G 消化する投資(2枚)すら回収できない
            net_loss_per_game=2.0,
        )
        self.assertIsNone(ev.breakeven_start_games(model))


class TestLedger(unittest.TestCase):
    def setUp(self):
        self.rows = ledger.load(LEDGER_CSV)

    def test_balance_is_derived_from_cash_flow(self):
        self.assertEqual(len(self.rows), 7)
        first = self.rows[0]
        self.assertAlmostEqual(first["収支"], first["回収額"] - first["投資額"])
        self.assertAlmostEqual(first["収支"], 21000.0)

    def test_rows_are_sorted_by_date(self):
        dates = [row["日付"] for row in self.rows]
        self.assertEqual(dates, sorted(dates))

    def test_summary_totals(self):
        summary = ledger.summarize(self.rows)
        self.assertEqual(summary["件数"], 7.0)
        expected_total = sum(row["収支"] for row in self.rows)
        self.assertAlmostEqual(summary["合計収支"], expected_total)
        self.assertAlmostEqual(summary["時給"], expected_total / summary["合計稼働時間"])

    def test_group_by_machine_is_sorted(self):
        grouped = ledger.group_by(self.rows, "機種")
        self.assertEqual(len(grouped), 2)
        self.assertGreaterEqual(grouped[0]["合計収支"], grouped[1]["合計収支"])

    def test_monthly_and_cumulative(self):
        months = ledger.monthly(self.rows)
        self.assertEqual(months[0]["年月"], "2026-09")
        running = ledger.cumulative(self.rows)
        self.assertAlmostEqual(running[-1]["累計収支"], ledger.summarize(self.rows)["合計収支"])

    def test_parse_date_formats(self):
        self.assertEqual(str(ledger.parse_date("2026/09/01")), "2026-09-01")
        with self.assertRaises(ValueError):
            ledger.parse_date("9月1日")

    def test_bootstrap_ci_brackets_mean(self):
        values = [float(value) for value in range(-10, 11)]
        lower, upper = ledger.bootstrap_ci(values, trials=2000, seed=3)
        self.assertLess(lower, 0.0)
        self.assertGreater(upper, 0.0)

    def test_significance_verdicts(self):
        winning = [5000.0] * 30
        losing = [-5000.0] * 30
        noisy = [30000.0, -30000.0] * 15
        self.assertTrue(ledger.is_profitable(winning, trials=500, seed=1))
        self.assertFalse(ledger.is_profitable(losing, trials=500, seed=1))
        self.assertIsNone(ledger.is_profitable(noisy, trials=500, seed=1))
        self.assertIsNone(ledger.is_profitable([100.0], trials=500, seed=1))

    def test_required_sessions_grows_with_noise(self):
        steady = [3000.0, 3200.0, 2800.0, 3100.0]
        wild = [3000.0, -20000.0, 25000.0, 1000.0]
        self.assertLess(ledger.required_sessions(steady), ledger.required_sessions(wild))
        self.assertIsNone(ledger.required_sessions([100.0, -100.0]))


if __name__ == "__main__":
    unittest.main()


class TestLedgerRobustness(unittest.TestCase):
    """表計算アプリから来る欠損値（NaN）混じりの行を想定したケース。"""

    def test_nan_like_values_become_blank(self):
        rows = ledger.normalize(
            [
                {
                    "日付": "2026-09-01",
                    "店舗": float("nan"),
                    "機種": "none",
                    "投資額": "10,000円",
                    "回収額": "15,000",
                    "稼働時間": "",
                    "収支": "",
                }
            ]
        )
        self.assertEqual(rows[0]["店舗"], "")
        self.assertEqual(rows[0]["機種"], "")
        self.assertAlmostEqual(rows[0]["投資額"], 10000.0)
        self.assertAlmostEqual(rows[0]["収支"], 5000.0)

    def test_rows_without_date_are_skipped(self):
        rows = ledger.normalize(
            [
                {"日付": "2026-09-01", "収支": "1000"},
                {"日付": float("nan"), "収支": "9999"},
                {"日付": "", "収支": "8888"},
            ]
        )
        self.assertEqual(len(rows), 1)

    def test_missing_date_column_raises(self):
        with self.assertRaises(KeyError):
            ledger.normalize([{"店舗": "Aホール", "収支": "1000"}])

    def test_explicit_balance_column_wins(self):
        rows = ledger.normalize(
            [{"日付": "2026-09-01", "投資額": "10000", "回収額": "12000", "収支": "-500"}]
        )
        self.assertAlmostEqual(rows[0]["収支"], -500.0)

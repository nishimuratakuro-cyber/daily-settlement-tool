"""ホールデータ解析（slot_core.hall）の単体テスト。

fixture は生成データで、「Aホールの 7 のつく日に設定が入る」癖を意図的に
埋め込んである。解析がその癖を拾えることまで確認する。
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slot_core import hall  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HALL_CSV = os.path.join(DATA_DIR, "hall_sample.csv")


class TestColumnResolution(unittest.TestCase):
    def test_resolves_japanese_aliases(self):
        columns = hall.resolve_columns(
            ["営業日", "店舗名", "台番号", "機種名", "総回転数", "差枚", "BB回数", "RB回数"]
        )
        self.assertEqual(columns["日付"], "営業日")
        self.assertEqual(columns["店舗"], "店舗名")
        self.assertEqual(columns["台番"], "台番号")
        self.assertEqual(columns["総回転数"], "総回転数")

    def test_resolves_english_and_spaced_aliases(self):
        columns = hall.resolve_columns(["date", " Store ", "台 No", "G数", "差枚数"])
        self.assertEqual(columns["日付"], "date")
        self.assertEqual(columns["店舗"], " Store ")
        self.assertEqual(columns["台番"], "台 No")
        self.assertEqual(columns["総回転数"], "G数")
        self.assertEqual(columns["差枚"], "差枚数")

    def test_missing_required_column_raises(self):
        with self.assertRaises(ValueError) as caught:
            hall.resolve_columns(["日付", "店舗", "機種"])
        self.assertIn("総回転数", str(caught.exception))


class TestNumberParsing(unittest.TestCase):
    def test_thousand_separators_and_signs(self):
        self.assertAlmostEqual(hall.to_number("1,200"), 1200.0)
        self.assertAlmostEqual(hall.to_number("+1200"), 1200.0)
        self.assertAlmostEqual(hall.to_number("3,000枚"), 3000.0)
        self.assertAlmostEqual(hall.to_number(-500), -500.0)

    def test_accounting_style_negatives(self):
        self.assertAlmostEqual(hall.to_number("▲1,200"), -1200.0)
        self.assertAlmostEqual(hall.to_number("△800"), -800.0)
        self.assertAlmostEqual(hall.to_number("(1200)"), -1200.0)

    def test_blank_becomes_nan(self):
        self.assertTrue(math.isnan(hall.to_number("")))
        self.assertTrue(math.isnan(hall.to_number("-")))
        self.assertTrue(math.isnan(hall.to_number(float("nan"))))

    def test_payout_rate(self):
        self.assertAlmostEqual(hall.payout_rate(3000, 5000), 1.20)
        self.assertAlmostEqual(hall.payout_rate(-1500, 5000), 0.90)
        self.assertTrue(math.isnan(hall.payout_rate(100, 0)))


class TestParseRows(unittest.TestCase):
    def setUp(self):
        self.records = hall.load(HALL_CSV)

    def test_loads_fixture(self):
        self.assertEqual(len(self.records), 480)
        self.assertEqual({record["店舗"] for record in self.records}, {"Aホール", "Bホール"})

    def test_sorted_by_date(self):
        dates = [record["日付"] for record in self.records]
        self.assertEqual(dates, sorted(dates))

    def test_derived_fields(self):
        record = self.records[0]
        self.assertIn(record["曜日"], hall.WEEKDAY_NAMES)
        self.assertIsInstance(record["台番"], int)
        self.assertAlmostEqual(
            record["機械割"], hall.payout_rate(record["差枚"], record["総回転数"])
        )

    def test_rows_with_missing_numbers_are_skipped(self):
        rows = [
            {"日付": "2026-09-01", "総回転数": "5000", "差枚": "1200"},
            {"日付": "2026-09-02", "総回転数": "", "差枚": "900"},
            {"日付": "2026-09-03", "総回転数": "4000", "差枚": "-"},
        ]
        self.assertEqual(len(hall.parse_rows(rows)), 1)

    def test_missing_machine_number_is_none(self):
        rows = [{"日付": "2026-09-01", "台番": "", "総回転数": "5000", "差枚": "0"}]
        self.assertIsNone(hall.parse_rows(rows)[0]["台番"])

    def test_month_day_format_uses_default_year(self):
        rows = [{"日付": "9/7", "総回転数": "5000", "差枚": "0"}]
        records = hall.parse_rows(rows, default_year=2026)
        self.assertEqual(records[0]["日付"], date(2026, 9, 7))


class TestAggregation(unittest.TestCase):
    def test_weighted_rate_is_not_dominated_by_short_samples(self):
        records = hall.parse_rows(
            [
                {"日付": "2026-09-01", "総回転数": "6000", "差枚": "0"},
                {"日付": "2026-09-01", "総回転数": "100", "差枚": "600"},
            ]
        )
        naive = sum(record["機械割"] for record in records) / len(records)
        weighted = hall.weighted_payout_rate(records)
        self.assertGreater(naive, 1.5)  # 100G で +600枚 の台が平均を壊す
        self.assertLess(weighted, 1.05)

    def test_win_ratio(self):
        records = hall.parse_rows(
            [
                {"日付": "2026-09-01", "総回転数": "5000", "差枚": "500"},
                {"日付": "2026-09-01", "総回転数": "5000", "差枚": "-500"},
                {"日付": "2026-09-01", "総回転数": "5000", "差枚": "0"},
            ]
        )
        self.assertAlmostEqual(hall.win_ratio(records), 1 / 3)

    def test_high_setting_ratio_filters_short_samples(self):
        records = hall.parse_rows(
            [
                {"日付": "2026-09-01", "総回転数": "1000", "差枚": "900"},   # 130%だが短い
                {"日付": "2026-09-01", "総回転数": "5000", "差枚": "-900"},
            ]
        )
        self.assertAlmostEqual(hall.high_setting_ratio(records, min_games=2000), 0.0)
        self.assertAlmostEqual(hall.high_setting_ratio(records, min_games=500), 0.5)

    def test_high_setting_ratio_without_eligible_rows(self):
        records = hall.parse_rows([{"日付": "2026-09-01", "総回転数": "500", "差枚": "10"}])
        self.assertTrue(math.isnan(hall.high_setting_ratio(records, min_games=2000)))

    def test_summary_keys(self):
        summary = hall.summarize(hall.load(HALL_CSV))
        self.assertEqual(summary["台数"], 480.0)
        self.assertEqual(summary["日数"], 20.0)
        self.assertEqual(summary["店舗数"], 2.0)


class TestGroupStats(unittest.TestCase):
    def setUp(self):
        self.records = hall.load(HALL_CSV)

    def test_sorted_by_payout_rate(self):
        rows = hall.group_stats(self.records, "機種")
        rates = [row["機械割"] for row in rows]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_difference_is_against_overall(self):
        baseline = hall.weighted_payout_rate(self.records)
        for row in hall.group_stats(self.records, "店舗"):
            self.assertAlmostEqual(row["全体との差"], row["機械割"] - baseline)

    def test_min_count_filter(self):
        rows = hall.group_stats(self.records, "日付", min_count=1000)
        self.assertEqual(rows, [])

    def test_confidence_interval_brackets_point_estimate(self):
        rows = hall.group_stats(self.records, "店舗", ci_trials=200, seed=1)
        for row in rows:
            self.assertLessEqual(row["機械割下限"], row["機械割"])
            self.assertGreaterEqual(row["機械割上限"], row["機械割"])
            self.assertIn(row["全体より上"], {True, False, None})


class TestPlantedPatterns(unittest.TestCase):
    """fixture に埋め込んだ癖を解析が拾えるか。"""

    def setUp(self):
        self.records = hall.load(HALL_CSV)

    def test_store_with_settings_ranks_higher(self):
        rows = {row["区分"]: row for row in hall.group_stats(self.records, "店舗")}
        self.assertGreater(rows["Aホール"]["機械割"], rows["Bホール"]["機械割"])

    def test_seven_day_is_the_strongest_pattern(self):
        rows = {row["区分"]: row for row in hall.special_day_stats(self.records)}
        self.assertGreater(rows["7のつく日"]["機械割"], rows["該当なし"]["機械割"])
        self.assertGreater(rows["7のつく日"]["全体との差"], 0.02)

    def test_top_store_day_is_a_seven_day_at_the_generous_store(self):
        top = hall.store_day_ranking(self.records, top=1)[0]
        self.assertEqual(top["店舗"], "Aホール")
        self.assertEqual(top["日付"].day % 10, 7)

    def test_machine_number_digit_label(self):
        self.assertEqual(hall.machine_number_digit({"台番": 107}), "末尾7")
        self.assertEqual(hall.machine_number_digit({"台番": None}), "（台番なし）")


if __name__ == "__main__":
    unittest.main()

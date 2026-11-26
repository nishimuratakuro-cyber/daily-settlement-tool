"""実データの機種テーブル（マイジャグラーV / Ⅵ）のテスト。

同梱のダミーではなく、公開されている解析値で組んだテーブルを使う。
ボーナス確率と機械割は複数の解析サイトで一致している値。ぶどうは出典ごとに
小数第2位がずれるため、V は一貫した1組を採用し、Ⅵ（未公表・逆算値のみ）は
行を持たせていない。

ここでは「読めること」だけでなく、**このテーブルでどこまで判別できるのか**を
数値で固定している。立ち回りの判断に直接使う情報なので、解析値を差し替えたら
この数値が動くことを検知したい。
"""

from __future__ import annotations

import csv
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slot_core import bayes  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
V_CSV = os.path.join(DATA_DIR, "machines_myjuggler5.csv")
VI_CSV = os.path.join(DATA_DIR, "machines_myjuggler6.csv")

SETTINGS = ("設定1", "設定2", "設定3", "設定4", "設定5", "設定6")
PAYOUT_RATES = (0.970, 0.980, 0.999, 1.028, 1.053, 1.094)
# 設定6相当の出方（BIG・REG ともに 1/229.1、ぶどう 1/5.66）
SETTING6 = (229.1, 229.1, 5.66)


def counts_for(games, big, reg, budou=None):
    result = {"BIG": round(games / big), "REG": round(games / reg)}
    if budou is not None:
        result["ぶどう"] = round(games / budou)
    return result


class TestMyJugglerVTable(unittest.TestCase):
    def setUp(self):
        self.spec = bayes.load_machines(V_CSV)["マイジャグラーV"]

    def test_table_shape(self):
        self.assertEqual(self.spec.settings, SETTINGS)
        self.assertEqual(self.spec.event_names(), ["BIG", "REG", "ぶどう"])

    def test_payout_rates_are_loaded(self):
        self.assertIsNotNone(self.spec.payout_rates)
        for actual, expected in zip(self.spec.payout_rates, PAYOUT_RATES):
            self.assertAlmostEqual(actual, expected, places=4)

    def test_setting6_has_equal_big_and_reg(self):
        # マイジャグラーの設定6は BIG と REG が同確率になるのが特徴
        self.assertAlmostEqual(
            self.spec.probabilities("BIG")[-1], self.spec.probabilities("REG")[-1], places=6
        )

    def test_probabilities_increase_with_setting(self):
        for event in ("BIG", "REG", "ぶどう"):
            values = list(self.spec.probabilities(event))
            self.assertEqual(values, sorted(values), f"{event} が設定順に単調増加していない")

    def test_payout_rates_increase_with_setting(self):
        values = list(self.spec.payout_rates)
        self.assertEqual(values, sorted(values))

    def test_reg_has_the_largest_setting_spread(self):
        """REG の設定差が BIG・ぶどうより大きいこと（判別の主役が REG である根拠）。"""
        def spread(event):
            values = self.spec.probabilities(event)
            return values[-1] / values[0]

        self.assertGreater(spread("REG"), spread("BIG"))
        self.assertGreater(spread("REG"), spread("ぶどう"))
        self.assertAlmostEqual(spread("REG"), 409.6 / 229.1, places=2)

    def test_low_setting_sample_points_to_low_setting(self):
        result = bayes.posterior(self.spec, counts_for(6000, 273.1, 409.6, 5.90), 6000)
        self.assertEqual(result.best()[0], "設定1")

    def test_high_setting_sample_favours_setting6(self):
        result = bayes.posterior(self.spec, counts_for(6000, *SETTING6), 6000)
        self.assertEqual(result.best()[0], "設定6")

    def test_setting6_needs_about_8000_games_to_pass_half(self):
        """6 設定すべてを候補にすると、設定6 の確定は 6000G では届かず 8000G で超える。

        設定2・4 を持たない 4 設定テーブルだと同じ 6000G データで 57% まで出てしまう。
        候補を削ると判別が実際より簡単に見える、という落とし穴を固定しておく。
        """
        at_6000 = bayes.posterior(self.spec, counts_for(6000, *SETTING6), 6000).posterior[-1]
        at_8000 = bayes.posterior(self.spec, counts_for(8000, *SETTING6), 8000).posterior[-1]
        self.assertLess(at_6000, 0.5)
        self.assertGreater(at_8000, 0.5)

        four_settings = bayes.parse_machines(list(csv.DictReader(io.StringIO(
            "機種,事象,種別,設定1,設定3,設定5,設定6\n"
            "V4,BIG,確率,1/273.1,1/266.4,1/240.1,1/229.1\n"
            "V4,REG,確率,1/409.6,1/336.1,1/268.6,1/229.1\n"
            "V4,ぶどう,確率,1/5.90,1/5.84,1/5.74,1/5.66\n"
        ))))["V4"]
        narrowed = bayes.posterior(four_settings, counts_for(6000, *SETTING6), 6000).posterior[-1]
        self.assertGreater(narrowed, at_6000 + 0.05)

    def test_high_setting_group_settles_much_faster_than_setting6(self):
        """『設定4以上か』は 6000G で 95% 決まる。設定6 の特定を待つ必要はない。"""
        result = bayes.posterior(self.spec, counts_for(6000, *SETTING6), 6000)
        self.assertGreater(bayes.high_setting_probability(result), 0.9)
        self.assertLess(result.posterior[-1], 0.5)

    def test_budou_source_discrepancy_barely_matters(self):
        """ぶどうの出典差（1/5.66 と 1/5.69）が結論をほとんど動かさないこと。

        出典ごとに小数第2位がずれるが、設定6 の事後確率への影響は 1pt 未満。
        どの解析サイトの値を使うかで悩む必要はない、という確認。
        """
        def table(budou6):
            text = (
                "機種,事象,種別,設定1,設定2,設定3,設定4,設定5,設定6\n"
                "X,BIG,確率,1/273.1,1/270.8,1/266.4,1/254.0,1/240.1,1/229.1\n"
                "X,REG,確率,1/409.6,1/385.5,1/336.1,1/290.0,1/268.6,1/229.1\n"
                f"X,ぶどう,確率,1/5.90,1/5.89,1/5.84,1/5.78,1/5.74,1/{budou6}\n"
            )
            return bayes.parse_machines(list(csv.DictReader(io.StringIO(text))))["X"]

        for games in (3000, 6000, 10000):
            observed = counts_for(games, *SETTING6)
            tight = bayes.posterior(table("5.66"), observed, games).posterior[-1]
            loose = bayes.posterior(table("5.69"), observed, games).posterior[-1]
            self.assertLess(abs(tight - loose), 0.01, f"{games}G での差が 1pt を超えた")

    def test_small_sample_stays_undecided(self):
        result = bayes.posterior(self.spec, counts_for(1000, *SETTING6), 1000)
        self.assertLess(result.best()[1], 0.5)

    def test_prior_can_exclude_settings(self):
        """設定1・6 しか使わないホールを想定して、他を事前確率 0 で外す。"""
        result = bayes.posterior(
            self.spec,
            counts_for(6000, *SETTING6),
            6000,
            prior=[1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        )
        for index in range(1, 5):
            self.assertEqual(result.posterior[index], 0.0)
        self.assertAlmostEqual(result.posterior[0] + result.posterior[-1], 1.0)
        # 候補が 2 つに絞れるので、同じデータでも設定6 を言い切れる
        self.assertGreater(result.posterior[-1], 0.99)

    def test_expected_payout_rate_tracks_the_evidence(self):
        high = bayes.posterior(self.spec, counts_for(6000, *SETTING6), 6000)
        low = bayes.posterior(self.spec, counts_for(6000, 273.1, 409.6, 5.90), 6000)
        self.assertGreater(bayes.expected_payout_rate(high, self.spec), 1.05)
        self.assertLess(bayes.expected_payout_rate(low, self.spec), 1.0)


class TestMyJugglerVITable(unittest.TestCase):
    """マイジャグラーⅥ（2026-10-05 導入）。ぶどうは未公表なので行が無い。"""

    def setUp(self):
        self.spec = bayes.load_machines(VI_CSV)["マイジャグラーⅥ"]

    def test_table_shape(self):
        self.assertEqual(self.spec.settings, SETTINGS)
        self.assertEqual(self.spec.event_names(), ["BIG", "REG"])

    def test_budou_row_is_absent(self):
        """逆算値しか無いぶどうは入れていない（出典ごとに 1/5.79〜1/5.67 と散る）。"""
        self.assertNotIn("ぶどう", self.spec.events)

    def test_bonus_and_payout_match_the_previous_model(self):
        """ボーナス確率と機械割は V と同一。Ⅵ 固有の差はぶどうの未公表のみ。"""
        previous = bayes.load_machines(V_CSV)["マイジャグラーV"]
        for event in ("BIG", "REG"):
            self.assertEqual(self.spec.probabilities(event), previous.probabilities(event))
        self.assertEqual(self.spec.payout_rates, previous.payout_rates)

    def test_works_without_budou_but_needs_more_games(self):
        """ぶどう無しでも判別は進むが、同じ G 数では V のテーブルに劣ること。"""
        previous = bayes.load_machines(V_CSV)["マイジャグラーV"]
        for games in (6000, 10000):
            bonus_only = bayes.posterior(self.spec, counts_for(games, 229.1, 229.1), games)
            with_budou = bayes.posterior(previous, counts_for(games, *SETTING6), games)
            self.assertEqual(bonus_only.best()[0], "設定6")
            self.assertLess(bonus_only.posterior[-1], with_budou.posterior[-1])
            self.assertGreater(bonus_only.posterior[-1], 0.25)

    def test_unknown_event_is_rejected(self):
        """テーブルに無い事象を渡したら黙って無視せずエラーにすること。"""
        with self.assertRaises(KeyError):
            bayes.posterior(self.spec, {"BIG": 26, "ぶどう": 1000}, 6000)


if __name__ == "__main__":
    unittest.main()

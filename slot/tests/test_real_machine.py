"""実データの機種テーブル（マイジャグラーV）が正しく機能するかのテスト。

同梱のダミーではなく、公開されている解析値で組んだテーブルを使う。
設定2・4 は確認が取れなかったため列を持たせていない。列数は可変なので
後から追加できる。
"""

from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slot_core import bayes  # noqa: E402

CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "machines_myjuggler5.csv",
)


class TestMyJuggler5Table(unittest.TestCase):
    def setUp(self):
        self.spec = bayes.load_machines(CSV)["マイジャグラーV"]

    def counts_for(self, games, big, reg, budou):
        return {
            "BIG": round(games / big),
            "REG": round(games / reg),
            "ぶどう": round(games / budou),
        }

    def test_table_shape(self):
        self.assertEqual(self.spec.settings, ("設定1", "設定3", "設定5", "設定6"))
        self.assertEqual(self.spec.event_names(), ["BIG", "REG", "ぶどう"])

    def test_setting6_has_equal_big_and_reg(self):
        # マイジャグラーの設定6は BIG と REG が同確率になるのが特徴
        big = self.spec.probabilities("BIG")[-1]
        reg = self.spec.probabilities("REG")[-1]
        self.assertAlmostEqual(big, reg, places=6)

    def test_probabilities_increase_with_setting(self):
        for event in ("BIG", "REG", "ぶどう"):
            values = list(self.spec.probabilities(event))
            self.assertEqual(values, sorted(values), f"{event} が設定順に単調増加していない")

    def test_low_setting_sample_points_to_low_setting(self):
        result = bayes.posterior(self.spec, self.counts_for(6000, 273.1, 409.6, 5.90), 6000)
        self.assertEqual(result.best()[0], "設定1")

    def test_high_setting_sample_points_to_setting6(self):
        result = bayes.posterior(self.spec, self.counts_for(6000, 229.1, 229.1, 5.66), 6000)
        setting, probability = result.best()
        self.assertEqual(setting, "設定6")
        self.assertGreater(probability, 0.5)

    def test_reg_carries_most_of_the_signal(self):
        # BIG は設定1相当のまま REG だけ強い場合、高設定側へ寄る
        counts = {"BIG": round(6000 / 273.1), "REG": round(6000 / 240.0), "ぶどう": round(6000 / 5.83)}
        result = bayes.posterior(self.spec, counts, 6000)
        self.assertIn(result.best()[0], {"設定5", "設定6"})

    def test_small_sample_stays_undecided(self):
        # 1000G 程度では設定6相当の出方でも判別がつかない
        result = bayes.posterior(self.spec, self.counts_for(1000, 229.1, 229.1, 5.66), 1000)
        self.assertLess(result.best()[1], 0.5)

    def test_prior_can_exclude_settings(self):
        # 設定1と6しか使わないホールを想定
        result = bayes.posterior(
            self.spec,
            self.counts_for(6000, 229.1, 229.1, 5.66),
            6000,
            prior=[1.0, 0.0, 0.0, 1.0],
        )
        self.assertEqual(result.posterior[1], 0.0)
        self.assertEqual(result.posterior[2], 0.0)
        self.assertAlmostEqual(result.posterior[0] + result.posterior[3], 1.0)

    def test_no_payout_rate_row_is_handled(self):
        # 機械割の行は確認が取れなかったため入れていない
        self.assertIsNone(self.spec.payout_rates)
        result = bayes.posterior(self.spec, self.counts_for(6000, 229.1, 229.1, 5.66), 6000)
        self.assertNotEqual(bayes.expected_payout_rate(result, self.spec), 0.0)

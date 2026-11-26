"""統計ヘルパ（slot_core.stats）の単体テスト。"""

from __future__ import annotations

import math
import os
import statistics
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from slot_core import stats  # noqa: E402


class TestBootstrapCI(unittest.TestCase):
    def test_mean_interval_brackets_sample_mean(self):
        values = [float(value) for value in range(1, 51)]
        lower, upper = stats.bootstrap_ci(values, trials=2000, seed=11)
        mean = sum(values) / len(values)
        self.assertLess(lower, mean)
        self.assertGreater(upper, mean)

    def test_custom_statistic_is_used(self):
        # 外れ値が 1 つ混じった標本。平均は引っ張られるが中央値は動かない。
        values = [1.0] * 20 + [1000.0]
        mean_ci = stats.bootstrap_ci(values, trials=1000, seed=5)
        median_ci = stats.bootstrap_ci(values, trials=1000, seed=5, statistic=statistics.median)
        self.assertAlmostEqual(median_ci[0], 1.0)
        self.assertAlmostEqual(median_ci[1], 1.0)
        self.assertGreater(mean_ci[1] - mean_ci[0], median_ci[1] - median_ci[0])

    def test_too_few_samples_return_nan(self):
        lower, upper = stats.bootstrap_ci([1.0])
        self.assertTrue(math.isnan(lower) and math.isnan(upper))

    def test_invalid_alpha_rejected(self):
        with self.assertRaises(ValueError):
            stats.bootstrap_ci([1.0, 2.0], alpha=0.0)

    def test_seed_makes_it_reproducible(self):
        values = [float(value) for value in range(20)]
        self.assertEqual(
            stats.bootstrap_ci(values, trials=500, seed=3),
            stats.bootstrap_ci(values, trials=500, seed=3),
        )


class TestBootstrapIndexCI(unittest.TestCase):
    def test_ratio_statistic(self):
        numerators = [10.0, 20.0, 30.0, 40.0]
        denominators = [100.0, 200.0, 300.0, 400.0]

        def ratio(indices):
            return sum(numerators[i] for i in indices) / sum(denominators[i] for i in indices)

        lower, upper = stats.bootstrap_index_ci(4, ratio, trials=500, seed=2)
        self.assertLessEqual(lower, 0.1)
        self.assertGreaterEqual(upper, 0.1)  # どう選んでも比は 0.1

    def test_too_few_samples_return_nan(self):
        lower, upper = stats.bootstrap_index_ci(1, lambda indices: 1.0)
        self.assertTrue(math.isnan(lower) and math.isnan(upper))


class TestVerdict(unittest.TestCase):
    def test_above_below_and_straddling(self):
        self.assertIs(stats.verdict((1.0, 2.0)), True)
        self.assertIs(stats.verdict((-2.0, -1.0)), False)
        self.assertIsNone(stats.verdict((-1.0, 1.0)))
        self.assertIsNone(stats.verdict((math.nan, math.nan)))

    def test_custom_baseline(self):
        self.assertIs(stats.verdict((1.01, 1.03), baseline=1.0), True)
        self.assertIsNone(stats.verdict((0.99, 1.03), baseline=1.0))


class TestRequiredSamples(unittest.TestCase):
    def test_noise_increases_requirement(self):
        steady = [3000.0, 3200.0, 2800.0, 3100.0]
        wild = [3000.0, -20000.0, 25000.0, 1000.0]
        self.assertLess(stats.required_samples(steady), stats.required_samples(wild))

    def test_zero_mean_is_undecidable(self):
        self.assertIsNone(stats.required_samples([100.0, -100.0]))
        self.assertIsNone(stats.required_samples([1.0]))

    def test_zero_variance_needs_no_more_samples(self):
        self.assertEqual(stats.required_samples([5.0, 5.0, 5.0]), 3)

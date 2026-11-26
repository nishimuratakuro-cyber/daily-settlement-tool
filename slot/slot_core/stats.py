"""統計の共通部品。

標本が少ないうちに「勝っている / 傾向がある」と言い切らないための道具。
標準ライブラリだけで動く。
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence

# 95% 信頼区間で使う正規分布の分位点
Z_95 = 1.959963984540054


def bootstrap_ci(
    values: Sequence[float],
    trials: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = None,
    statistic: Callable[[Sequence[float]], float] | None = None,
) -> tuple[float, float]:
    """統計量のブートストラップ信頼区間（パーセンタイル法）。

    ``statistic`` を渡さなければ平均。
    """
    if len(values) < 2:
        return (math.nan, math.nan)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha は 0〜1 の間にしてください")

    compute = statistic or (lambda sample: sum(sample) / len(sample))
    rng = random.Random(seed)
    size = len(values)
    estimates = []
    for _ in range(trials):
        sample = [values[rng.randrange(size)] for _ in range(size)]
        estimates.append(compute(sample))
    estimates.sort()

    lower_index = int(trials * (alpha / 2))
    upper_index = min(trials - 1, int(trials * (1 - alpha / 2)))
    return (estimates[lower_index], estimates[upper_index])


def bootstrap_index_ci(
    size: int,
    statistic: Callable[[Sequence[int]], float],
    trials: int = 2_000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """添字を再標本化するブートストラップ。

    「差枚の合計 ÷ 回転数の合計」のように、複数列をまとめて扱う統計量に使う。
    """
    if size < 2:
        return (math.nan, math.nan)
    rng = random.Random(seed)
    estimates = []
    for _ in range(trials):
        indices = [rng.randrange(size) for _ in range(size)]
        estimates.append(statistic(indices))
    estimates.sort()

    lower_index = int(trials * (alpha / 2))
    upper_index = min(trials - 1, int(trials * (1 - alpha / 2)))
    return (estimates[lower_index], estimates[upper_index])


def verdict(interval: tuple[float, float], baseline: float = 0.0) -> bool | None:
    """信頼区間が基準値をまたいでいないか。

    ``True`` なら基準より上、``False`` なら下、``None`` なら判断できない。
    """
    lower, upper = interval
    if math.isnan(lower) or math.isnan(upper):
        return None
    if lower > baseline:
        return True
    if upper < baseline:
        return False
    return None


def required_samples(values: Sequence[float], alpha: float = 0.05) -> int | None:
    """いまの平均・ばらつきが続いた場合、有意になるのに必要な標本数の目安。

    ``n = (z * 標準偏差 / 平均)^2``。平均が 0 に近いと発散するので ``None``。
    """
    if len(values) < 2:
        return None
    size = len(values)
    mean = sum(values) / size
    if math.isclose(mean, 0.0):
        return None
    variance = sum((value - mean) ** 2 for value in values) / (size - 1)
    deviation = math.sqrt(variance)
    if deviation == 0.0:
        return size
    needed = (Z_95 * deviation / abs(mean)) ** 2
    if not math.isfinite(needed) or needed > 1e7:
        return None
    return max(size, math.ceil(needed))

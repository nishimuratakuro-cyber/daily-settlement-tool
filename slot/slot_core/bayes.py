"""設定判別（ベイズ推定）。

観測した小役・ボーナス回数から、各設定の事後確率を求める。

機種ごとの解析値（設定別の確率）はこのモジュールには持たせない。機種数が多く
改訂もされるため、ユーザーが CSV で用意する方式にしている。書式は
``data/machines_sample.csv`` を参照。

標準ライブラリだけで動く。pandas / numpy に依存しない。
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

# CSV の「種別」列で使う値
KIND_PROBABILITY = "確率"
KIND_PAYOUT_RATE = "機械割"

REQUIRED_COLUMNS = ("機種", "事象", "種別")

_EMPTY = {"", "nan", "none", "-", "—", "ー"}


def parse_probability(value: object) -> float:
    """``1/149.3`` / ``0.0067`` / ``149.3`` のいずれの書き方も確率(0〜1)に直す。

    ``149.3`` のように分母だけ書かれた場合は ``1/149.3`` と解釈する。
    空欄は ``nan``（判別に使わない事象）として返す。
    """
    if isinstance(value, bool):
        raise ValueError(f"確率として解釈できません: {value!r}")
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().replace(" ", "").replace("　", "")
        if text.lower() in _EMPTY:
            return math.nan
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            number = (float(numerator) if numerator else 1.0) / float(denominator)
        else:
            number = float(text)

    if math.isnan(number):
        return math.nan
    if number > 1.0:  # 分母だけ書かれたケース
        number = 1.0 / number
    if not 0.0 < number < 1.0:
        raise ValueError(f"確率は 0 < p < 1 の範囲である必要があります: {value!r}")
    return number


def parse_payout_rate(value: object) -> float:
    """``105.5`` / ``1.055`` / ``105.5%`` のいずれも倍率(1.055)に直す。"""
    text = str(value).strip().replace("%", "").replace("％", "")
    if text.lower() in _EMPTY:
        return math.nan
    number = float(text)
    return number / 100.0 if number > 5.0 else number


@dataclass(frozen=True)
class MachineSpec:
    """1 機種分の設定別解析値。"""

    name: str
    settings: tuple[str, ...]
    events: dict[str, tuple[float, ...]]
    payout_rates: tuple[float, ...] | None = None

    def event_names(self) -> list[str]:
        return list(self.events)

    def probabilities(self, event: str) -> tuple[float, ...]:
        if event not in self.events:
            raise KeyError(f"事象 {event!r} は機種 {self.name!r} のテーブルにありません")
        return self.events[event]


@dataclass(frozen=True)
class PosteriorResult:
    """設定判別の計算結果。"""

    settings: tuple[str, ...]
    prior: tuple[float, ...]
    log_likelihood: tuple[float, ...]
    posterior: tuple[float, ...]
    used_events: tuple[str, ...]

    def best(self) -> tuple[str, float]:
        """最も確からしい設定とその事後確率。"""
        index = max(range(len(self.posterior)), key=lambda i: self.posterior[i])
        return self.settings[index], self.posterior[index]

    def as_rows(self) -> list[dict[str, float | str]]:
        return [
            {
                "設定": setting,
                "事前確率": prior,
                "対数尤度": log_likelihood,
                "事後確率": posterior,
            }
            for setting, prior, log_likelihood, posterior in zip(
                self.settings, self.prior, self.log_likelihood, self.posterior
            )
        ]


def load_machines(path: str) -> dict[str, MachineSpec]:
    """機種テーブル CSV を読み込む。機種名 -> MachineSpec。"""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return parse_machines(csv.DictReader(handle))


def parse_machines(rows: Iterable[Mapping[str, str]]) -> dict[str, MachineSpec]:
    """``csv.DictReader`` 相当の行列から機種テーブルを組み立てる。"""
    events: dict[str, dict[str, tuple[float, ...]]] = {}
    rates: dict[str, tuple[float, ...]] = {}
    settings: tuple[str, ...] = ()

    for row in rows:
        missing = [column for column in REQUIRED_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"必要な列がありません: {missing}")
        if not settings:
            settings = tuple(
                column for column in row if column not in REQUIRED_COLUMNS and column
            )
            if len(settings) < 2:
                raise ValueError("設定の列が 2 つ以上必要です（例: 設定1〜設定6）")

        machine = str(row["機種"]).strip()
        event = str(row["事象"]).strip()
        kind = str(row["種別"]).strip()
        if not machine:
            continue

        if kind == KIND_PAYOUT_RATE:
            rates[machine] = tuple(parse_payout_rate(row[column]) for column in settings)
        elif kind == KIND_PROBABILITY:
            values = tuple(parse_probability(row[column]) for column in settings)
            events.setdefault(machine, {})[event] = values
        else:
            raise ValueError(f"種別は {KIND_PROBABILITY!r} か {KIND_PAYOUT_RATE!r} にしてください: {kind!r}")

    if not events:
        raise ValueError("確率行が 1 つもありません")

    return {
        machine: MachineSpec(
            name=machine,
            settings=settings,
            events=table,
            payout_rates=rates.get(machine),
        )
        for machine, table in events.items()
    }


def posterior(
    spec: MachineSpec,
    counts: Mapping[str, float],
    total_games: int,
    prior: Sequence[float] | None = None,
) -> PosteriorResult:
    """設定ごとの事後確率を返す。

    Parameters
    ----------
    spec:
        対象機種の解析値。
    counts:
        事象名 -> 観測回数。テーブルに無い事象を渡すと ``KeyError``。
    total_games:
        観測した総ゲーム数。
    prior:
        設定ごとの事前確率（ホールの設定配分など）。既定は一様。
        0 を入れた設定は「そのホールでは使われない」として完全に除外できる。

    Notes
    -----
    各事象を独立な二項試行として扱う近似。実際の小役は排反なので厳密には多項分布
    だが、実用上の差は小さく、設定判別ツールで一般的な扱い。尤度の二項係数は設定
    間で共通なので省いている（事後確率には影響しない）。
    """
    if total_games <= 0:
        raise ValueError("総ゲーム数は 1 以上にしてください")

    size = len(spec.settings)
    log_likelihood = [0.0] * size
    used: list[str] = []

    for event, observed in counts.items():
        probabilities = spec.probabilities(event)
        if observed < 0:
            raise ValueError(f"観測回数が負です: {event}={observed}")
        if observed > total_games:
            raise ValueError(
                f"観測回数が総ゲーム数を超えています: {event}={observed} > {total_games}"
            )
        if any(math.isnan(p) for p in probabilities):
            continue  # 解析値が埋まっていない事象は判別に使わない
        for index, probability in enumerate(probabilities):
            log_likelihood[index] += observed * math.log(probability)
            log_likelihood[index] += (total_games - observed) * math.log1p(-probability)
        used.append(event)

    if prior is None:
        prior_values = [1.0 / size] * size
    else:
        prior_values = [float(value) for value in prior]
        if len(prior_values) != size:
            raise ValueError("prior の長さが設定数と一致しません")
        if any(value < 0 for value in prior_values):
            raise ValueError("prior に負の値は指定できません")
        total = sum(prior_values)
        if total <= 0:
            raise ValueError("prior の合計が 0 です")
        prior_values = [value / total for value in prior_values]

    log_posterior = [
        log_likelihood[index] + math.log(prior_values[index]) if prior_values[index] > 0 else -math.inf
        for index in range(size)
    ]
    peak = max(log_posterior)
    if math.isinf(peak):
        raise ValueError("すべての設定の事後確率が 0 になりました（prior を確認してください）")
    weights = [math.exp(value - peak) if value != -math.inf else 0.0 for value in log_posterior]
    total_weight = sum(weights)

    return PosteriorResult(
        settings=spec.settings,
        prior=tuple(prior_values),
        log_likelihood=tuple(log_likelihood),
        posterior=tuple(weight / total_weight for weight in weights),
        used_events=tuple(used),
    )


def observed_rates(counts: Mapping[str, float], total_games: int) -> list[dict[str, float | str]]:
    """実測の確率と 1/N を出す（表示用）。"""
    rows: list[dict[str, float | str]] = []
    for event, observed in counts.items():
        rows.append(
            {
                "事象": event,
                "回数": observed,
                "実測確率": observed / total_games if total_games else math.nan,
                "実測1/N": total_games / observed if observed else math.inf,
            }
        )
    return rows


def high_setting_probability(result: PosteriorResult, threshold_index: int = 3) -> float:
    """高設定（既定では並び順で 4 番目以降＝設定4以上）の合計事後確率。"""
    return sum(result.posterior[threshold_index:])


def expected_payout_rate(result: PosteriorResult, spec: MachineSpec) -> float:
    """事後確率で重み付けした期待機械割（倍率）。機械割が未登録なら ``nan``。"""
    if spec.payout_rates is None:
        return math.nan
    if any(math.isnan(rate) for rate in spec.payout_rates):
        return math.nan
    return sum(
        probability * rate for probability, rate in zip(result.posterior, spec.payout_rates)
    )

"""収支記録の集計。

「勝てている」のか「試行回数が足りていないだけ」なのかを、ブートストラップ
信頼区間で切り分ける。スロットは分散が大きいので、数十日程度の収支では
プラスでもマイナスでも運で説明がついてしまうことが多い。

標準ライブラリだけで動く。
"""

from __future__ import annotations

import csv
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime

# 95% 信頼区間で使う正規分布の分位点
Z_95 = 1.959963984540054

DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%y/%m/%d")

TEMPLATE_COLUMNS = (
    "日付",
    "店舗",
    "機種",
    "開始G",
    "終了G",
    "差枚",
    "投資額",
    "回収額",
    "稼働時間",
    "メモ",
)


def _to_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    text = str(value).strip().replace(",", "").replace("円", "").replace("枚", "")
    if text == "" or text.lower() in {"nan", "none", "-"}:
        return default
    return float(text)


def _to_text(value: object) -> str:
    """表計算由来の欠損（None / NaN）を空文字に潰す。"""
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none"} else text


def parse_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for pattern in DATE_FORMATS:
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            continue
    raise ValueError(f"日付として読めません: {value!r}")


def load(path: str) -> list[dict[str, object]]:
    """収支 CSV を読み込んで正規化する。"""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return normalize(csv.DictReader(handle))


def normalize(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    """行を正規化する。``収支`` 列が無ければ ``回収額 - 投資額`` で補う。"""
    result: list[dict[str, object]] = []
    for raw in rows:
        # 列自体が無いのは書式の誤り。値が空なのは単なる未入力行として読み飛ばす。
        if "日付" not in raw:
            raise KeyError("『日付』列が必要です")
        if not any(_to_text(value) for value in raw.values()):
            continue  # 空行
        if not _to_text(raw["日付"]):
            continue  # 日付未入力の行は集計対象外
        row: dict[str, object] = {
            "日付": parse_date(raw["日付"]),
            "店舗": _to_text(raw.get("店舗")),
            "機種": _to_text(raw.get("機種")),
            "差枚": _to_float(raw.get("差枚")),
            "投資額": _to_float(raw.get("投資額")),
            "回収額": _to_float(raw.get("回収額")),
            "稼働時間": _to_float(raw.get("稼働時間")),
            "メモ": _to_text(raw.get("メモ")),
        }
        row["開始G"] = _to_float(raw.get("開始G"))
        row["終了G"] = _to_float(raw.get("終了G"))
        if _to_text(raw.get("収支")):
            row["収支"] = _to_float(raw.get("収支"))
        else:
            row["収支"] = row["回収額"] - row["投資額"]
        result.append(row)
    result.sort(key=lambda item: item["日付"])
    return result


def summarize(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    """全体サマリ。件数・合計収支・時給・勝率など。"""
    if not rows:
        return {"件数": 0.0, "合計収支": 0.0, "平均収支": math.nan, "勝率": math.nan}

    balances = [float(row["収支"]) for row in rows]
    hours = sum(float(row.get("稼働時間", 0.0) or 0.0) for row in rows)
    size = len(balances)
    total = sum(balances)
    mean = total / size
    variance = sum((value - mean) ** 2 for value in balances) / (size - 1) if size > 1 else math.nan

    return {
        "件数": float(size),
        "合計収支": total,
        "平均収支": mean,
        "収支の標準偏差": math.sqrt(variance) if not math.isnan(variance) else math.nan,
        "勝率": sum(1 for value in balances if value > 0) / size,
        "合計稼働時間": hours,
        "時給": total / hours if hours > 0 else math.nan,
        "合計投資額": sum(float(row.get("投資額", 0.0) or 0.0) for row in rows),
        "合計回収額": sum(float(row.get("回収額", 0.0) or 0.0) for row in rows),
        "合計差枚": sum(float(row.get("差枚", 0.0) or 0.0) for row in rows),
    }


def group_by(rows: Sequence[Mapping[str, object]], key: str) -> list[dict[str, float | str]]:
    """機種別・店舗別などの集計。収支の大きい順。"""
    buckets: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        buckets.setdefault(str(row.get(key, "") or "（未入力）"), []).append(row)

    result = []
    for name, group in buckets.items():
        summary = summarize(group)
        result.append(
            {
                key: name,
                "件数": summary["件数"],
                "合計収支": summary["合計収支"],
                "平均収支": summary["平均収支"],
                "勝率": summary["勝率"],
                "時給": summary["時給"],
            }
        )
    result.sort(key=lambda item: item["合計収支"], reverse=True)
    return result


def monthly(rows: Sequence[Mapping[str, object]]) -> list[dict[str, float | str]]:
    """月次集計（日付昇順）。"""
    buckets: dict[str, list[Mapping[str, object]]] = {}
    for row in rows:
        day = row["日付"]
        buckets.setdefault(f"{day.year:04d}-{day.month:02d}", []).append(row)

    result = []
    for month in sorted(buckets):
        summary = summarize(buckets[month])
        result.append(
            {
                "年月": month,
                "件数": summary["件数"],
                "合計収支": summary["合計収支"],
                "勝率": summary["勝率"],
                "時給": summary["時給"],
            }
        )
    return result


def cumulative(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """日付順の累計収支。グラフ用。"""
    running = 0.0
    result = []
    for row in rows:
        running += float(row["収支"])
        result.append({"日付": row["日付"], "収支": float(row["収支"]), "累計収支": running})
    return result


def bootstrap_ci(
    values: Sequence[float],
    trials: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> tuple[float, float]:
    """平均値のブートストラップ信頼区間（パーセンタイル法）。"""
    if len(values) < 2:
        return (math.nan, math.nan)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha は 0〜1 の間にしてください")

    rng = random.Random(seed)
    size = len(values)
    means = []
    for _ in range(trials):
        total = 0.0
        for _ in range(size):
            total += values[rng.randrange(size)]
        means.append(total / size)
    means.sort()

    lower_index = int(trials * (alpha / 2))
    upper_index = min(trials - 1, int(trials * (1 - alpha / 2)))
    return (means[lower_index], means[upper_index])


def is_profitable(
    values: Sequence[float],
    trials: int = 10_000,
    alpha: float = 0.05,
    seed: int | None = None,
) -> bool | None:
    """信頼区間が 0 を跨がないか。

    ``True`` なら「勝ちが運では説明しにくい」、``False`` なら「負けが濃厚」、
    ``None`` なら「まだ判断できる試行回数ではない」。
    """
    lower, upper = bootstrap_ci(values, trials=trials, alpha=alpha, seed=seed)
    if math.isnan(lower) or math.isnan(upper):
        return None
    if lower > 0:
        return True
    if upper < 0:
        return False
    return None


def required_sessions(values: Sequence[float], alpha: float = 0.05) -> int | None:
    """いまの平均・ばらつきが続いた場合、有意になるのに必要な回数の目安。

    ``n = (z * 標準偏差 / 平均)^2``。平均が 0 に近いと発散するので ``None`` を返す。
    """
    if len(values) < 2:
        return None
    size = len(values)
    mean = sum(values) / size
    if math.isclose(mean, 0.0):
        return None
    variance = sum((value - mean) ** 2 for value in values) / (size - 1)
    sd = math.sqrt(variance)
    if sd == 0.0:
        return size
    needed = (Z_95 * sd / abs(mean)) ** 2
    if not math.isfinite(needed) or needed > 1e7:
        return None
    return max(size, math.ceil(needed))

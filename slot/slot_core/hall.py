"""ホールデータ（店舗×日付×台番の実績）から設定投入傾向を読む。

データサイトやホール公式で自分が閲覧できる数字を、手元で CSV に落として
読み込む前提のモジュール。**このモジュールは通信を一切行わない。**

機種選びより店選び・日選びのほうが期待値への寄与が大きいので、
「どの店の、どの日に、どの機種へ設定を使っているか」を数字で出すのが狙い。
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date, datetime

from .stats import bootstrap_index_ci, verdict

MEDALS_PER_GAME = 3.0
WEEKDAY_NAMES = ("月", "火", "水", "木", "金", "土", "日")

DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%y/%m/%d", "%m/%d")

# 列名のゆらぎを吸収する。左が正規化後の名前。
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "日付": ("日付", "営業日", "年月日", "date", "day"),
    "店舗": ("店舗", "店舗名", "店名", "ホール", "ホール名", "store", "hall", "shop"),
    "台番": ("台番", "台番号", "台no", "台ナンバー", "番号", "no", "machineno", "unit", "台"),
    "機種": ("機種", "機種名", "機械名", "machine", "model", "台名"),
    "総回転数": (
        "総回転数", "回転数", "総ゲーム数", "ゲーム数", "総g", "g数", "総スタート",
        "スタート", "総遊技数", "games", "game", "totalgames", "g",
    ),
    "差枚": ("差枚", "差枚数", "差玉", "出玉", "総差枚", "diff", "net", "payout"),
    "BB": ("bb", "bb回数", "ビッグ", "ビッグボーナス", "big", "big回数", "bigbonus"),
    "RB": ("rb", "rb回数", "レギュラー", "レギュラーボーナス", "reg", "reg回数", "regbonus"),
}

REQUIRED_KEYS = ("日付", "総回転数", "差枚")

_TRIM = str.maketrans({character: "" for character in " 　_-()（）/／・.:："})
_EMPTY = {"", "nan", "none", "-", "—", "ー"}


def normalize_key(name: object) -> str:
    """列名を突き合わせ用に潰す（全角空白・記号・大文字小文字を無視）。"""
    return str(name).strip().translate(_TRIM).lower()


def resolve_columns(fieldnames: Iterable[str]) -> dict[str, str]:
    """実際の列名を正規化後の名前へ対応づける。"""
    lookup = {normalize_key(name): name for name in fieldnames if name is not None}
    resolved: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            actual = lookup.get(normalize_key(alias))
            if actual is not None:
                resolved[canonical] = actual
                break
    missing = [key for key in REQUIRED_KEYS if key not in resolved]
    if missing:
        raise ValueError(
            f"必要な列が見つかりません: {missing}（読み取れた列: {sorted(resolved)}）"
        )
    return resolved


def _to_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in _EMPTY else text


def to_number(value: object) -> float:
    """``1,200`` / ``+1200`` / ``▲1,200``（会計式のマイナス）を数値にする。"""
    text = _to_text(value)
    if not text:
        return math.nan
    negative = text[0] in "▲△" or (text.startswith("(") and text.endswith(")"))
    text = text.lstrip("▲△").strip("()").replace(",", "").replace("+", "")
    text = text.replace("枚", "").replace("G", "").replace("回", "")
    if not text or text == "-":
        return math.nan
    number = float(text)
    return -abs(number) if negative else number


def parse_date(value: object, default_year: int | None = None) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = _to_text(value)
    for pattern in DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, pattern)
        except ValueError:
            continue
        if pattern == "%m/%d":  # 年が無い書式は補う
            year = default_year or date.today().year
            return date(year, parsed.month, parsed.day)
        return parsed.date()
    raise ValueError(f"日付として読めません: {value!r}")


def payout_rate(diff_medals: float, games: float, medals_per_game: float = MEDALS_PER_GAME) -> float:
    """差枚と回転数から機械割（倍率）を出す。

    投入枚数 = 3枚 × 回転数 として、``1 + 差枚 / 投入枚数``。
    """
    if games is None or games <= 0 or math.isnan(games):
        return math.nan
    return 1.0 + diff_medals / (medals_per_game * games)


def load(path: str, default_year: int | None = None) -> list[dict[str, object]]:
    """ホールデータ CSV を読み込んで正規化する。"""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return parse_rows(reader, fieldnames=reader.fieldnames or [], default_year=default_year)


def parse_rows(
    rows: Iterable[Mapping[str, object]],
    fieldnames: Iterable[str] | None = None,
    default_year: int | None = None,
    medals_per_game: float = MEDALS_PER_GAME,
) -> list[dict[str, object]]:
    """行の並びを正規化し、機械割・曜日を付与して返す。"""
    rows = list(rows)
    if fieldnames is None:
        if not rows:
            return []
        fieldnames = list(rows[0].keys())
    columns = resolve_columns(fieldnames)

    records: list[dict[str, object]] = []
    for raw in rows:
        if not any(_to_text(value) for value in raw.values()):
            continue
        games = to_number(raw.get(columns["総回転数"]))
        diff = to_number(raw.get(columns["差枚"]))
        if math.isnan(games) or math.isnan(diff):
            continue  # 数字が欠けた行は集計に載せない
        day = parse_date(raw[columns["日付"]], default_year=default_year)

        number_text = _to_text(raw.get(columns.get("台番", ""), ""))
        machine_number = to_number(number_text) if number_text else math.nan

        records.append(
            {
                "日付": day,
                "曜日": WEEKDAY_NAMES[day.weekday()],
                "店舗": _to_text(raw.get(columns.get("店舗", ""), "")) or "（未入力）",
                "台番": None if math.isnan(machine_number) else int(machine_number),
                "機種": _to_text(raw.get(columns.get("機種", ""), "")) or "（未入力）",
                "総回転数": games,
                "差枚": diff,
                "BB": to_number(raw.get(columns.get("BB", ""), "")),
                "RB": to_number(raw.get(columns.get("RB", ""), "")),
                "機械割": payout_rate(diff, games, medals_per_game),
            }
        )
    records.sort(key=lambda record: (record["日付"], record["店舗"], record["台番"] or 0))
    return records


# --- 集計 -------------------------------------------------------------------
def weighted_payout_rate(
    records: Sequence[Mapping[str, object]], medals_per_game: float = MEDALS_PER_GAME
) -> float:
    """台数平均ではなく、総差枚 ÷ 総投入で出す機械割。

    回転数の少ない台の暴れた数字に引っ張られないようにするため、こちらを主指標にする。
    """
    total_games = sum(float(record["総回転数"]) for record in records)
    total_diff = sum(float(record["差枚"]) for record in records)
    return payout_rate(total_diff, total_games, medals_per_game)


def win_ratio(records: Sequence[Mapping[str, object]]) -> float:
    """差枚がプラスだった台の割合。"""
    if not records:
        return math.nan
    return sum(1 for record in records if float(record["差枚"]) > 0) / len(records)


def high_setting_ratio(
    records: Sequence[Mapping[str, object]],
    threshold: float = 1.05,
    min_games: float = 2000.0,
) -> float:
    """高設定が疑われる台の割合。

    回転数が少ない台は機械割が暴れるので ``min_games`` 未満は母数から外す。
    """
    eligible = [record for record in records if float(record["総回転数"]) >= min_games]
    if not eligible:
        return math.nan
    return sum(1 for record in eligible if float(record["機械割"]) >= threshold) / len(eligible)


def summarize(
    records: Sequence[Mapping[str, object]], medals_per_game: float = MEDALS_PER_GAME
) -> dict[str, float]:
    return {
        "台数": float(len(records)),
        "総回転数": sum(float(record["総回転数"]) for record in records),
        "総差枚": sum(float(record["差枚"]) for record in records),
        "機械割": weighted_payout_rate(records, medals_per_game),
        "勝ち台率": win_ratio(records),
        "高設定率": high_setting_ratio(records),
        "日数": float(len({record["日付"] for record in records})),
        "店舗数": float(len({record["店舗"] for record in records})),
    }


def payout_ci(
    records: Sequence[Mapping[str, object]],
    trials: int = 2_000,
    alpha: float = 0.05,
    seed: int | None = None,
    medals_per_game: float = MEDALS_PER_GAME,
) -> tuple[float, float]:
    """機械割のブートストラップ信頼区間（台を再標本化する）。"""
    games = [float(record["総回転数"]) for record in records]
    diffs = [float(record["差枚"]) for record in records]

    def statistic(indices: Sequence[int]) -> float:
        total_games = sum(games[index] for index in indices)
        total_diff = sum(diffs[index] for index in indices)
        return payout_rate(total_diff, total_games, medals_per_game)

    return bootstrap_index_ci(len(records), statistic, trials=trials, alpha=alpha, seed=seed)


def _label(record: Mapping[str, object], key: str | Callable[[Mapping[str, object]], object]) -> object:
    return key(record) if callable(key) else record.get(key)


def group_stats(
    records: Sequence[Mapping[str, object]],
    key: str | Callable[[Mapping[str, object]], object],
    min_count: int = 1,
    ci_trials: int = 0,
    seed: int | None = None,
    medals_per_game: float = MEDALS_PER_GAME,
) -> list[dict[str, object]]:
    """区分ごとの機械割を出し、全体平均との差で並べる。

    ``ci_trials`` を指定すると信頼区間と「全体平均より上と言えるか」の判定も付く。
    """
    baseline = weighted_payout_rate(records, medals_per_game)
    buckets: dict[object, list[Mapping[str, object]]] = {}
    for record in records:
        buckets.setdefault(_label(record, key), []).append(record)

    result: list[dict[str, object]] = []
    for label, group in buckets.items():
        if len(group) < min_count:
            continue
        rate = weighted_payout_rate(group, medals_per_game)
        row: dict[str, object] = {
            "区分": label,
            "台数": len(group),
            "総回転数": sum(float(record["総回転数"]) for record in group),
            "総差枚": sum(float(record["差枚"]) for record in group),
            "機械割": rate,
            "勝ち台率": win_ratio(group),
            "高設定率": high_setting_ratio(group),
            "全体との差": rate - baseline,
        }
        if ci_trials:
            interval = payout_ci(group, trials=ci_trials, seed=seed, medals_per_game=medals_per_game)
            row["機械割下限"], row["機械割上限"] = interval
            row["全体より上"] = verdict(interval, baseline)
        result.append(row)

    result.sort(key=lambda row: (math.isnan(float(row["機械割"])), -float(row["機械割"])))
    return result


# --- 日付の癖 ---------------------------------------------------------------
def special_day_labels(record: Mapping[str, object]) -> list[str]:
    """イベント日として語られがちな日付パターン。"""
    day = record["日付"]
    labels = []
    if day.day % 10 == 7:
        labels.append("7のつく日")
    if day.day in (11, 22):
        labels.append("ゾロ目の日")
    if day.day == day.month:
        labels.append("月と同じ数字の日")
    if day.day <= 5:
        labels.append("月初")
    return labels or ["該当なし"]


def special_day_stats(
    records: Sequence[Mapping[str, object]],
    ci_trials: int = 0,
    seed: int | None = None,
) -> list[dict[str, object]]:
    """1 台が複数パターンに属しうるので、パターンごとに母集団を作り直す。"""
    baseline = weighted_payout_rate(records)
    buckets: dict[str, list[Mapping[str, object]]] = {}
    for record in records:
        for label in special_day_labels(record):
            buckets.setdefault(label, []).append(record)

    result = []
    for label, group in buckets.items():
        rate = weighted_payout_rate(group)
        row: dict[str, object] = {
            "区分": label,
            "台数": len(group),
            "日数": len({record["日付"] for record in group}),
            "機械割": rate,
            "勝ち台率": win_ratio(group),
            "高設定率": high_setting_ratio(group),
            "全体との差": rate - baseline,
        }
        if ci_trials:
            interval = payout_ci(group, trials=ci_trials, seed=seed)
            row["機械割下限"], row["機械割上限"] = interval
            row["全体より上"] = verdict(interval, baseline)
        result.append(row)

    result.sort(key=lambda row: -float(row["機械割"]))
    return result


def machine_number_digit(record: Mapping[str, object]) -> str:
    number = record.get("台番")
    return "（台番なし）" if number is None else f"末尾{int(number) % 10}"


def store_day_ranking(
    records: Sequence[Mapping[str, object]], top: int = 10
) -> list[dict[str, object]]:
    """店舗×日付の機械割ランキング。どの営業日に使ったかを見る。"""
    rows = group_stats(records, lambda record: (record["店舗"], record["日付"], record["曜日"]))
    ranked = []
    for row in rows[:top]:
        store, day, weekday = row["区分"]
        ranked.append(
            {
                "店舗": store,
                "日付": day,
                "曜日": weekday,
                "台数": row["台数"],
                "機械割": row["機械割"],
                "総差枚": row["総差枚"],
                "高設定率": row["高設定率"],
            }
        )
    return ranked

"""スロット立ち回り支援ツール（Streamlit）。

計算はすべて ``slot_core`` 側にあり、ここは入出力だけを担当する。
起動:
    streamlit run slot/app.py
"""

from __future__ import annotations

import csv
import io
import math
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slot_core import bayes, ev, hall, ledger  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MACHINES_CSV = os.path.join(DATA_DIR, "machines_sample.csv")
LEDGER_CSV = os.path.join(DATA_DIR, "ledger_sample.csv")
HALL_CSV = os.path.join(DATA_DIR, "hall_sample.csv")

st.set_page_config(page_title="スロット立ち回り支援ツール", page_icon="🎰", layout="wide")


# --- 共通パーツ -------------------------------------------------------------
def exchange_input(key_prefix: str) -> ev.Exchange:
    """交換率の入力欄。等価 / 非等価を切り替える。"""
    mode = st.radio(
        "交換率",
        ["等価（20円）", "非等価（枚数指定）"],
        horizontal=True,
        key=f"{key_prefix}_mode",
    )
    if mode.startswith("等価"):
        return ev.Exchange()
    medals = st.number_input(
        "100円で交換できる枚数",
        min_value=5.0,
        max_value=10.0,
        value=5.6,
        step=0.1,
        key=f"{key_prefix}_medals",
        help="5.6枚交換なら 5.6。換金単価は 100 ÷ この枚数 で計算されます",
    )
    exchange = ev.Exchange.from_medals_per_100yen(medals)
    st.caption(f"換金単価 {exchange.payout_yen:.2f} 円/枚（貸出 {exchange.rental_yen:.0f} 円/枚）")
    return exchange


def probability_input(label: str, default_denominator: float, key: str) -> float:
    """1/N 形式で確率を入力させる。"""
    denominator = st.number_input(
        label, min_value=1.0, max_value=100_000.0, value=default_denominator, step=1.0, key=key
    )
    return 1.0 / denominator


def histogram(values: list[float], bins: int = 30) -> pd.DataFrame:
    """ヒストグラムを DataFrame にする（matplotlib を使わずに描くため）。"""
    low, high = min(values), max(values)
    if math.isclose(low, high):
        return pd.DataFrame({"件数": [len(values)]}, index=[f"{low:.0f}"])
    width = (high - low) / bins
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - low) / width))
        counts[index] = counts[index] + 1
    labels = [f"{low + width * i:,.0f}" for i in range(bins)]
    return pd.DataFrame({"件数": counts}, index=labels)


# --- 1. 設定判別 -------------------------------------------------------------
def page_setting_estimation() -> None:
    st.header("🎯 設定判別（ベイズ推定）")
    st.caption(
        "小役・ボーナスの観測回数から、各設定の事後確率を計算します。"
        "機種の解析値は CSV で差し替えてください（同梱のサンプルはダミー値です）。"
    )

    uploaded = st.file_uploader("機種テーブル CSV", type="csv", key="machine_csv")
    try:
        if uploaded is not None:
            text = uploaded.getvalue().decode("utf-8-sig")
            machines = bayes.parse_machines(csv.DictReader(io.StringIO(text)))
        else:
            machines = bayes.load_machines(MACHINES_CSV)
            st.info("サンプルの機種テーブルを使用中です（数値はダミー）。", icon="ℹ️")
    except (ValueError, KeyError) as error:
        st.error(f"機種テーブルを読めませんでした: {error}")
        return

    machine_name = st.selectbox("機種", list(machines))
    spec = machines[machine_name]

    st.subheader("観測値の入力")
    total_games = st.number_input("総ゲーム数", min_value=1, max_value=200_000, value=8000, step=100)

    counts: dict[str, float] = {}
    columns = st.columns(min(4, max(1, len(spec.events))))
    for index, event in enumerate(spec.event_names()):
        with columns[index % len(columns)]:
            counts[event] = st.number_input(
                event, min_value=0, max_value=int(total_games), value=0, step=1, key=f"count_{event}"
            )

    with st.expander("ホールの設定配分（事前確率）を指定する"):
        st.caption("相対値でかまいません。0 にすると『その設定は使われない』として除外します。")
        prior_columns = st.columns(len(spec.settings))
        prior = []
        for index, setting in enumerate(spec.settings):
            with prior_columns[index]:
                prior.append(
                    st.number_input(setting, min_value=0.0, value=1.0, step=0.1, key=f"prior_{setting}")
                )
        if sum(prior) <= 0:
            st.warning("事前確率の合計が 0 です。一様分布として扱います。")
            prior = None

    if sum(counts.values()) == 0:
        st.info("観測回数を入力すると判別を開始します。")
        return

    try:
        result = bayes.posterior(spec, counts, int(total_games), prior=prior)
    except (ValueError, KeyError) as error:
        st.error(f"計算できませんでした: {error}")
        return

    best_setting, best_probability = result.best()
    high_probability = bayes.high_setting_probability(result)
    expected_rate = bayes.expected_payout_rate(result, spec)

    st.subheader("判別結果")
    metric_columns = st.columns(3)
    metric_columns[0].metric("最有力", best_setting, delta=f"{best_probability:.1%}")
    metric_columns[1].metric("高設定期待度（設定4以上）", f"{high_probability:.1%}")
    metric_columns[2].metric(
        "期待機械割", "—" if math.isnan(expected_rate) else f"{expected_rate * 100:.1f}%"
    )

    frame = pd.DataFrame(result.as_rows()).set_index("設定")
    st.bar_chart(frame["事後確率"])
    st.dataframe(
        frame.style.format({"事前確率": "{:.1%}", "対数尤度": "{:.1f}", "事後確率": "{:.2%}"}),
        use_container_width=True,
    )

    if not math.isnan(expected_rate):
        if expected_rate >= 1.0:
            st.success(f"期待機械割 {expected_rate * 100:.1f}% — 続行が有利な水準です。")
        else:
            st.warning(f"期待機械割 {expected_rate * 100:.1f}% — 現時点では続行が不利な水準です。")

    st.subheader("実測値と理論値")
    observed = pd.DataFrame(bayes.observed_rates(counts, int(total_games))).set_index("事象")
    theoretical = pd.DataFrame(
        {setting: {event: 1 / spec.events[event][index] for event in spec.events}
         for index, setting in enumerate(spec.settings)}
    )
    st.dataframe(
        observed.join(theoretical).style.format("{:.1f}", subset=list(spec.settings)).format(
            {"実測確率": "{:.4f}", "実測1/N": "{:.1f}", "回数": "{:.0f}"}
        ),
        use_container_width=True,
    )
    st.caption("表の設定列は理論上の 1/N。実測1/N がどの列に近いかが判別の直感的な根拠になります。")


# --- 2. 天井狙い -------------------------------------------------------------
def page_ceiling() -> None:
    st.header("⏱ 天井狙いの期待値")
    st.caption(
        "通常時の当選率を一定とみなす近似モデルです。ゾーンやモードが濃い機種ほど誤差が出ます。"
    )

    left, right = st.columns(2)
    with left:
        ceiling_games = st.number_input("天井ゲーム数", min_value=1, max_value=10_000, value=1000, step=50)
        hit_probability = probability_input("通常時の当選確率 1/N", 300.0, "ceiling_hit")
        current_games = st.number_input(
            "現在のゲーム数", min_value=0, max_value=int(ceiling_games), value=700, step=10
        )
    with right:
        average_payout = st.number_input("自力当選時の平均獲得枚数", min_value=0.0, value=500.0, step=50.0)
        ceiling_payout = st.number_input("天井到達時の平均獲得枚数", min_value=0.0, value=800.0, step=50.0)
        coin_persistence = st.number_input(
            "コイン持ち（50枚あたりのゲーム数）", min_value=1.0, max_value=100.0, value=25.0, step=0.5
        )
    exchange = exchange_input("ceiling")

    model = ev.CeilingModel(
        ceiling_games=int(ceiling_games),
        hit_probability=hit_probability,
        average_payout=average_payout,
        ceiling_payout=ceiling_payout,
        net_loss_per_game=50.0 / coin_persistence,
    )
    result = ev.ceiling_ev(model, int(current_games), exchange=exchange)
    breakeven = ev.breakeven_start_games(model)

    metric_columns = st.columns(4)
    metric_columns[0].metric("期待差枚", f"{result['期待差枚']:+,.0f} 枚")
    metric_columns[1].metric("期待収支", f"{result['期待収支(円)']:+,.0f} 円")
    metric_columns[2].metric("天井到達率", f"{result['天井到達率']:.1%}")
    metric_columns[3].metric(
        "狙い目ライン", "なし" if breakeven is None else f"{breakeven:,}G〜"
    )

    if breakeven is None:
        st.error("この条件では天井直前でも期待値がプラスになりません。")
    elif current_games >= breakeven:
        st.success(f"{breakeven:,}G 以上なら期待値プラス。現在 {int(current_games):,}G は打てるゾーンです。")
    else:
        st.warning(f"期待値がプラスになるのは {breakeven:,}G から。現在 {int(current_games):,}G では見送りです。")

    detail = pd.DataFrame(
        {
            "項目": list(result),
            "値": [result[key] for key in result],
        }
    ).set_index("項目")
    st.dataframe(detail.style.format({"値": "{:,.2f}"}), use_container_width=True)

    step = max(1, int(ceiling_games) // 100)
    curve = pd.DataFrame(
        {
            "現在ゲーム数": list(range(0, int(ceiling_games), step)),
            "期待差枚": [
                ev.ceiling_ev(model, games)["期待差枚"] for games in range(0, int(ceiling_games), step)
            ],
        }
    ).set_index("現在ゲーム数")
    st.subheader("ゲーム数別の期待差枚")
    st.line_chart(curve)


# --- 3. シミュレーション -----------------------------------------------------
def page_simulation() -> None:
    st.header("📐 差枚シミュレーション")
    st.caption("『機械割はプラスなのに負ける』がどれくらい起きるかを、分布として確認します。")

    left, right = st.columns(2)
    with left:
        hit_probability = probability_input("通常時の初当り確率 1/N", 300.0, "sim_hit")
        average_payout = st.number_input("初当り1回あたりの平均獲得枚数", min_value=1.0, value=500.0, step=50.0)
        payout_sd = st.number_input("獲得枚数のばらつき（標準偏差）", min_value=0.0, value=600.0, step=50.0)
    with right:
        coin_persistence = st.number_input(
            "コイン持ち（50枚あたりのゲーム数）", min_value=1.0, max_value=100.0, value=25.0, step=0.5, key="sim_coin"
        )
        normal_games = st.number_input("通常時の消化ゲーム数", min_value=1, max_value=50_000, value=5000, step=500)
        bankroll = st.number_input("持ち込み資金（円）", min_value=0, value=50_000, step=10_000)
    exchange = exchange_input("sim")

    trials = st.slider("試行回数", min_value=1_000, max_value=50_000, value=20_000, step=1_000)
    seed = st.number_input("乱数シード（同じ値なら結果が再現します）", min_value=0, value=42, step=1)

    model = ev.SessionModel(
        hit_probability=hit_probability,
        average_payout=average_payout,
        payout_sd=payout_sd,
        net_loss_per_game=50.0 / coin_persistence,
    )
    diffs = ev.simulate_diffs(model, int(normal_games), trials=int(trials), seed=int(seed))
    stats = ev.session_stats(diffs, exchange=exchange, bankroll_yen=bankroll or None)

    metric_columns = st.columns(4)
    metric_columns[0].metric("期待差枚", f"{stats['期待差枚']:+,.0f} 枚")
    metric_columns[1].metric("期待収支", f"{stats['期待収支(円)']:+,.0f} 円")
    metric_columns[2].metric("勝率", f"{stats['勝率']:.1%}")
    metric_columns[3].metric(
        "資金ショート確率", f"{stats.get('資金ショート確率', float('nan')):.1%}" if bankroll else "—"
    )

    if not exchange.is_even:
        st.info(
            f"交換率による目減り: {stats['交換率による目減り(円)']:+,.0f} 円。"
            "非等価では差枚がプラスマイナスゼロでも現金収支はマイナスになります。",
            icon="ℹ️",
        )
    if bankroll:
        st.caption("資金ショート確率は終了時点で必要な現金が資金を超えた割合です。途中経過はさらに沈むため、実際の破産確率はこれより高くなります。")

    st.subheader("差枚の分布")
    st.bar_chart(histogram(diffs))

    quantiles = pd.DataFrame(
        {
            "分位点": ["下位5%", "下位25%", "中央値", "上位75%", "上位95%"],
            "差枚": [
                stats["下位5%差枚"],
                stats["下位25%差枚"],
                stats["中央値差枚"],
                stats["上位75%差枚"],
                stats["上位95%差枚"],
            ],
        }
    ).set_index("分位点")
    quantiles["収支(円)"] = [exchange.cash(value) for value in quantiles["差枚"]]
    st.dataframe(
        quantiles.style.format({"差枚": "{:+,.0f}", "収支(円)": "{:+,.0f}"}), use_container_width=True
    )


# --- 4. 収支管理 -------------------------------------------------------------
def page_ledger() -> None:
    st.header("💰 収支管理")
    st.caption("記録した収支が『勝ち』と言い切れるのか、試行回数が足りないだけなのかを判定します。")

    uploaded = st.file_uploader("収支 CSV", type="csv", key="ledger_csv")
    if uploaded is not None:
        frame = pd.read_csv(uploaded)
    elif "ledger_frame" in st.session_state:
        frame = st.session_state.ledger_frame
    else:
        frame = pd.read_csv(LEDGER_CSV)
        st.info("サンプルの収支データを表示しています。自分の記録に置き換えてください。", icon="ℹ️")

    edited = st.data_editor(frame, num_rows="dynamic", use_container_width=True, key="ledger_editor")
    st.session_state.ledger_frame = edited

    if "日付" not in edited.columns:
        st.error("『日付』列が必要です。テンプレート（data/ledger_sample.csv）の列構成に合わせてください。")
        return

    records = edited.dropna(subset=["日付"]).to_dict("records")
    if not records:
        st.info("1 行以上入力すると集計します。")
        return

    try:
        rows = ledger.normalize(records)
    except (ValueError, KeyError) as error:
        st.error(f"集計できませんでした: {error}")
        return

    summary = ledger.summarize(rows)
    metric_columns = st.columns(4)
    metric_columns[0].metric("合計収支", f"{summary['合計収支']:+,.0f} 円")
    metric_columns[1].metric("時給", "—" if math.isnan(summary["時給"]) else f"{summary['時給']:+,.0f} 円")
    metric_columns[2].metric("勝率", f"{summary['勝率']:.1%}")
    metric_columns[3].metric("実戦回数", f"{int(summary['件数'])} 回")

    st.subheader("累計収支")
    running = pd.DataFrame(ledger.cumulative(rows)).set_index("日付")
    st.area_chart(running["累計収支"])

    left, right = st.columns(2)
    with left:
        st.subheader("機種別")
        st.dataframe(
            pd.DataFrame(ledger.group_by(rows, "機種")).set_index("機種").style.format(
                {"件数": "{:.0f}", "合計収支": "{:+,.0f}", "平均収支": "{:+,.0f}", "勝率": "{:.0%}", "時給": "{:+,.0f}"}
            ),
            use_container_width=True,
        )
    with right:
        st.subheader("店舗別")
        st.dataframe(
            pd.DataFrame(ledger.group_by(rows, "店舗")).set_index("店舗").style.format(
                {"件数": "{:.0f}", "合計収支": "{:+,.0f}", "平均収支": "{:+,.0f}", "勝率": "{:.0%}", "時給": "{:+,.0f}"}
            ),
            use_container_width=True,
        )

    st.subheader("月次")
    st.dataframe(
        pd.DataFrame(ledger.monthly(rows)).set_index("年月").style.format(
            {"件数": "{:.0f}", "合計収支": "{:+,.0f}", "勝率": "{:.0%}", "時給": "{:+,.0f}"}
        ),
        use_container_width=True,
    )

    st.subheader("その収支、運の範囲内か？")
    balances = [float(row["収支"]) for row in rows]
    lower, upper = ledger.bootstrap_ci(balances, trials=5000, seed=0)
    verdict = ledger.is_profitable(balances, trials=5000, seed=0)
    needed = ledger.required_sessions(balances)

    if math.isnan(lower):
        st.info("判定には 2 回以上の記録が必要です。")
    else:
        st.write(f"1回あたり平均収支の95%信頼区間: **{lower:+,.0f} 円 〜 {upper:+,.0f} 円**")
        if verdict is True:
            st.success("信頼区間がプラス側に収まっています。運だけでは説明しにくい水準です。")
        elif verdict is False:
            st.error("信頼区間がマイナス側に収まっています。立ち回りの見直しが必要です。")
        else:
            st.warning("信頼区間が 0 を跨いでいます。現時点では勝ちとも負けとも言えません。")
        if needed is not None:
            st.caption(f"いまのペースが続く場合、判断がつくまでの目安は約 {needed:,} 回です。")

    st.download_button(
        "💾 収支CSVをダウンロード",
        edited.to_csv(index=False).encode("utf-8-sig"),
        "slot_ledger.csv",
        "text/csv",
    )


# --- 5. ホール傾向分析 -------------------------------------------------------
def group_table(rows: list[dict], label: str = "区分"):
    """集計結果を見やすい表にする。"""
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.rename(columns={"区分": label}).set_index(label)
    formats = {
        "台数": "{:,.0f}",
        "日数": "{:,.0f}",
        "総回転数": "{:,.0f}",
        "総差枚": "{:+,.0f}",
        "機械割": "{:.1%}",
        "勝ち台率": "{:.0%}",
        "高設定率": "{:.0%}",
        "全体との差": "{:+.1%}",
        "機械割下限": "{:.1%}",
        "機械割上限": "{:.1%}",
    }
    return frame.style.format({key: value for key, value in formats.items() if key in frame.columns})


def page_hall() -> None:
    st.header("🏠 ホール傾向分析")
    st.caption(
        "店舗×日付×台番の実績から「どの店の、どの日に、どの機種へ設定を使っているか」を出します。"
        "通信は一切せず、手元の CSV だけを読みます。"
    )

    uploaded = st.file_uploader("ホールデータ CSV", type="csv", key="hall_csv")
    try:
        if uploaded is not None:
            reader = csv.DictReader(io.StringIO(uploaded.getvalue().decode("utf-8-sig")))
            records = hall.parse_rows(reader, fieldnames=reader.fieldnames or [])
        else:
            records = hall.load(HALL_CSV)
            st.info("サンプルを表示中です（生成データであり、実在ホールの記録ではありません）。", icon="ℹ️")
    except (ValueError, KeyError) as error:
        st.error(f"読み込めませんでした: {error}")
        st.caption("必須列は 日付 / 総回転数 / 差枚 の 3 つ。列名のゆらぎ（営業日・G数・差枚数など）は自動で吸収します。")
        return

    if not records:
        st.warning("集計できる行がありませんでした。")
        return

    stores = sorted({record["店舗"] for record in records})
    machine_names = sorted({record["機種"] for record in records})
    left, middle, right = st.columns(3)
    with left:
        picked_stores = st.multiselect("店舗", stores, default=stores)
    with middle:
        picked_machines = st.multiselect("機種", machine_names, default=machine_names)
    with right:
        min_games = st.number_input(
            "最低回転数", min_value=0, max_value=20_000, value=0, step=500,
            help="これ未満しか回っていない台を除外します。短時間の台は機械割が暴れるため",
        )

    records = [
        record
        for record in records
        if record["店舗"] in picked_stores
        and record["機種"] in picked_machines
        and record["総回転数"] >= min_games
    ]
    if not records:
        st.warning("条件に合う台がありません。フィルタを緩めてください。")
        return

    summary = hall.summarize(records)
    metric_columns = st.columns(5)
    metric_columns[0].metric("機械割", f"{summary['機械割'] * 100:.1f}%")
    metric_columns[1].metric("勝ち台率", f"{summary['勝ち台率']:.1%}")
    metric_columns[2].metric(
        "高設定率", "—" if math.isnan(summary["高設定率"]) else f"{summary['高設定率']:.1%}"
    )
    metric_columns[3].metric("台数", f"{int(summary['台数']):,}")
    metric_columns[4].metric("総差枚", f"{summary['総差枚']:+,.0f}")
    st.caption(
        "機械割は台ごとの平均ではなく、総差枚 ÷ 総投入（3枚 × 総回転数）の加重値です。"
        "高設定率は 2000G 以上回った台のうち機械割 105% 以上だった割合。"
    )

    st.subheader("日付別の機械割")
    daily = hall.group_stats(records, "日付")
    daily_frame = (
        pd.DataFrame([{"日付": row["区分"], "機械割": row["機械割"]} for row in daily])
        .sort_values("日付")
        .set_index("日付")
    )
    st.line_chart(daily_frame)

    tabs = st.tabs(["店舗別", "日付パターン", "曜日別", "機種別", "台番末尾", "店舗×日付"])
    with tabs[0]:
        st.dataframe(group_table(hall.group_stats(records, "店舗", ci_trials=400, seed=0), "店舗"), use_container_width=True)
        st.caption("「全体より上」が True の店舗は、差が偶然では説明しにくい水準です。")
    with tabs[1]:
        st.dataframe(group_table(hall.special_day_stats(records, ci_trials=400, seed=0), "パターン"), use_container_width=True)
        st.caption("1 台が複数パターンに属しうるため、パターンごとに母集団を作り直しています。")
    with tabs[2]:
        st.dataframe(group_table(hall.group_stats(records, "曜日"), "曜日"), use_container_width=True)
    with tabs[3]:
        st.dataframe(group_table(hall.group_stats(records, "機種"), "機種"), use_container_width=True)
    with tabs[4]:
        st.dataframe(group_table(hall.group_stats(records, hall.machine_number_digit), "台番末尾"), use_container_width=True)
    with tabs[5]:
        ranking = pd.DataFrame(hall.store_day_ranking(records, top=20))
        st.dataframe(
            ranking.style.format(
                {"機械割": "{:.1%}", "高設定率": "{:.0%}", "総差枚": "{:+,.0f}", "台数": "{:,.0f}"}
            ),
            use_container_width=True,
        )
        st.caption("設定を使った営業日の候補。上位に同じ曜日・日付パターンが並ぶなら、それが狙い目です。")


# --- 6. 手順チェック ---------------------------------------------------------
# 立ち回り情報で見かける手順。消化ゲーム数は仕様からの推定値で、主張の出所ではない。
PROCEDURE_PRESETS: dict[str, tuple[float, int] | None] = {
    "（手入力）": None,
    "北斗転生2 41G確認のみ（主張107%）": (107.0, 41),
    "北斗転生2 リセ〜当選まで（主張107%）": (107.0, 270),
    "東京喰種 朝一〜CZまで（主張103%）": (103.0, 143),
    "東京喰種 CZ外し〜天国（主張104%）": (104.0, 150),
    "北斗の拳 朝一0Gから（主張102%）": (102.0, 250),
    "北斗の拳 100Gから（主張105%）": (105.0, 220),
    "北斗の拳 200Gから（主張110%）": (110.0, 180),
    "モンキーターンV 手順全体（主張105%）": (105.0, 200),
}


def page_procedure() -> None:
    st.header("🧮 手順チェック")
    st.caption(
        "「この手順で機械割○%over」という主張を、1台あたりの実額・実効時給・"
        "交換率別の損益分岐に変換します。率が高くても消化が短ければ実額は小さい、"
        "という点を見えるようにするためのページです。"
    )

    preset_name = st.selectbox("プリセット", list(PROCEDURE_PRESETS))
    preset = PROCEDURE_PRESETS[preset_name]
    default_rate, default_games = preset if preset else (105.0, 200)
    if preset:
        st.caption("消化ゲーム数は機種仕様からの推定値です。実際の平均消化に合わせて調整してください。")

    left, middle, right = st.columns(3)
    with left:
        claimed_rate = st.number_input(
            "主張されている機械割(%)", min_value=50.0, max_value=200.0,
            value=float(default_rate), step=0.1,
        )
        games = st.number_input(
            "1台あたりの消化ゲーム数", min_value=1, max_value=20_000,
            value=int(default_games), step=10,
        )
    with middle:
        loss_per_game = st.number_input(
            "通常時の1Gあたり期待負け枚数", min_value=0.0, max_value=3.0, value=1.72, step=0.01,
            help="通常時の純減 × 非当選率が目安。純増2枚・的中率15%なら約1.72枚",
        )
        games_per_hour = st.number_input(
            "消化速度(G/h)", min_value=100, max_value=1200, value=600, step=50
        )
    with right:
        overhead = st.number_input(
            "1台あたりの付帯時間(分)", min_value=0, max_value=60, value=6, step=1,
            help="台探し・移動・リセット判別にかかる時間。リセ狩りではここが効きます",
        )
    exchange = exchange_input("procedure")

    result = ev.procedure_value(
        claimed_rate,
        int(games),
        exchange=exchange,
        games_per_hour=float(games_per_hour),
        overhead_minutes=float(overhead),
        loss_medals_per_game=loss_per_game,
    )

    metric_columns = st.columns(4)
    metric_columns[0].metric("期待差枚", f"{result['期待差枚']:+,.1f} 枚")
    metric_columns[1].metric("1台あたり期待収支", f"{result['期待収支(円)']:+,.0f} 円")
    metric_columns[2].metric("名目時給", f"{result['名目時給(円)']:+,.0f} 円")
    metric_columns[3].metric(
        "実効時給", f"{result['実効時給(円)']:+,.0f} 円",
        delta=f"付帯{int(overhead)}分込み" if overhead else None,
    )
    st.caption(
        f"拘束時間 {result['拘束時間(分)']:.0f}分（消化 {result['消化時間(分)']:.0f}分 ＋ 付帯 {int(overhead)}分）。"
        "名目時給は機械割だけで決まるので、実額と実効時給のほうが立ち回りの判断材料になります。"
    )

    breakeven = result["現金投資の損益分岐機械割"]
    st.subheader("交換率の影響")
    if exchange.is_even:
        st.success("等価なので損益分岐は機械割100%。主張どおりの率が出ていれば期待値はプラスです。")
    else:
        st.write(
            f"現金投資でこなす場合の損益分岐機械割: **{breakeven * 100:.1f}%**"
            f"（貸出 {exchange.rental_yen:.2f}円 / 換金 {exchange.payout_yen:.2f}円）"
        )
        if result["損益分岐との差"] >= 0:
            st.success(
                f"主張の {claimed_rate:.1f}% は損益分岐を {result['損益分岐との差'] * 100:+.1f}pt 上回っています。"
            )
        else:
            st.error(
                f"主張の {claimed_rate:.1f}% は損益分岐に {result['損益分岐との差'] * 100:.1f}pt 届きません。"
                "現金投資では成立せず、貯メダル・持ちメダルでこなす必要があります。"
            )
        st.caption(
            "メダルは機械内で循環するので、現金で買うのは負けた分だけ。"
            "負け分は貸出単価・勝ち分は換金単価という非対称性が、非等価のペナルティの正体です。"
            "貯メダル／持ちメダル遊技なら単価差が無いので損益分岐は100%に戻ります。"
        )

    st.subheader("機械割と時給の対応")
    rates = [1.00, 1.02, 1.03, 1.04, 1.05, 1.07, 1.10]
    table = pd.DataFrame(
        {
            "機械割": rates,
            "名目時給(円)": [
                ev.hourly_from_payout_rate(rate, float(games_per_hour), exchange=exchange)
                for rate in rates
            ],
            "この消化Gでの実額(円)": [
                3 * int(games) * (rate - 1) * exchange.payout_yen for rate in rates
            ],
        }
    ).set_index("機械割")
    st.dataframe(
        table.style.format({"名目時給(円)": "{:+,.0f}", "この消化Gでの実額(円)": "{:+,.0f}"}),
        use_container_width=True,
    )
    st.caption("等価・600G/h なら 時給 = 36,000 × (機械割 − 1)。小数点以下を36,000倍すれば時給になります。")

    if not exchange.is_even:
        st.subheader("負け枚数別の損益分岐（現金投資）")
        losses = [1.0, 1.5, 1.72, 2.0, 2.5, 3.0]
        st.dataframe(
            pd.DataFrame(
                {
                    "1Gあたり負け枚数": losses,
                    "損益分岐機械割": [
                        ev.breakeven_payout_rate(exchange, loss) for loss in losses
                    ],
                }
            ).set_index("1Gあたり負け枚数").style.format({"損益分岐機械割": "{:.1%}"}),
            use_container_width=True,
        )


PAGES = {
    "🎯 設定判別": page_setting_estimation,
    "⏱ 天井狙い": page_ceiling,
    "📐 シミュレーション": page_simulation,
    "💰 収支管理": page_ledger,
    "🏠 ホール傾向分析": page_hall,
    "🧮 手順チェック": page_procedure,
}


def main() -> None:
    st.sidebar.title("🎰 スロット立ち回り支援")
    choice = st.sidebar.radio("メニュー", list(PAGES))
    st.sidebar.caption(
        "解析値はユーザーが用意する前提のツールです。"
        "同梱データはすべてダミーなので、実戦前に解析サイトの最新値へ差し替えてください。"
    )
    PAGES[choice]()


if __name__ == "__main__":
    main()

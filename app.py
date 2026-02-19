import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import calendar
import io

st.set_page_config(
    page_title="AI推定日次決算ツール",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- サンプルデータ生成 ---
def generate_sample_revenue_master():
    np.random.seed(42)
    companies = []
    for i in range(1, 51):
        companies.append({
            "company_id": f"R{i:03d}",
            "company_name": f"収入先{i:02d}株式会社",
            "monthly_base": np.random.randint(50, 500) * 10000,
            "payment_cycle_days": np.random.choice([30, 45, 60]),
            "shipping_fee_rate": np.random.choice([0.0, 0.0, 0.0, 0.05, 0.07, 0.10]),
            "seasonal_peak": np.random.choice([3, 6, 7, 8, 12])
        })
    return pd.DataFrame(companies)

def generate_sample_expense_master():
    np.random.seed(123)
    companies = []
    for i in range(1, 51):
        companies.append({
            "company_id": f"E{i:03d}",
            "company_name": f"支払先{i:02d}株式会社",
            "monthly_base": np.random.randint(20, 300) * 10000,
            "payment_cycle_days": np.random.choice([30, 45, 60]),
            "category": np.random.choice(["燃料費", "人件費", "車両維持費", "保険料", "外注費", "事務用品", "通信費", "その他"])
        })
    return pd.DataFrame(companies)

# --- 推定ロジック ---
def estimate_daily_settlement(rev_master, exp_master, target_year, target_month, inflation_rate=0.02, seasonal_factors=None):
    if seasonal_factors is None:
        seasonal_factors = {1:0.85, 2:0.88, 3:1.05, 4:0.95, 5:0.92, 6:1.02, 7:1.10, 8:1.08, 9:0.98, 10:1.00, 11:1.05, 12:1.15}
    days_in_month = calendar.monthrange(target_year, target_month)[1]
    season_factor = seasonal_factors.get(target_month, 1.0)
    inflation_factor = 1 + inflation_rate
    records = []
    for day in range(1, days_in_month + 1):
        d = date(target_year, target_month, day)
        dow = d.weekday()
        is_weekday = dow < 5
        day_factor = 1.0 if is_weekday else 0.3
        daily_revenue = 0
        daily_shipping = 0
        for _, comp in rev_master.iterrows():
            base = comp["monthly_base"] / days_in_month
            peak_bonus = 1.2 if comp["seasonal_peak"] == target_month else 1.0
            noise = np.random.normal(1.0, 0.05)
            rev = base * season_factor * inflation_factor * day_factor * peak_bonus * noise
            ship = rev * comp["shipping_fee_rate"]
            daily_revenue += rev
            daily_shipping += ship
        daily_expense = 0
        for _, comp in exp_master.iterrows():
            base = comp["monthly_base"] / days_in_month
            noise = np.random.normal(1.0, 0.08)
            exp = base * season_factor * inflation_factor * day_factor * noise
            daily_expense += exp
        records.append({
            "日付": d,
            "曜日": ["月","火","水","木","金","土","日"][dow],
            "推定収入": round(daily_revenue),
            "運送費収入": round(daily_shipping),
            "収入合計": round(daily_revenue + daily_shipping),
            "推定支出": round(daily_expense),
            "推定損益": round(daily_revenue + daily_shipping - daily_expense),
        })
    df = pd.DataFrame(records)
    df["累計損益"] = df["推定損益"].cumsum()
    return df

# --- メインUI ---
def main():
    st.title("📊 AI推定日次決算ツール")
    st.caption("収入先50社・支払先50社の推定日次決算を自動生成します")
    menu = st.sidebar.radio("メニュー", ["📁 データ入力", "📈 日次推定レポート", "📊 分析ダッシュボード"])

    if "rev_master" not in st.session_state:
        st.session_state.rev_master = generate_sample_revenue_master()
    if "exp_master" not in st.session_state:
        st.session_state.exp_master = generate_sample_expense_master()

    if menu == "📁 データ入力":
        show_data_input()
    elif menu == "📈 日次推定レポート":
        show_daily_report()
    elif menu == "📊 分析ダッシュボード":
        show_dashboard()

def show_data_input():
    st.header("📁 データ入力")
    tab1, tab2, tab3 = st.tabs(["収入先マスター", "支払先マスター", "CSVアップロード"])
    with tab1:
        st.subheader("収入先50社マスター")
        st.info("運送費率には売上の7%が運送費として入金される場合は 0.07 と入力してください")
        edited_rev = st.data_editor(st.session_state.rev_master, num_rows="dynamic", use_container_width=True, key="rev_editor")
        if st.button("収入先マスターを保存", key="save_rev"):
            st.session_state.rev_master = edited_rev
            st.success("保存しました")
    with tab2:
        st.subheader("支払先50社マスター")
        edited_exp = st.data_editor(st.session_state.exp_master, num_rows="dynamic", use_container_width=True, key="exp_editor")
        if st.button("支払先マスターを保存", key="save_exp"):
            st.session_state.exp_master = edited_exp
            st.success("保存しました")
    with tab3:
        st.subheader("CSVアップロード")
        up_rev = st.file_uploader("収入先マスターCSV", type="csv", key="up_rev")
        if up_rev:
            st.session_state.rev_master = pd.read_csv(up_rev)
            st.success("収入先マスターを読み込みました")
        up_exp = st.file_uploader("支払先マスターCSV", type="csv", key="up_exp")
        if up_exp:
            st.session_state.exp_master = pd.read_csv(up_exp)
            st.success("支払先マスターを読み込みました")

def show_daily_report():
    st.header("📈 日次推定レポート")
    col1, col2, col3 = st.columns(3)
    with col1:
        target_year = st.number_input("年", min_value=2020, max_value=2030, value=datetime.now().year)
    with col2:
        target_month = st.number_input("月", min_value=1, max_value=12, value=datetime.now().month)
    with col3:
        inflation_rate = st.number_input("インフレ率(%)", min_value=-5.0, max_value=20.0, value=2.0, step=0.1) / 100

    st.subheader("季節指数設定")
    season_cols = st.columns(6)
    seasonal_factors = {}
    month_names = ["1月","2月","3月","4月","5月","6月","7月","8月","9月","10月","11月","12月"]
    defaults = [0.85, 0.88, 1.05, 0.95, 0.92, 1.02, 1.10, 1.08, 0.98, 1.00, 1.05, 1.15]
    for i in range(12):
        with season_cols[i % 6]:
            seasonal_factors[i+1] = st.number_input(month_names[i], min_value=0.1, max_value=2.0, value=defaults[i], step=0.01, key=f"sf_{i}")

    if st.button("🚀 日次推定を実行", type="primary", use_container_width=True):
        with st.spinner("推定計算中..."):
            df = estimate_daily_settlement(
                st.session_state.rev_master,
                st.session_state.exp_master,
                int(target_year), int(target_month),
                inflation_rate, seasonal_factors
            )
            st.session_state.daily_result = df

    if "daily_result" in st.session_state:
        df = st.session_state.daily_result
        st.subheader("📌 月間KPI")
        k1, k2, k3, k4 = st.columns(4)
        total_rev = df["収入合計"].sum()
        total_exp = df["推定支出"].sum()
        total_pl = df["推定損益"].sum()
        weekday_avg = df[df["曜日"].isin(["月","火","水","木","金"])]["推定損益"].mean()
        k1.metric("月間収入", f"¥{total_rev:,.0f}")
        k2.metric("月間支出", f"¥{total_exp:,.0f}")
        k3.metric("月間損益", f"¥{total_pl:,.0f}", delta=f"利益率{total_pl/total_rev*100:.1f}%" if total_rev > 0 else "")
        k4.metric("平日平均損益", f"¥{weekday_avg:,.0f}")

        st.subheader("📅 日次テーブル")
        def color_pl(val):
            if isinstance(val, (int, float)):
                return "color: green" if val >= 0 else "color: red"
            return ""
        styled = df.style.applymap(color_pl, subset=["推定損益", "累計損益"]).format({
            "推定収入": "¥{:,.0f}", "運送費収入": "¥{:,.0f}", "収入合計": "¥{:,.0f}",
            "推定支出": "¥{:,.0f}", "推定損益": "¥{:,.0f}", "累計損益": "¥{:,.0f}"
        })
        st.dataframe(styled, use_container_width=True, height=600)

        st.subheader("📈 日次チャート")
        chart_df = df[["日付", "収入合計", "推定支出", "推定損益"]].set_index("日付")
        st.line_chart(chart_df)

        st.subheader("📈 累計損益推移")
        cum_df = df[["日付", "累計損益"]].set_index("日付")
        st.area_chart(cum_df)

        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("💾 CSVダウンロード", csv, f"daily_settlement_{int(target_year)}_{int(target_month):02d}.csv", "text/csv")

def show_dashboard():
    st.header("📊 分析ダッシュボード")
    if "daily_result" not in st.session_state:
        st.warning("先に「日次推定レポート」で推定を実行してください")
        return
    df = st.session_state.daily_result

    st.subheader("📅 週次集計")
    df_copy = df.copy()
    df_copy["日付"] = pd.to_datetime(df_copy["日付"])
    df_copy["週"] = df_copy["日付"].dt.isocalendar().week.astype(int)
    weekly = df_copy.groupby("週").agg({"収入合計": "sum", "推定支出": "sum", "推定損益": "sum"}).reset_index()
    st.dataframe(weekly.style.format({"収入合計": "¥{:,.0f}", "推定支出": "¥{:,.0f}", "推定損益": "¥{:,.0f}"}), use_container_width=True)

    st.subheader("📆 曜日別平均損益")
    dow_order = ["月","火","水","木","金","土","日"]
    dow_avg = df.groupby("曜日")["推定損益"].mean().reindex(dow_order)
    st.bar_chart(dow_avg)

    st.subheader("🎯 目標管理")
    target_pl = st.number_input("月間目標損益(円)", value=5000000, step=100000)
    actual_pl = df["推定損益"].sum()
    progress = min(actual_pl / target_pl, 1.0) if target_pl > 0 else 0
    st.progress(progress)
    st.write(f"達成率: {progress*100:.1f}% (推定損益 ¥{actual_pl:,.0f} / 目標 ¥{target_pl:,.0f})")

    if actual_pl >= target_pl:
        st.success("目標達成！")
    else:
        remaining = target_pl - actual_pl
        days_left = len(df[df["曜日"].isin(["月","火","水","木","金"])])
        if days_left > 0:
            daily_needed = remaining / days_left
            st.info(f"目標まであと ¥{remaining:,.0f}。平日あたり ¥{daily_needed:,.0f} の損益が必要です")


if __name__ == "__main__":
    main()

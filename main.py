"""
這個 main5.py 是用來執行各種分析的主程式，包含：
- 單一決策分析（run_single_decision）
- 每年定期投資分析（run_yearly_backtest）
- 參數網格搜尋（run_grid_search_yearly）
- 高點情境分析（run_peak_scenarios）
- 長期蒙地卡羅模擬（run_long_horizon_mc）
- 根據特定條件（如近高點）進行蒙地卡羅模擬（run_conditional_high_mc）
每個分析函式都會產出結果表格，並儲存在 reports/tables/ 目錄下。
長期蒙地卡羅模擬還會產生圖表，儲存在 reports/figures/ 目錄下。


"""
import re
from src.analysis.conditional_mc import mc_condition_on_high_state
from src.analysis.long_horizon import (
    to_trade_price_series,
    long_horizon_expected_value_module,
    # plot_hist,
    DCAPlan
)
from src.backtest.engine import backtest_all_in, backtest_dca, compute_metrics
from src.data.load_data import load_price_data
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any
from matplotlib import pyplot as plt
from src.utils.io_helpers import save_figure, save_csv
import matplotlib
matplotlib.use("Agg")


# ============================================================
# IO
# ============================================================

def slugify(text: str) -> str:
    text = str(text).strip()
    text = text.replace(".TW", "")
    text = text.replace(".TX", "")
    text = text.replace(" ", "_")
    text = text.replace(":", "-")
    text = text.replace("/", "-")
    text = text.replace("\\", "-")
    text = re.sub(r"[^A-Za-z0-9_\-\.]", "", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def date_range_tag(start: str, end: str) -> str:
    return f"{slugify(start)}_to_{slugify(end)}"


def build_output_name(
    analysis: str,
    ticker: str,
    start: str,
    end: str,
    ext: str,
    **params,
) -> str:
    parts = [
        slugify(analysis),
        slugify(ticker),
        date_range_tag(start, end),
    ]

    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, float) and float(v).is_integer():
            v = int(v)
        parts.append(f"{slugify(k)}-{slugify(v)}")

    return "__".join(parts) + f".{ext}"


def save_output_table(
    df: pd.DataFrame,
    analysis: str,
    ticker: str,
    start: str,
    end: str,
    category: str = "model_outputs",
    **params,
):
    filename = build_output_name(
        analysis=analysis,
        ticker=ticker,
        start=start,
        end=end,
        ext="csv",
        **params,
    )
    return save_csv(df, filename, category=category)


def save_output_figure(
    fig,
    analysis: str,
    ticker: str,
    start: str,
    end: str,
    category: str = "model_outputs",
    **params,
):
    filename = build_output_name(
        analysis=analysis,
        ticker=ticker,
        start=start,
        end=end,
        ext="png",
        **params,
    )
    return save_figure(fig, filename, category=category)


# ============================================================
# Core helpers
# ============================================================

def get_trade_df(df: pd.DataFrame) -> pd.DataFrame:
    # 檢查 dtype，避免"False"等判定為資料而顯示 True
    if not df["is_trade_day"].dropna().isin([True, False]).all():
        raise ValueError("is_trade_day contains non-boolean values")

    mask = df["is_trade_day"] == True
    trade_df = df.loc[mask].sort_index()
    return trade_df


def get_next_trade_date(trade_df: pd.DataFrame, base_date) -> pd.Timestamp:
    base_date = pd.to_datetime(base_date)
    future = trade_df.loc[trade_df.index >= base_date]
    if future.empty:
        raise ValueError("No future trade day found after base_date.")
    return future.index[0]


def slice_horizon(price_series: pd.Series, horizon_td) -> pd.Series:
    """
    horizon_td:
      - int: 取前 N 個交易日
      - None 或 "to_end": 持有到資料結尾
    """
    if horizon_td is None or horizon_td == "to_end":
        return price_series
    return price_series.iloc[: int(horizon_td)]


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """
    1) 統一 key 命名（snake_case）
    2) 統一型態：n_splits/gap_days/horizon_td
    """
    p = dict(params)

    # required common params
    p["n_splits"] = int(p.get("n_splits", 3))
    p["gap_days"] = int(p.get("gap_days", 20))

    h = p.get("horizon_td", 120)
    if h not in (None, "to_end"):
        h = int(h)
    p["horizon_td"] = h

    # optional
    if "base_month" in p and p["base_month"] is not None:
        p["base_month"] = int(p["base_month"])

    return p


# ============================================================
# Audit
# ============================================================

def audit_data(df: pd.DataFrame):
    required_cols = ["Close", "is_trade_day", "year", "month", "day"]
    missing = [c for c in required_cols if c not in df.columns]
    assert not missing, f"Missing columns: {missing}"

    assert pd.api.types.is_numeric_dtype(
        df["Close"]), f"Close dtype not numeric: {df['Close'].dtype}"

    assert set(df["is_trade_day"].dropna().unique()) <= {
        True, False}, "is_trade_day not boolean-like"

    trade_df = get_trade_df(df)

    # trade_df integrity
    assert not trade_df.empty, "trade_df is empty"
    assert trade_df["is_trade_day"].all(), "trade_df has non-trade-day rows"

    assert trade_df.index.is_monotonic_increasing, "trade_df index not sorted"
    assert trade_df.index.is_unique, "trade_df index has dulicates"

    # 基本價格檢查
    assert trade_df["Close"].notna().all(), "trade_df contants NaN Close"
    assert (trade_df["Close"] > 0).all(
    ), "trade_df contains non-positive Close"

    # # 台股歷史資料可能包含週六補班交易日，因此週末交易日僅提示，不直接視為錯誤
    weekend_trade_df = trade_df[trade_df.index.weekday >= 5]
    weekend_cnt = len(weekend_trade_df)

    if weekend_cnt > 0:
        print(
            f"INFO: trade_df contains {weekend_cnt} weekend trade dates "
            f"(possible special TWSE make-up trading days)."
        )
        print(weekend_trade_df[["Close", "is_trade_day"]].head())

    print("AUDIT OK:",
          f"df_rows={len(df)} trade_rows={len(trade_df)}",
          f"date_range={df.index.min().date()}..{df.index.max().date()}")


# ============================================================
# Single decision
# ============================================================

def run_single_decision(
    df: pd.DataFrame,
    base_date,
    n_splits: int = 3,
    gap_days: int = 20,
    horizon_td=250,
    min_trade_days: int = 10,
) -> dict:
    """
    給定一個 base_date，假設投資人從下一個交易日開始進場，然後比較 All-in 與 DCA 在未來一段期間內的績效與風險。

    base_date -> 下一交易日 entry_date -> 從 entry_date 回測到 horizon_td 或資料尾
    """
    trade_df = get_trade_df(df)
    entry_date = get_next_trade_date(trade_df, base_date)

    price = trade_df.loc[entry_date:, "Close"].copy()
    price = slice_horizon(price, horizon_td)

    if len(price) == 0:
        return {
            "Base_Date": pd.to_datetime(base_date).date(),
            "Entry_Date": entry_date.date(),
            "error": "Empty price path",
        }

    exit_date = price.index[-1]

    if len(price) < min_trade_days:
        return {
            "Base_Date": pd.to_datetime(base_date).date(),
            "Entry_Date": entry_date.date(),
            "Exit_Date": exit_date.date(),
            "Holding_TradeDays": int(len(price)),
            "Holding_Years_Approx": float(len(price) / 250.0),
            "n_splits": int(n_splits),
            "gap_days": int(gap_days),
            "error": "Not enough trade days",
        }

    nav_all = backtest_all_in(price)
    met_all = compute_metrics(nav_all)

    nav_dca = backtest_dca(price, n_splits=int(
        n_splits), gap_days=int(gap_days))
    met_dca = compute_metrics(nav_dca)

    return {
        "Base_Date": pd.to_datetime(base_date).date(),
        "Entry_Date": entry_date.date(),
        "Exit_Date": exit_date.date(),
        "Holding_TradeDays": int(len(price)),
        "Holding_Years_Approx": float(len(price) / 250.0),

        "n_splits": int(n_splits),
        "gap_days": int(gap_days),

        "AllIn_Return": float(met_all["Total_Return"]),
        "AllIn_MDD": float(met_all["Max_Drawdown"]),

        "DCA_Return": float(met_dca["Total_Return"]),
        "DCA_MDD": float(met_dca["Max_Drawdown"]),

        "DCA_Beat_AllIn": bool(met_dca["Total_Return"] > met_all["Total_Return"]),
    }


# ============================================================
# Yearly backtest
# ============================================================

def run_yearly_backtest(
    df: pd.DataFrame,
    n_splits: int = 3,
    gap_days: int = 20,
    base_month: int = 1,
    horizon_td: int = 250,
    min_trade_days: int = 30,
):
    trade_df = get_trade_df(df)
    results = []

    for year, data in trade_df.groupby(trade_df["year"]):
        m = data.loc[data["month"] == base_month]
        if m.empty:
            continue

        entry_date = m.index[0]
        price = data.loc[entry_date:, "Close"]
        price = slice_horizon(price, horizon_td)

        if len(price) < min_trade_days:
            continue

        nav_all = backtest_all_in(price)
        met_all = compute_metrics(nav_all)

        nav_dca = backtest_dca(price, n_splits=int(
            n_splits), gap_days=int(gap_days))
        met_dca = compute_metrics(nav_dca)

        results.append({
            "Year": int(year),
            "Entry_Date": entry_date.date(),
            "Exit_Date": price.index[-1].date(),
            "Horizon_TradeDays": int(len(price)),

            "n_splits": int(n_splits),
            "gap_days": int(gap_days),

            "AllIn_Return": float(met_all["Total_Return"]),
            "AllIn_MDD": float(met_all["Max_Drawdown"]),
            "DCA_Return": float(met_dca["Total_Return"]),
            "DCA_MDD": float(met_dca["Max_Drawdown"]),
            "DCA_Beat_AllIn": bool(met_dca["Total_Return"] > met_all["Total_Return"]),
        })

    res = pd.DataFrame(results).sort_values("Year").reset_index(drop=True)

    summary = {
        "Years": int(len(res)),
        "n_splits": int(n_splits),
        "gap_days": int(gap_days),
        "horizon_td": (horizon_td if horizon_td in (None, "to_end") else int(horizon_td)),

        "WinRate": float(res["DCA_Beat_AllIn"].mean()) if len(res) else float("nan"),
        "Avg_Return_Diff(DCA-AllIn)": float((res["DCA_Return"] - res["AllIn_Return"]).mean()) if len(res) else float("nan"),
        "Avg_MDD_Diff(DCA-AllIn)": float((res["DCA_MDD"] - res["AllIn_MDD"]).mean()) if len(res) else float("nan"),
        "AllIn_Worst5%": float(res["AllIn_Return"].quantile(0.05)) if len(res) else float("nan"),
        "DCA_Worst5%": float(res["DCA_Return"].quantile(0.05)) if len(res) else float("nan"),
    }
    summary_df = pd.DataFrame([summary])
    return res, summary_df


# ============================================================
# Grid search (yearly)
# ============================================================

def run_grid_search_yearly(
    df: pd.DataFrame,
    n_splits_list: list[int],
    gap_days_list: list[int],
    base_month: int = 1,
    horizon_td=250,
) -> pd.DataFrame:
    grid = []
    for n in n_splits_list:
        for g in gap_days_list:
            _, summary_df = run_yearly_backtest(
                df,
                n_splits=int(n),
                gap_days=int(g),
                base_month=int(base_month),
                horizon_td=horizon_td,
            )
            grid.append(summary_df.iloc[0].to_dict())

    grid_df = pd.DataFrame(grid).sort_values(
        ["Avg_MDD_Diff(DCA-AllIn)", "Avg_Return_Diff(DCA-AllIn)"],
        ascending=[True, False]
    ).reset_index(drop=True)

    for c in ["Years", "n_splits", "gap_days"]:
        if c in grid_df.columns:
            grid_df[c] = pd.to_numeric(
                grid_df[c], errors="coerce").astype("Int64")

    return grid_df


# ============================================================
# Peak scenarios
# ============================================================


def pick_peak_base_dates_rolling(
    trade_df: pd.DataFrame,
    top_n: int = 10,
    min_gap_days: int = 30,
    min_future_trade_days: int = 250,
    window_td: int = 252,
    min_periods: int | None = None,
) -> list[pd.Timestamp]:
    """
    從股價資料中，挑出「近一段時間創新高」的日期，作為高點情境分析的 base dates

    Rolling Peak 定義：
      - 在 trade_df 上，用 window_td (交易日) 做 rolling max
      - 當日 Close >= rolling_max (允許浮點誤差) -> 視為「近 window_td 日新高」

    再從這些新高日中：
      - 依 Close 由高到低排序
      - 用 min_gap_days 去重（避免同一波連續創高）
      - 用 min_future_trade_days 過濾（避免 horizon 不足）
      - 取 top_n 個 base_dates
    """
    trade_df = trade_df.sort_index()
    s = trade_df["Close"].astype(float)

    if min_periods is None:
        min_periods = window_td  # 一定要滿 window 才算新高（較嚴謹）

    rolling_max = s.rolling(window=window_td, min_periods=min_periods).max()

    # 浮點誤差容忍：Close 幾乎等於 rolling_max 就算
    is_peak = s >= (rolling_max * (1 - 1e-12))

    peak_candidates = trade_df.loc[is_peak].copy()
    if peak_candidates.empty:
        return []

    # 依 Close 高到低排序（較符合「高點情境」）
    peak_candidates = peak_candidates.sort_values("Close", ascending=False)

    picked: list[pd.Timestamp] = []
    for dt in peak_candidates.index:
        remaining = len(trade_df.loc[dt:])
        if remaining < int(min_future_trade_days):
            continue

        if all(abs((dt - p).days) >= int(min_gap_days) for p in picked):
            picked.append(dt)

        if len(picked) >= int(top_n):
            break

    return picked


def pick_peak_base_dates(
    trade_df: pd.DataFrame,
    top_n: int = 10,
    min_gap_days: int = 30,
    min_future_trade_days: int = 250,
) -> list[pd.Timestamp]:
    """
    trade_df: 交易日資料（index=trade date）
    """
    s = trade_df["Close"].copy()
    candidates = s.sort_values(ascending=False)

    picked: list[pd.Timestamp] = []
    for dt in candidates.index:
        remaining = len(trade_df.loc[dt:])
        if remaining < int(min_future_trade_days):
            continue

        if all(abs((dt - p).days) >= int(min_gap_days) for p in picked):
            picked.append(dt)

        if len(picked) >= int(top_n):
            break

    return picked


def run_peak_scenarios(
    df: pd.DataFrame,
    top_n: int = 10,
    min_gap_days: int = 30,
    n_splits: int = 3,
    gap_days: int = 20,
    horizon_td=250,
    drop_error_rows: bool = True,
    window_td: int = 252,
    min_periods=252
) -> pd.DataFrame:
    trade_df = get_trade_df(df)

    # min_future_trade_days: 固定 horizon 才有可比性；to_end 則不限制
    if horizon_td is None or horizon_td == "to_end":
        min_future_trade_days = 0
    else:
        min_future_trade_days = int(horizon_td)

    base_dates = pick_peak_base_dates_rolling(
        trade_df,
        top_n=int(top_n),
        min_gap_days=int(min_gap_days),
        min_future_trade_days=min_future_trade_days,
        window_td=int(window_td),
        min_periods=int(min_periods)
    )

    rows = []
    for bd in base_dates:
        rows.append(run_single_decision(
            trade_df,  # 傳 trade_df 進去，確保交易日
            bd,
            n_splits=int(n_splits),
            gap_days=int(gap_days),
            horizon_td=horizon_td,
        ))

    out = pd.DataFrame(rows).sort_values("Entry_Date").reset_index(drop=True)
    if drop_error_rows and "error" in out.columns:
        out = out[out["error"].isna()].reset_index(drop=True)
    return out


def run_long_horizon_mc(
    df: pd.DataFrame,
    ticker="0050.TX",
    start="2015-01-01",
    end=datetime.today().strftime("%Y-%m-%d"),
    horizons: list = [10, 20],  # 投資幾年
    n_sims: int = 10_000,
    method='gbm',  # "bootstrap" or "gbm"
    dca_months: int = 12,  # 分幾個月投入
    # 可以設定一個合理的無風險利率（rf_annual），例如 10年期美國公債殖利率，來計算夏普比率。這樣可以更貼近投資者的機會成本，而不是假設 rf=0。
    rf_annual: float = 0.0141,
    # 可以設定一個合理的 MAR（Minimum Acceptable Return），例如 CAPM 預期報酬率或 0% 實質報酬率，來計算下行風險（MAR ratio）。這樣可以更貼近投資者的風險偏好，而不是單純看波動率。
    mar_annual: float = 0.09,
    seed: int = 42,
    batch_size: int = 10_000,  # 每批次模擬多少條路徑，避免一次模擬太多導致記憶體爆掉
    save_outputs: bool = True,
    output_category: str = "model_outputs",
) -> pd.DataFrame:

    # takes Close price Series
    price = to_trade_price_series(df, price_col="Close")

    plan = DCAPlan(
        months=int(dca_months),
        gap_trade_days=252 // int(dca_months)
    )

    all_rows = []

    for hy in horizons:
        res = long_horizon_expected_value_module(
            price=price,
            horizon_years=int(hy),
            n_sims=n_sims,
            method=method,
            dca_plan=plan,
            rf_annual=float(rf_annual),
            mar_annual=float(mar_annual),
            seed=int(seed),
            batch_size=int(batch_size),
        )

        if save_outputs:
            # 1) 存每個 horizon 的 summary tables
            save_output_table(
                res["summary_terminal_all_in"],
                analysis="long_horizon_terminal_summary",
                ticker=ticker,
                start=start,
                end=end,
                category=output_category,
                strategy="all_in",
                horizon_years=hy,
                method=method,
            )

            save_output_table(
                res["summary_terminal_dca"],
                analysis="long_horizon_terminal_summary",
                ticker=ticker,
                start=start,
                end=end,
                category=output_category,
                strategy="dca",
                horizon_years=hy,
                method=method,
            )

            save_output_table(
                res["summary_cagr_all_in"],
                analysis="long_horizon_cagr_summary",
                ticker=ticker,
                start=start,
                end=end,
                category=output_category,
                strategy="all_in",
                horizon_years=hy,
                method=method,
            )

            save_output_table(
                res["summary_cagr_dca"],
                analysis="long_horizon_cagr_summary",
                ticker=ticker,
                start=start,
                end=end,
                category=output_category,
                strategy="dca",
                horizon_years=hy,
                method=method,
            )
        # 2) 產出「一行摘要」方便放總表
        all_rows.append({
            "horizon_years": int(hy),
            "method": method,
            "n_sims": n_sims,
            "dca_months": int(dca_months),
            "expected_cagr_all_in": res["expected_cagr_all_in"],
            "expected_cagr_dca": res["expected_cagr_dca"],
            "expected_terminal_all_in": res["expected_terminal_all_in"],
            "expected_terminal_dca": res["expected_terminal_dca"],
            "p_dca_beats_all_in_terminal": res["p_dca_beats_all_in_terminal"],
            "p_dca_beats_all_in_cagr": res["p_dca_beats_all_in_cagr"],
            "ann_geo_return_hist": res["ann_geo_return_hist"],
            "ann_vol_hist": res["ann_vol_hist"],
        })

        # 3) 畫圖（終值倍數 + CAGR）
        # 1. Terminal Multiple Distribution
        bins = 100
        xlims = (0, np.percentile(res["terminal_multiple_all_in"], 99.5) * 1.5)
        title = f"{hy}Y Terminal Multiple Distribution ({method})"
      
        plt.figure(figsize=(8, 6))
        plt.hist(res["terminal_multiple_all_in"], bins=bins, density=True,
                 label="All-in", alpha=0.3, color="skyblue", edgecolor="navy")
        plt.hist(res["terminal_multiple_dca"], bins=bins, density=True,
                 label="DCA", alpha=0.3, color="lightcoral", edgecolor="black")

        plt.title(title)
        plt.xlabel("Future Value / Initial Capital of All-in and DCA Strategies")
        plt.xlim(xlims)
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save_outputs:
            plt.tight_layout()
            fig = plt.gcf()
            save_output_figure(
                fig,
                analysis="long_horizon_terminal_distribution",
                ticker=ticker,
                start=start,
                end=end,
                category=output_category,
                horizon_years=hy,
                method=method,
            )

        # plt.show()
        plt.close()

        # 2. CAGR Distribution
        bins = 100
        title = f"{hy}Y CAGR Distribution ({method})"
        
        # Distribution of CAGR for both All-in and DCA
        plt.figure(figsize=(8, 6))
        plt.hist(res["cagr_all_in"], bins=bins, density=True,
                 label="All-in", alpha=0.3, color="skyblue", edgecolor="navy")
        plt.hist(res["cagr_dca"], bins=bins, density=True,
                 label="DCA", alpha=0.3, color="lightcoral", edgecolor="black")

        # All-in mean + std lines
        plt.axvline(res["expected_cagr_all_in"], color="blue", linestyle="--",
                    label=f"Mean of All-in: {res['expected_cagr_all_in']:.2%}")
        plt.axvline(res['expected_cagr_all_in'] + res['cagr_dca'].std(ddof=1), color="blue", linestyle="-.",
                    label=f"Mean + Std of All-in: {res['expected_cagr_all_in'] + res['cagr_dca'].std(ddof=1):.2%}")
        plt.axvline(res['expected_cagr_all_in'] - res['cagr_dca'].std(ddof=1), color="blue", linestyle="-.",
                    label=f"Mean - Std of All-in: {res['expected_cagr_all_in'] - res['cagr_dca'].std(ddof=1):.2%}")
        plt.axvline(res['expected_cagr_all_in'] + 2*res['cagr_dca'].std(ddof=1), color="blue", linestyle=":",
                    label=f"Mean + 2*Std of All-in: {res['expected_cagr_all_in'] + 2*res['cagr_dca'].std(ddof=1):.2%}")
        plt.axvline(res['expected_cagr_all_in'] - 2*res['cagr_dca'].std(ddof=1), color="blue", linestyle=":",
                    label=f"Mean - 2*Std of All-in: {res['expected_cagr_all_in'] - 2*res['cagr_dca'].std(ddof=1):.2%}")

        # DCA mean + std lines
        plt.axvline(res["expected_cagr_dca"], color="red", linestyle="--",
                    label=f"Mean of DCA: {res['expected_cagr_dca']:.2%}")
        plt.axvline(res['expected_cagr_dca'] + res['cagr_dca'].std(ddof=1), color="red", linestyle="-.",
                    label=f"Mean + Std of DCA: {res['expected_cagr_dca'] + res['cagr_dca'].std(ddof=1):.2%}")
        plt.axvline(res['expected_cagr_dca'] - res['cagr_dca'].std(ddof=1), color="red", linestyle="-.",
                    label=f"Mean - Std of DCA: {res['expected_cagr_dca'] - res['cagr_dca'].std(ddof=1):.2%}")
        plt.axvline(res['expected_cagr_dca'] + res['cagr_dca'].std(ddof=1), color="red", linestyle="-.",
                    label=f"Mean + Std of DCA: {res['expected_cagr_dca'] + res['cagr_dca'].std(ddof=1):.2%}")
        plt.axvline(res['expected_cagr_dca'] - res['cagr_dca'].std(ddof=1), color="red", linestyle="-.",
                    label=f"Mean - Std of DCA: {res['expected_cagr_dca'] - res['cagr_dca'].std(ddof=1):.2%}")
        plt.axvline(res['expected_cagr_dca'] + 2*res['cagr_dca'].std(ddof=1), color="red", linestyle=":",
                    label=f"Mean + 2*Std of DCA: {res['expected_cagr_dca'] + 2*res['cagr_dca'].std(ddof=1):.2%}")
        plt.axvline(res['expected_cagr_dca'] - 2*res['cagr_dca'].std(ddof=1), color="red", linestyle=":",
                    label=f"Mean - 2*Std of DCA: {res['expected_cagr_dca'] - 2*res['cagr_dca'].std(ddof=1):.2%}")

        plt.title(title)
        plt.xlabel("CAGR of All-in and DCA Strategies")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.legend()

        if save_outputs:
            plt.tight_layout()
            fig = plt.gcf() # ⭐ 抓目前這張圖
            save_output_figure(
                fig,
                analysis="long_horizon_cagr_distribution",
                ticker=ticker,
                start=start,
                end=end,
                category=output_category,
                horizon_years=hy,
                method=method,
            )
        plt.close()

    summary_df = pd.DataFrame(all_rows).sort_values(
        "horizon_years").reset_index(drop=True)

    if save_outputs:
        save_output_table(
            summary_df,
            analysis="long_horizon_overview",
            ticker=ticker,
            start=start,
            end=end,
            category=output_category,
            method=method,
            n_sims=n_sims,
            dca_months=dca_months,
        )

    print("=== Long Horizon MC Summary ===")
    print(summary_df)

    return summary_df


# ============================================================
# Main
# ============================================================

def main(MODE: str = "single"):

    BASE_PARAMS = {
        # data
        "ticker": "0050.TW",
        "start": "2015-01-01",
        "end": datetime.today().strftime("%Y-%m-%d"),
        "refresh": False,
        "trade_day_only": True,

        # common
        "n_splits": 5,
        "gap_days": 20,
        "horizon_td": 120,
    }

    MODE_PARAMS = {
        "single": {
            "custom_base_date": "2013-10-30",
            "horizon_td": "to_end",
        },
        "yearly": {
            "base_month": 1,
            "horizon_td": 250,
        },
        "grid_search": {
            "base_month": 1,
            "grid_splits": [2, 3, 4, 6, 12],
            "grid_gaps": [10, 20, 30],
            "horizon_td": 120,
        },
        "peak_scenarios": {
            "peak_top_n": 20,
            "peak_min_gap_days": 45,
            "horizon_td": 120,
            "n_splits": 3,
            "gap_days": 30,
            "peak_window_td": 252,
            "peak_min_periods": 252,
        },
        "long_horizon_mc": {
            "horizons": [10, 20],  # 由 long_horizon 模組內部控制
            "n_sims": 100_000,
            "method": "gbm",    # "bootstrap" or "gbm"
            "dca_months": 12,   # 分幾個月投入
            # 可以設定一個合理的無風險利率（rf_annual），例如 10年期美國公債殖利率，來計算夏普比率。這樣可以更貼近投資者的機會成本，而不是假設 rf=0。
            "rf_annual": 0.0141,
            # 可以設定一個合理的 MAR（Minimum Acceptable Return），例如 CAPM 預期報酬率或 0% 實質報酬率，來計算下行風險（MAR ratio）。這樣可以更貼近投資者的風險偏好，而不是單純看波動率。
            "mar_annual": 0.09,
            "seed": 42,
            "batch_size": 10_000,
            "save_outputs": True,
            "output_category": "model_outputs",
        },
        "conditional_high_mc": {
            "horizons": [10, 20],
            "n_sims": 1e5,
            "window_td": 252,
            "near_high_threshold": -0.01,
            "dca_months": 3
        },
    }

    if MODE not in MODE_PARAMS:
        raise ValueError(
            f"Unknown MODE: {MODE}. Valid: {list(MODE_PARAMS.keys())}")

    PARAMS = normalize_params(BASE_PARAMS | MODE_PARAMS.get(MODE, {}))

    # load data using params
    df = load_price_data(
        ticker=PARAMS["ticker"],
        start=PARAMS["start"],
        end=PARAMS["end"],
        refresh=PARAMS["refresh"],
        trade_day_only=PARAMS["trade_day_only"],
    )
    audit_data(df)

    if MODE == "single":
        result = run_single_decision(
            df,
            PARAMS["custom_base_date"],
            n_splits=PARAMS["n_splits"],
            gap_days=PARAMS["gap_days"],
            horizon_td=PARAMS["horizon_td"],
        )
        print("=== Single Decision ===")
        print(result)

        save_output_table(
            pd.DataFrame([result]),
            analysis="single_decision",
            ticker=PARAMS["ticker"],
            start=PARAMS["start"],
            end=PARAMS["end"],
            category="model_outputs",
            base_date=PARAMS["custom_base_date"],
            horizon_td=PARAMS["horizon_td"],
            n_splits=PARAMS["n_splits"],
            gap_days=PARAMS["gap_days"],
        )

        return

    if MODE == "yearly":
        res, summary_df = run_yearly_backtest(
            df,
            n_splits=PARAMS["n_splits"],
            gap_days=PARAMS["gap_days"],
            base_month=PARAMS["base_month"],
            horizon_td=PARAMS["horizon_td"],
        )
        print("=== Yearly (head) ===")
        print(res.head(10))
        print("\n=== Summary ===")
        print(summary_df)
        
        save_output_table(
            res,
            analysis="yearly_backtest_results",
            ticker=PARAMS["ticker"],
            start=PARAMS["start"],
            end=PARAMS["end"],
            category="model_outputs",
            base_month=PARAMS["base_month"],
            horizon_td=PARAMS["horizon_td"],
            n_splits=PARAMS["n_splits"],
            gap_days=PARAMS["gap_days"],
        )

        save_output_table(
            summary_df,
            analysis="yearly_backtest_summary",
            ticker=PARAMS["ticker"],
            start=PARAMS["start"],
            end=PARAMS["end"],
            category="model_outputs",
            base_month=PARAMS["base_month"],
            horizon_td=PARAMS["horizon_td"],
            n_splits=PARAMS["n_splits"],
            gap_days=PARAMS["gap_days"],
        )

        return

    if MODE == "grid_search":
        grid_df = run_grid_search_yearly(
            df,
            PARAMS["grid_splits"],
            PARAMS["grid_gaps"],
            base_month=PARAMS["base_month"],
            horizon_td=PARAMS["horizon_td"],
        )
        print("=== Grid Search (head) ===")
        print(grid_df.head(10))

        save_output_table(
            grid_df,
            analysis="yearly_grid_search",
            ticker=PARAMS["ticker"],
            start=PARAMS["start"],
            end=PARAMS["end"],
            category="model_outputs",
            base_month=PARAMS["base_month"],
            horizon_td=PARAMS["horizon_td"],
        )

        return

    if MODE == "peak_scenarios":
        peak_df = run_peak_scenarios(
            df,
            top_n=PARAMS["peak_top_n"],
            min_gap_days=PARAMS["peak_min_gap_days"],
            n_splits=PARAMS["n_splits"],
            gap_days=PARAMS["gap_days"],
            horizon_td=PARAMS["horizon_td"],
            window_td=PARAMS["peak_window_td"],
            min_periods=PARAMS["peak_min_periods"],
        )
        print("=== Peak Scenarios ===")
        print(peak_df)

        save_output_table(
            peak_df,
            analysis="peak_scenarios",
            ticker=PARAMS["ticker"],
            start=PARAMS["start"],
            end=PARAMS["end"],
            category="model_outputs",
            top_n=PARAMS["peak_top_n"],
            min_gap_days=PARAMS["peak_min_gap_days"],
            horizon_td=PARAMS["horizon_td"],
            n_splits=PARAMS["n_splits"],
            gap_days=PARAMS["gap_days"],
            window_td=PARAMS["peak_window_td"],
        )
        return

    if MODE == "long_horizon_mc":
        summary_df = run_long_horizon_mc(
            df,
            horizons=PARAMS["horizons"],
            n_sims=PARAMS["n_sims"],
            method=PARAMS["method"],
            dca_months=PARAMS["dca_months"],
            # ticker= "0050.TX",
            # 可以設定一個合理的無風險利率（rf_annual），例如 10年期美國公債殖利率，來計算夏普比率。這樣可以更貼近投資者的機會成本，而不是假設 rf=0。
            rf_annual=PARAMS["rf_annual"],
            # 可以設定一個合理的 MAR（Minimum Acceptable Return），例如 CAPM 預期報酬率或 0% 實質報酬率，來計算下行風險（MAR ratio）。這樣可以更貼近投資者的風險偏好，而不是單純看波動率。
            mar_annual=PARAMS["mar_annual"],
            seed=PARAMS['seed'],
            batch_size=PARAMS['batch_size'],
            save_outputs=PARAMS['save_outputs'],
            output_category=PARAMS['output_category']

        )
       
        return

    if MODE == "conditional_high_mc":

        price = df.loc[df["is_trade_day"], "Close"].astype(float)

        for hy in PARAMS["horizons"]:
            res = mc_condition_on_high_state(
                price=price,
                horizon_years=hy,
                n_sims=PARAMS["n_sims"],
                window_td=PARAMS["window_td"],
                state_mode="near_high",
                near_high_threshold=PARAMS["near_high_threshold"],
                breakout_only=True,
                first_block_days=252,
                dca_months=PARAMS["dca_months"]
            )

            print("\n==== Horizon:", hy, "Years ====")
            print("High-state sample count:", res["n_candidates"])
            print("P(DCA beats All-in):", res["p_dca_beats_all_in_terminal"])
            print("E[CAGR All-in]:", res["expected_cagr_all_in"])
            print("E[CAGR DCA]:", res["expected_cagr_dca"])

            out_df = pd.DataFrame([{
                "horizon_years": hy,
                "p_dca_win": res["p_dca_beats_all_in_terminal"],
                "expected_cagr_all_in": res["expected_cagr_all_in"],
                "expected_cagr_dca": res["expected_cagr_dca"],
                "n_candidates": res["n_candidates"],
            }])

            save_output_table(
                out_df,
                analysis="conditional_high_mc",
                ticker=PARAMS["ticker"],
                start=PARAMS["start"],
                end=PARAMS["end"],
                category="model_outputs",
                horizon_years=hy,
                n_sims=PARAMS["n_sims"],
                window_td=PARAMS["window_td"],
                near_high_threshold=PARAMS["near_high_threshold"],
                dca_months=PARAMS["dca_months"],
            )


if __name__ == "__main__":
    # "single" / "yearly" / "grid_search" / "peak_scenarios" / "long_horizon_mc" / "conditional_high_mc"

    # main(MODE="single")
    # main(MODE="yearly")
    # main(MODE="grid_search")
    # main(MODE="peak_scenarios")
    # main(MODE="long_horizon_mc")
    main(MODE="conditional_high_mc")

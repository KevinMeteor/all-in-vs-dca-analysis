"""
回測引擎：計算績效指標、模擬 All-in 和 DCA 策略的 NAV 序列

"""

import pandas as pd


# # ==========================================================
# # 基本績效指標
# # ==========================================================

def compute_metrics(nav_series: pd.Series) -> dict:
    nav_series = pd.to_numeric(nav_series, errors="coerce").dropna()

    total_return = float(nav_series.iloc[-1] / nav_series.iloc[0] - 1.0)

    rolling_max = nav_series.cummax()
    drawdown = nav_series / rolling_max - 1.0
    max_dd = float(drawdown.min())

    return {
        "Total_Return": total_return,
        "Max_Drawdown": max_dd
    }


# # ==========================================================
# # All-in 策略
# # ==========================================================

def backtest_all_in(price_series: pd.Series, initial_capital: float = 1.0) -> pd.Series:
    price_series = pd.to_numeric(price_series, errors="coerce").dropna()

    shares = initial_capital / float(price_series.iloc[0])
    nav_series = shares * price_series
    nav_series.name = "NAV"

    return nav_series

# # ==========================================================
# # DCA 策略
# # ==========================================================


def backtest_dca(price_series: pd.Series,
                 initial_capital: float = 1.0,
                 n_splits: int = 3,
                 gap_days: int = 20) -> pd.Series:

    price_series = pd.to_numeric(price_series, errors="coerce").dropna()

    capital_per_batch = initial_capital / n_splits
    shares = 0.0
    cash = initial_capital

    nav_list = []

    for i, dt in enumerate(price_series.index):

        if (i % gap_days == 0) and (cash > 1e-12):
            invest = min(capital_per_batch, cash)
            shares += invest / float(price_series.loc[dt])
            cash -= invest

        nav = shares * float(price_series.loc[dt]) + cash
        nav_list.append(nav)

    return pd.Series(nav_list, index=price_series.index, name="NAV")

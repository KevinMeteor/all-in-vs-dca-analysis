"""
Long-horizon expected value module:
- Build historical log returns
- Simulate 10/20-year paths
- Compare All-in vs DCA terminal distribution
- Report CAGR mean/median, Sharpe/Sortino (distribution-level approximations)
Notes:
- Sharpe/Sortino here are computed from simulated CAGR and simulated annual vol estimates
  using mu/sigma assumption. If you want path-wise realized vol, we can add it too.
"""


from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Any


TRADING_DAYS_PER_YEAR = 252


def to_trade_price_series(df: pd.DataFrame, price_col: str = "Close") -> pd.Series:
    """
    Ensure using trade-day price series.
    Expected df index is datetime, and df has 'is_trade_day' if calendar includes non-trade days.
    """
    if "is_trade_day" in df.columns:
        df = df.loc[df["is_trade_day"].astype(bool)].copy()
    s = df[price_col].astype(float).dropna().sort_index()
    return s


def daily_log_returns(price: pd.Series) -> pd.Series:
    # exp^{log{p_t} - log{p_{t-1}}} approx. (p_{t}/p_{t-1})
    # {log{p_t} - log{p_{t-1}}} approx. log{p_{t}/p_{t-1}}
    return np.log(price).diff().dropna()  # shape (n_days,) to (n_days-1,)


def annualize_return_from_logr(mean_logr: float) -> float:
    """Annualized geometric mean return from mean daily log return."""
    return float(np.exp(mean_logr * TRADING_DAYS_PER_YEAR) - 1.0)


def annualize_vol_from_logr(std_logr: float) -> float:
    return float(std_logr * np.sqrt(TRADING_DAYS_PER_YEAR))


def cagr_from_terminal_multiple(mult: np.ndarray, years: float) -> np.ndarray:
    # mult: terminal wealth / initial wealth
    # 年複合成長率 = (終值/初值)^(1/年數) - 1
    return np.power(mult, 1.0 / years) - 1.0  # shape (n_sims,)


def sharpe_ratio(ann_return: np.ndarray, ann_vol: np.ndarray, rf: float = 0.0) -> np.ndarray:
    # avoid divide by zero
    return (ann_return - rf) / np.maximum(ann_vol, 1e-12)


def sortino_ratio(ann_return: np.ndarray, ann_downside_vol: np.ndarray, rf: float = 0.0) -> np.ndarray:
    return (ann_return - rf) / np.maximum(ann_downside_vol, 1e-12)


def downside_vol_annualized(daily_logr: np.ndarray, mar: float = 0.0) -> float:
    """
    Downside deviation using daily log returns.
    mar: minimum acceptable return (annual). We convert to daily log MAR approx.
    """
    # Recommand that MAR ratio can be set to capital asset pricing model (CAPM) expected return,
    # or a fixed threshold like 0% real return.

    # convert annual MAR to daily log MAR
    mar_daily = np.log(1.0 + mar) / TRADING_DAYS_PER_YEAR
    downside = np.minimum(daily_logr - mar_daily, 0.0)  # shape (n_days,)
    # shape (1,), this is the daily downside deviation
    dd = np.sqrt(np.mean(downside ** 2))
    return float(dd * np.sqrt(TRADING_DAYS_PER_YEAR))  # annualize


# auto-generates __init__, __repr__, __eq__, etc.
# https://docs.python.org/3/library/dataclasses.html
@dataclass
class DCAPlan:
    months: int = 12   # e.g. 12-month DCA
    # invest schedule in trade days:
    # invest every approx (TRADING_DAYS_PER_YEAR/12) ≈ 21 days
    gap_trade_days: int = 21


def simulate_paths_bootstrap(
    hist_logr: np.ndarray,
    n_days: int,
    n_sims: int,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Non-parametric bootstrap: sample daily log returns with replacement.
    Returns log-return paths shape (n_sims, n_days)
    """
    rng = np.random.default_rng(seed)
    # Return random integers from low (inclusive) to high (exclusive), shape (n_sims, n_days)
    idx = rng.integers(0, len(hist_logr), size=(n_sims, n_days))
    return hist_logr[idx]


def simulate_paths_gbm(
    mu_daily: float,
    sigma_daily: float,
    n_days: int,
    n_sims: int,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Parametric GBM log returns: r_t ~ N(mu_daily, sigma_daily^2)
    """
    rng = np.random.default_rng(seed)
    return rng.normal(loc=mu_daily, scale=sigma_daily, size=(n_sims, n_days))


def terminal_multiple_all_in(logr_paths: np.ndarray) -> np.ndarray:
    """
    All-in: terminal multiple = exp(sum log returns)
    """
    # \Sigma_{i=1}^N {log(P_{t+1}) - log(P_{t}} = log{P_{N}} - log{P_{1}} => exp(...) approx. P_{N}/P_{0}
    # => exp(...) = P_{N}/P_{1}
    # So, exp(sum log returns) gives us the terminal multiple directly.
    return np.exp(logr_paths.sum(axis=1))  # shape (n_sims,)


# def terminal_multiple_dca(
#     logr_paths: np.ndarray,
#     plan: DCAPlan,
#     initial_capital: float = 1.0
# ) -> np.ndarray:
#     """
#     DCA: invest equal capital each month (or batch), at scheduled trade-day indices.
#     Assumption: uninvested cash earns 0% (can extend later).
#     We compute terminal wealth multiple relative to initial_capital.
#     """
#     n_sims, n_days = logr_paths.shape
#     # relative price path, starts at day1 relative to day0
#     # shape (n_sims, n_days), price_rel[:, t] = P_{t}/P_{0} for each path
#     price_rel = np.exp(np.cumsum(logr_paths, axis=1))

#     # invest times (0-indexed day positions)
#     k = plan.months
#     gap = plan.gap_trade_days
#     # 0, gap, 2*gap, ..., (k-1)*gap, but capped at n_days-1 to avoid out-of-bounds
#     invest_idx = np.array([min(i * gap, n_days - 1)
#                           for i in range(k)], dtype=int)

#     cash_per_batch = initial_capital / k

#     # buy shares at each invest_idx at that day's relative price
#     # shares = sum (cash_per_batch / price_rel[:, t])
#     shares = np.zeros(n_sims, dtype=float)
#     for t in invest_idx:
#         shares += cash_per_batch / price_rel[:, t]

#     terminal_price = price_rel[:, -1]
#     terminal_wealth = shares * terminal_price

#     return terminal_wealth / initial_capital


def terminal_multiple_dca(
    logr_paths: np.ndarray,
    plan: DCAPlan,
    initial_capital: float = 1.0
) -> np.ndarray:
    """
    DCA terminal wealth multiple relative to initial_capital.

    Notes
    -----
    - Invest equal capital at scheduled trade-day indices.
    - Uninvested cash is assumed to earn 0%.
    - Memory-optimized: does not build full price_rel array.
    """
    n_sims, n_days = logr_paths.shape

    k = plan.months
    gap = plan.gap_trade_days

    invest_idx = np.array(
        [min(i * gap, n_days - 1) for i in range(k)],
        dtype=np.int32
    )

    cash_per_batch = initial_capital / k

    dtype = logr_paths.dtype
    cum_logr = np.zeros(n_sims, dtype=dtype)
    shares = np.zeros(n_sims, dtype=dtype)

    buy_ptr = 0
    n_buys = len(invest_idx)

    for t in range(n_days):
        cum_logr += logr_paths[:, t]

        while buy_ptr < n_buys and t == invest_idx[buy_ptr]:
            price_t = np.exp(cum_logr)
            shares += cash_per_batch / price_t
            buy_ptr += 1

    terminal_price = np.exp(cum_logr)
    terminal_wealth = shares * terminal_price

    return terminal_wealth / initial_capital


def summarize_distribution(x: np.ndarray, name: str = "x", probs=(0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99)) -> pd.DataFrame:
    q = np.quantile(x, probs)
    return pd.DataFrame({
        "metric": [f"{name}_q{int(p*100):02d}" for p in probs],
        "value": q
    })


def plot_hist(x: np.ndarray, title: str, path: Optional[str] = None, bins: int = 60):
    plt.figure()
    plt.hist(x, bins=bins, density=True)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True, alpha=0.3)
    if path:
        plt.tight_layout()
        plt.savefig(path, dpi=150)
    plt.show()


# def long_horizon_expected_value_module(
#     price: pd.Series,
#     horizon_years: int,
#     n_sims: int = 20000,
#     method: Literal["bootstrap", "gbm"] = "bootstrap",
#     dca_plan: Optional[DCAPlan] = None,
#     rf_annual: float = 0.0,  # for Sharpe/Sortino calculation
#     mar_annual: float = 0.0,  # for Sortino calculation
#     seed: int = 42,
# ) -> Dict[str, Any]:
#     """
#     Main entry:
#       - Build historical log returns
#       - Simulate 10/20-year paths
#       - Compare All-in vs DCA terminal distribution
#       - Report CAGR mean/median, Sharpe/Sortino (distribution-level approximations)

#     Notes:
#       - Sharpe/Sortino here are computed from simulated CAGR and simulated annual vol estimates
#         using mu/sigma assumption. If you want path-wise realized vol, we can add it too.
#     """
#     if dca_plan is None:
#         dca_plan = DCAPlan(
#             months=12, gap_trade_days=TRADING_DAYS_PER_YEAR // 12)

#     hist_lr = daily_log_returns(price).values
#     if len(hist_lr) < 500:
#         raise ValueError(
#             "Not enough historical data for stable long-horizon simulation.")

#     # total simulated days for the horizon (e.g. 10Y * 252 = 2520 days)
#     n_days = int(horizon_years * TRADING_DAYS_PER_YEAR)

#     # mean and std of historical daily log returns for GBM parameters or sanity check
#     mu_d = float(np.mean(hist_lr))
#     sig_d = float(np.std(hist_lr, ddof=1))

#     if method == "bootstrap":
#         # {use non-parametric bootstrap of historical log returns}
#         # *** Note that this assumes i.i.d. log returns, which may not capture all real-world dynamics,
#         #     but is a common approach for long-term simulations. ***
#         logr_paths = simulate_paths_bootstrap(
#             hist_lr, n_days=n_days, n_sims=n_sims, seed=seed)
#     elif method == "gbm":
#         # {use historical mu/sigma as GBM parameters}
#         # *** This assumes log returns are normally distributed and independent,
#         #     which is a strong assumption but serves as a useful parametric benchmark. ***
#         logr_paths = simulate_paths_gbm(
#             mu_daily=mu_d, sigma_daily=sig_d, n_days=n_days, n_sims=n_sims, seed=seed)
#     else:
#         raise ValueError("method must be 'bootstrap' or 'gbm'")

#     # Terminal multiples
#     mult_all = terminal_multiple_all_in(logr_paths)
#     mult_dca = terminal_multiple_dca(logr_paths, plan=dca_plan)

#     # Convert to CAGR
#     years = float(horizon_years)
#     cagr_all = cagr_from_terminal_multiple(mult_all, years)
#     cagr_dca = cagr_from_terminal_multiple(mult_dca, years)

#     # Annual vol proxy: from historical daily vol (same for both under model assumption)
#     ann_vol = annualize_vol_from_logr(sig_d)

#     # Downside vol proxy (Sortino): from historical daily log returns
#     ann_down = downside_vol_annualized(hist_lr, mar=mar_annual)

#     # Sharpe/Sortino using simulated CAGR as "ann_return"
#     sharpe_all = sharpe_ratio(cagr_all, ann_vol, rf=rf_annual)
#     sharpe_dca = sharpe_ratio(cagr_dca, ann_vol, rf=rf_annual)

#     sortino_all = sortino_ratio(cagr_all, ann_down, rf=rf_annual)
#     sortino_dca = sortino_ratio(cagr_dca, ann_down, rf=rf_annual)

#     # Key probabilities
#     out = {
#         "horizon_years": horizon_years,
#         "n_sims": n_sims,
#         "method": method,
#         "mu_daily_hist": mu_d,
#         "sigma_daily_hist": sig_d,
#         "ann_geo_return_hist": annualize_return_from_logr(mu_d),
#         "ann_vol_hist": ann_vol,
#         "ann_downside_vol_hist": ann_down,

#         # distributions
#         "terminal_multiple_all_in": mult_all,
#         "terminal_multiple_dca": mult_dca,
#         "cagr_all_in": cagr_all,
#         "cagr_dca": cagr_dca,
#         "sharpe_all_in": sharpe_all,
#         "sharpe_dca": sharpe_dca,
#         "sortino_all_in": sortino_all,
#         "sortino_dca": sortino_dca,

#         # summary tables
#         "summary_terminal_all_in": summarize_distribution(mult_all, "mult_all_in"),
#         "summary_terminal_dca": summarize_distribution(mult_dca, "mult_dca"),
#         "summary_cagr_all_in": summarize_distribution(cagr_all, "cagr_all_in"),
#         "summary_cagr_dca": summarize_distribution(cagr_dca, "cagr_dca"),
#         "summary_sharpe_all_in": summarize_distribution(sharpe_all, "sharpe_all_in"),
#         "summary_sharpe_dca": summarize_distribution(sharpe_dca, "sharpe_dca"),
#         "summary_sortino_all_in": summarize_distribution(sortino_all, "sortino_all_in"),
#         "summary_sortino_dca": summarize_distribution(sortino_dca, "sortino_dca"),

#         # win-rate style metrics
#         "p_dca_beats_all_in_terminal": float(np.mean(mult_dca > mult_all)),
#         "p_dca_beats_all_in_cagr": float(np.mean(cagr_dca > cagr_all)),
#         "expected_terminal_all_in": float(np.mean(mult_all)),
#         "expected_terminal_dca": float(np.mean(mult_dca)),
#         "expected_cagr_all_in": float(np.mean(cagr_all)),
#         "expected_cagr_dca": float(np.mean(cagr_dca)),
#         "median_cagr_all_in": float(np.median(cagr_all)),
#         "median_cagr_dca": float(np.median(cagr_dca)),
#     }

#     return out


def _simulate_logr_paths_batch(
    method: str,
    hist_lr: np.ndarray,
    mu_d: float,
    sig_d: float,
    n_days: int,
    batch_size: int,
    seed: int,
    dtype=np.float32,
) -> np.ndarray:
    """
    Generate one batch of simulated log-return paths.
    """
    if method == "bootstrap":
        logr_paths = simulate_paths_bootstrap(
            hist_lr, n_days=n_days, n_sims=batch_size, seed=seed
        )
    elif method == "gbm":
        logr_paths = simulate_paths_gbm(
            mu_daily=mu_d, sigma_daily=sig_d, n_days=n_days, n_sims=batch_size, seed=seed
        )
    else:
        raise ValueError("method must be 'bootstrap' or 'gbm'")

    return np.asarray(logr_paths, dtype=dtype)


def long_horizon_expected_value_module(
    price: pd.Series,
    horizon_years: int,
    n_sims: int = 20000,
    method: Literal["bootstrap", "gbm"] = "bootstrap",
    dca_plan: Optional[DCAPlan] = None,
    rf_annual: float = 0.0,
    mar_annual: float = 0.0,
    seed: int = 42,
    batch_size: int = 10000,
    dtype=np.float32,
) -> Dict[str, Any]:
    """
    Batch version:
      - build historical log returns
      - simulate in batches to reduce peak memory usage
      - compare All-in vs DCA terminal distribution
      - return the same output structure as before

    Notes
    -----
    - This version avoids materializing one huge (n_sims, n_days) matrix.
    - It still returns full simulation distributions, so downstream summary/plot code
      can remain mostly unchanged.
    """
    if dca_plan is None:
        dca_plan = DCAPlan(
            months=12,
            gap_trade_days=TRADING_DAYS_PER_YEAR // 12
        )

    hist_lr = daily_log_returns(price).values
    if len(hist_lr) < 500:
        raise ValueError(
            "Not enough historical data for stable long-horizon simulation.")

    n_days = int(horizon_years * TRADING_DAYS_PER_YEAR)

    mu_d = float(np.mean(hist_lr))
    sig_d = float(np.std(hist_lr, ddof=1))

    years = float(horizon_years)

    ann_vol = annualize_vol_from_logr(sig_d)
    ann_down = downside_vol_annualized(hist_lr, mar=mar_annual)

    mult_all_list = []
    mult_dca_list = []
    cagr_all_list = []
    cagr_dca_list = []
    sharpe_all_list = []
    sharpe_dca_list = []
    sortino_all_list = []
    sortino_dca_list = []

    sims_done = 0
    batch_id = 0

    while sims_done < n_sims:
        cur_n = min(batch_size, n_sims - sims_done)

        batch_seed = seed + batch_id

        logr_paths = _simulate_logr_paths_batch(
            method=method,
            hist_lr=hist_lr,
            mu_d=mu_d,
            sig_d=sig_d,
            n_days=n_days,
            batch_size=cur_n,
            seed=batch_seed,
            dtype=dtype,
        )

        mult_all = terminal_multiple_all_in(logr_paths)
        mult_dca = terminal_multiple_dca(logr_paths, plan=dca_plan)

        cagr_all = cagr_from_terminal_multiple(mult_all, years)
        cagr_dca = cagr_from_terminal_multiple(mult_dca, years)

        sharpe_all = sharpe_ratio(cagr_all, ann_vol, rf=rf_annual)
        sharpe_dca = sharpe_ratio(cagr_dca, ann_vol, rf=rf_annual)

        sortino_all = sortino_ratio(cagr_all, ann_down, rf=rf_annual)
        sortino_dca = sortino_ratio(cagr_dca, ann_down, rf=rf_annual)

        mult_all_list.append(np.asarray(mult_all, dtype=dtype))
        mult_dca_list.append(np.asarray(mult_dca, dtype=dtype))
        cagr_all_list.append(np.asarray(cagr_all, dtype=dtype))
        cagr_dca_list.append(np.asarray(cagr_dca, dtype=dtype))
        sharpe_all_list.append(np.asarray(sharpe_all, dtype=dtype))
        sharpe_dca_list.append(np.asarray(sharpe_dca, dtype=dtype))
        sortino_all_list.append(np.asarray(sortino_all, dtype=dtype))
        sortino_dca_list.append(np.asarray(sortino_dca, dtype=dtype))

        sims_done += cur_n
        batch_id += 1

    mult_all = np.concatenate(mult_all_list)
    mult_dca = np.concatenate(mult_dca_list)
    cagr_all = np.concatenate(cagr_all_list)
    cagr_dca = np.concatenate(cagr_dca_list)
    sharpe_all = np.concatenate(sharpe_all_list)
    sharpe_dca = np.concatenate(sharpe_dca_list)
    sortino_all = np.concatenate(sortino_all_list)
    sortino_dca = np.concatenate(sortino_dca_list)

    out = {
        "horizon_years": horizon_years,
        "n_sims": n_sims,
        "method": method,
        "batch_size": batch_size,
        "dtype": str(np.dtype(dtype)),
        "mu_daily_hist": mu_d,
        "sigma_daily_hist": sig_d,
        "ann_geo_return_hist": annualize_return_from_logr(mu_d),
        "ann_vol_hist": ann_vol,
        "ann_downside_vol_hist": ann_down,

        # distributions
        "terminal_multiple_all_in": mult_all,
        "terminal_multiple_dca": mult_dca,
        "cagr_all_in": cagr_all,
        "cagr_dca": cagr_dca,
        "sharpe_all_in": sharpe_all,
        "sharpe_dca": sharpe_dca,
        "sortino_all_in": sortino_all,
        "sortino_dca": sortino_dca,

        # summary tables
        "summary_terminal_all_in": summarize_distribution(mult_all, "mult_all_in"),
        "summary_terminal_dca": summarize_distribution(mult_dca, "mult_dca"),
        "summary_cagr_all_in": summarize_distribution(cagr_all, "cagr_all_in"),
        "summary_cagr_dca": summarize_distribution(cagr_dca, "cagr_dca"),
        "summary_sharpe_all_in": summarize_distribution(sharpe_all, "sharpe_all_in"),
        "summary_sharpe_dca": summarize_distribution(sharpe_dca, "sharpe_dca"),
        "summary_sortino_all_in": summarize_distribution(sortino_all, "sortino_all_in"),
        "summary_sortino_dca": summarize_distribution(sortino_dca, "sortino_dca"),

        # win-rate style metrics
        "p_dca_beats_all_in_terminal": float(np.mean(mult_dca > mult_all)),
        "p_dca_beats_all_in_cagr": float(np.mean(cagr_dca > cagr_all)),
        "expected_terminal_all_in": float(np.mean(mult_all)),
        "expected_terminal_dca": float(np.mean(mult_dca)),
        "expected_cagr_all_in": float(np.mean(cagr_all)),
        "expected_cagr_dca": float(np.mean(cagr_dca)),
        "median_cagr_all_in": float(np.median(cagr_all)),
        "median_cagr_dca": float(np.median(cagr_dca)),
    }

    return out

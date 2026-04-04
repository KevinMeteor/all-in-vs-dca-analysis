"""
條件模擬：從歷史上每次出現「高位狀態」的時間點開始，模擬未來幾年（例如 10 年）的報酬分布，來比較一次投入（all-in）和定期定額（DCA）的表現。
高位狀態的定義可以有幾種方式，例如：
- 距離 N 天高點在 -1% 以內（near high）
- 距離 N 天高點在前 10% 分位（pct high）
- 只考慮從未來 N 天高點突破（breakout only）的情況
模擬方法可以是：
- 直接從歷史的 log returns 中 bootstrap（有放回抽樣），保持原始的分布特性
- 或者用 GBM 模型（幾何布朗運動）來模擬，這樣可以更靈活地調整波動率和漂移率，但可能失去一些歷史的分布特性
模擬結果可以計算：
- 終值倍數（terminal multiple）和年化報酬率（CAGR）的分布
- 定期定額勝過一次投入的機率（P(DCA > All-in)）
- 期望年化報酬率（Expected CAGR）等指標


"""


import numpy as np
import pandas as pd
from typing import Literal, Optional, Dict, Any


TDY = 252


def compute_state_features(price: pd.Series, window_td: int = 252) -> pd.DataFrame:
    s = price.astype(float).dropna().sort_index()
    roll_high = s.rolling(window_td, min_periods=window_td).max()
    roll_low = s.rolling(window_td, min_periods=window_td).min()

    dist_to_high = s / roll_high - 1.0
    dist_to_low = s / roll_low - 1.0

    # rolling percentile (rank within window)
    # 用 rank(pct=True) 會比較慢；這裡用近似：把 window 內排序再定位，簡單做法是 rolling apply
    pct = s.rolling(window_td, min_periods=window_td).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1],
        raw=False
    )

    out = pd.DataFrame({
        "price": s,
        "roll_high": roll_high,
        "roll_low": roll_low,
        "dist_to_high": dist_to_high,
        "dist_to_low": dist_to_low,
        "pct_in_window": pct,
    }).dropna()
    return out


def pick_high_state_indices(
    feats: pd.DataFrame,
    mode: Literal["near_high", "pct_high"] = "near_high",
    near_high_threshold: float = -0.01,     # >= -1% 視為貼近新高
    pct_threshold: float = 0.90,            # >= 90% 分位視為高位
    breakout_only: bool = True
) -> np.ndarray:
    if mode == "near_high":
        flag = feats["dist_to_high"] >= near_high_threshold
    elif mode == "pct_high":
        flag = feats["pct_in_window"] >= pct_threshold
    else:
        raise ValueError("mode must be 'near_high' or 'pct_high'")

    flag = flag.fillna(False)

    if breakout_only:
        # 避免 shift(1) 為了塞入第一個缺值，讓布林序列暫時脫離了 bool dtype，使 fillna(False) 跳出 warning
        flag = flag.astype("boolean")
        prev = flag.shift(1).fillna(False).astype(bool)
        flag = flag & (~prev)

    return np.flatnonzero(flag.values)


def simulate_conditional_paths(
    logrets: np.ndarray,
    start_pos_candidates: np.ndarray,
    n_days: int,
    n_sims: int,
    first_block_days: int = 252,
    seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # start_pos_candidates 是對 feats 的位置；但 logrets 對 price 的 diff
    # 我們用同樣位置對齊，保守做法：要求 start + first_block_days <= len(logrets)
    valid = start_pos_candidates[start_pos_candidates +
                                 first_block_days <= len(logrets)]
    if len(valid) == 0:
        raise ValueError(
            "No valid high-state starts with enough forward days.")

    chosen = rng.choice(valid, size=n_sims, replace=True)

    block = np.empty((n_sims, first_block_days))
    for i, s0 in enumerate(chosen):
        block[i, :] = logrets[s0: s0 + first_block_days]

    tail_days = n_days - first_block_days
    idx = rng.integers(0, len(logrets), size=(n_sims, tail_days))
    tail = logrets[idx]

    return np.concatenate([block, tail], axis=1)


def terminal_multiple_all_in(logr_paths: np.ndarray) -> np.ndarray:
    return np.exp(logr_paths.sum(axis=1))


def terminal_multiple_dca(logr_paths: np.ndarray, months: int = 3, initial: float = 1.0) -> np.ndarray:
    n_sims, n_days = logr_paths.shape
    price_rel = np.exp(np.cumsum(logr_paths, axis=1))

    gap = TDY // months
    invest_idx = np.array([min(i * gap, n_days - 1)
                          for i in range(months)], dtype=int)

    cash_per = initial / months
    shares = np.zeros(n_sims)
    for t in invest_idx:
        shares += cash_per / price_rel[:, t]

    terminal = shares * price_rel[:, -1]
    return terminal / initial


def cagr(mult: np.ndarray, years: float) -> np.ndarray:
    return np.power(mult, 1.0 / years) - 1.0


def mc_condition_on_high_state(
    price: pd.Series,
    horizon_years: int,
    n_sims: int = 100_000,
    window_td: int = 252,
    state_mode: Literal["near_high", "pct_high"] = "near_high",
    near_high_threshold: float = -0.01,
    pct_threshold: float = 0.90,
    breakout_only: bool = True,
    first_block_days: int = 252,
    dca_months: int = 3,
    seed: int = 42
) -> Dict[str, Any]:

    n_sims = int(n_sims)

    feats = compute_state_features(price, window_td=window_td)

    # 用 feats 的價格序列計算 log returns（與 feats 同步）
    s = feats["price"]
    logrets = np.log(s).diff().dropna().values

    start_candidates = pick_high_state_indices(
        feats, mode=state_mode,
        near_high_threshold=near_high_threshold,
        pct_threshold=pct_threshold,
        breakout_only=breakout_only
    )

    n_days = int(horizon_years * TDY)
    if first_block_days >= n_days:
        raise ValueError("first_block_days must be < n_days")

    paths = simulate_conditional_paths(
        logrets, start_candidates, n_days=n_days, n_sims=n_sims,
        first_block_days=first_block_days, seed=seed
    )

    mult_all = terminal_multiple_all_in(paths)
    mult_dca = terminal_multiple_dca(paths, months=dca_months)

    cagr_all = cagr(mult_all, float(horizon_years))
    cagr_dca = cagr(mult_dca, float(horizon_years))

    return {
        "horizon_years": horizon_years,
        "n_sims": n_sims,
        "state_mode": state_mode,
        "window_td": window_td,
        "near_high_threshold": near_high_threshold,
        "pct_threshold": pct_threshold,
        "breakout_only": breakout_only,
        "first_block_days": first_block_days,
        "dca_months": dca_months,

        "p_dca_beats_all_in_terminal": float(np.mean(mult_dca > mult_all)),
        "p_dca_beats_all_in_cagr": float(np.mean(cagr_dca > cagr_all)),
        "expected_cagr_all_in": float(np.mean(cagr_all)),
        "expected_cagr_dca": float(np.mean(cagr_dca)),
        "cagr_all_in": cagr_all,
        "cagr_dca": cagr_dca,
        "terminal_all_in": mult_all,
        "terminal_dca": mult_dca,
        "n_candidates": int(len(start_candidates)),
    }

"""
Jan Stats
分析每年1月第N個交易日的股價，檢驗當年是否出現過更低的價格，以及第一次跌破基準價的時間點。
分析邏輯：
1. 以每年1月第N個交易日的收盤價作為基準價。
2. 檢查當年是否有任何一天的收盤價低於基準價（版本A：全年口徑）。
3. 檢查基準日之後是否有任何一天的收盤價低於基準價（版本B：買進後口徑）。
4. 如果有跌破，記錄第一次跌破的日期和距離基準日的交易日數。
5. 計算當年度的最低價和相對基準價的跌幅。
最後統計每年出現更低價的機率，以及第一次跌破的時間分布。


"""


import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.optimize import newton
import numpy as np
# import numpy_financial as npf
from scipy.optimize import brentq


# ============================
# 1. 參數設定
# ============================
TICKER = "0050.TW"          # 元大台灣50
# TICKER = "006208.TW"        # 富邦台50
START_DATE = "2003-01-01"   # 0050 成立後
END_DATE = datetime.today().strftime("%Y-%m-%d")
# MONTHLY_INVEST = 1000     # 每月固定投入金額
# SUBSIDY_RATE = 0.3
# INCOME_TAX_RATE = 0.05

# ============================
# 2. 抓取歷史股價
# ============================
df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)

# # %%
# # 這邊的 'Close' 已經是經過調整(配息、分割)後價格
# df = df[['Close']]

""""
Output : 
Price           Close
Ticker        0050.TW
Date                 
2020-01-02  19.940140
2020-01-03  19.940140
2020-01-06  19.684891
2020-01-07  19.623632
"""


# # ✅ 修正：yfinance 有時會回傳 MultiIndex 欄位（Price/Ticker）
# # 以下把欄位拍扁成單層
if isinstance(df.columns, pd.MultiIndex):
    # 取出 Close 那個欄位（會是一個 DataFrame，欄名可能含 ticker）
    close_df = df.xs('Close', axis=1, level=0)
    # 如果 close_df 還有多欄（例如欄名是 0050.TW），就取第一欄
    if isinstance(close_df, pd.DataFrame):
        close_s = close_df.iloc[:, 0]
    else:
        close_s = close_df
    df = close_s.to_frame(name='Close')
else:
    df = df[['Close']]


"""
Output : 
                Close
Date                 
2020-01-02  19.940140
2020-01-03  19.940140
2020-01-06  19.684893
2020-01-07  19.623631
"""

# 未必面後續擴充到全日期因 index 不一樣導致資料變成 missing
# 所以先將原本 yfinance 下載下來的資料 index 統一換成"無時區"版本
# Series.dt.normalize(): The time component of the date-time is converted to midnight i.e. 00:00:00.

df.index = pd.to_datetime(df.index).normalize()


# %%
# ============================
# 3. 建立完整日期表，並定義交易日欄位
# ============================
# full_calendar = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
# 用「實際資料範圍」建立日曆（不要用 START_DATE）

full_calendar = pd.date_range(
    start=df.index.min(),  # 避免對應不到資料
    end=df.index.max(),
    freq='D'
)


# %%
# 在 reindex 之前：記下原始交易日（yfinance 回來的那些日期）
trade_days = set(df.index)

# *** 重新對齊到所有日期(包含交易日&非交易日)，缺值用前一個最近開盤日股價差補 ***
df = df.reindex(full_calendar)

# ✅ 重要：在補值前，先標記「原本有資料的日期」= 交易日
# 建立一欄位，紀錄是否當日為股市交易日
# df['is_trade_day'] = df['Close'].notna()
df['is_trade_day'] = df.index.isin(trade_days)


# %%
# ============================
# 4. 資料差補規則
# ============================
# 1. 中間 & 結尾：用前一個最近交易日價格
df['Close'] = df['Close'].ffill()  # ffill(): forward fill

# 2. 開頭：用第一筆可觀測價格
df['Close'] = df['Close'].bfill()


# %%
# 補上日期欄位
df['date'] = df.index
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day


# %%
base_month = 1
nth_trade_date = 1

results = []

df_trade_date = df[df['is_trade_day'] == True].copy()

for year, data in df_trade_date.groupby('year'):
    # 只取當年 1 月「交易日」
    jan_trade = data[(data['month'] == base_month) &
                     (data['is_trade_day'])].sort_index()

    if len(jan_trade) < nth_trade_date:
        continue

    # 基準日：1月第 N 個交易日
    base_row = jan_trade.iloc[nth_trade_date - 1]
    base_dt = base_row.name
    base_price = float(base_row['Close'])

    # 當年度交易日
    y_trade = data[data['is_trade_day']].sort_index()

    # ✅ 版本A：全年是否出現更低（包含基準日前後）
    lower_all = y_trade[y_trade['Close'] < base_price]
    lower_count_all = int(lower_all['Close'].notna().sum())
    has_lower_all = lower_count_all > 0

    # ✅ 版本B：基準日「之後」是否出現更低（更符合買進後檢驗）
    y_trade_after = y_trade.loc[y_trade.index > base_dt]
    lower_after = y_trade_after[y_trade_after['Close'] < base_price]
    lower_count_after = int(len(lower_after))
    has_lower_after = lower_count_after > 0

    # ==== 第一次跌破 ====
    if has_lower_after:
        first_lower_dt = lower_after.index[0]
        days_to_first_lower = y_trade_after.index.get_loc(first_lower_dt) + 1
    else:
        first_lower_dt = pd.NaT
        days_to_first_lower = np.nan

    # 當年度最低價 & 跌幅（用交易日）
    min_price = float(y_trade['Close'].min())
    drawdown = (min_price - base_price) / base_price * 100

    results.append({
        'Year': int(year),
        'Base_Date': pd.to_datetime(base_dt).date(),
        'Base_Price': base_price,

        # A：全年口徑
        'Lower_Days_Count_AllYear': lower_count_all,
        'Has_Lower_Price_AllYear': has_lower_all,

        # B：買進後口徑
        'Lower_Days_Count_AfterBase': lower_count_after,
        'Has_Lower_Price_AfterBase': has_lower_after,

        # ==== 第一次跌破 ====
        'First_Lower_Date_AfterBase': first_lower_dt,
        'Days_To_First_Lower': days_to_first_lower,

        'Min_Close_In_Year': min_price,
        'Min_Drawdown_vs_Base': drawdown
    })

results_df = pd.DataFrame(results).sort_values('Year').reset_index(drop=True)

# ==========================================================
# 機率計算
# ==========================================================
total_years = len(results_df)

prob_all = results_df['Has_Lower_Price_AllYear'].mean(
) if total_years else np.nan
prob_after = results_df['Has_Lower_Price_AfterBase'].mean(
) if total_years else np.nan

print("========== 統計結果 ==========")
print(results_df)
print("\n總年份數:", total_years)
print("全年口徑：至少有一天更低價機率:", round(prob_all * 100, 2), "%")
print("買進後口徑：至少有一天更低價機率:", round(prob_after * 100, 2), "%")
# print(df.tail(50))


# %%
print(results_df['Days_To_First_Lower'].describe())

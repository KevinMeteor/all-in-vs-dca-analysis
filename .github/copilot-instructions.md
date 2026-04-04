# Copilot Instructions for All-In vs DCA Backtesting Framework

## Project Overview
A quantitative backtesting framework comparing two Taiwan stock investment strategies on the 0050.TW (TW50 ETF):
- **All-In**: Buy entire position on first trading day
- **DCA (Dollar-Cost Averaging)**: Spread purchases across multiple dates with fixed intervals

Annual performance comparison using January-entry backtests (2003-present).

## Architecture & Data Flow

### Core Components
1. **`src/data/load_data.py`** - Data pipeline
   - Fetches from yfinance with local CSV caching (`data_cache_*.csv`)
   - Critical: Handles yfinance's MultiIndex DataFrame quirk (see flatten logic at line 49-60)
   - Returns DataFrame with columns: `Close`, `is_trade_day`, `year`, `month`, `day`
   - **Key workflow**: Full calendar reindex → forward/backward fill non-trading days → add time features

2. **`src/backtest/engine.py`** - Strategy implementations
   - `compute_metrics()`: Returns dict with `Total_Return` and `Max_Drawdown`
   - `backtest_all_in()`: Calculates NAV for single entry at position start
   - `backtest_dca()`: Spreads investment across `gap_days` intervals with `n_splits` batches
   - Both accept price_series (pd.Series) and return metric dictionaries

3. **`src/backtest/main.py`** - Example orchestration
   - Groups data by year → filters January trading days → runs both strategies → collects results

4. **`src/analysis/jan_stats.py`** - Analysis code (mirrors load_data.py patterns)

## Critical Patterns & Conventions

### Data Processing Pipeline
```python
# Standard workflow in load_data.py and jan_stats.py:
1. Download/cache from yfinance
2. Flatten MultiIndex columns if present (df.xs('Close', axis=1, level=0))
3. Normalize datetime index (remove timezone/time component)
4. Reindex to full_calendar [df.index.min() : df.index.max()]
5. Mark is_trade_day before filling (trade_days = set(df.index))
6. Forward/backward fill Close prices (ffill → bfill)
7. Add time features (year, month, day)
```

### Backtest Input/Output Contract
- **Input**: `price_series` (pd.Series, sorted by date) - represents post-entry daily prices
- **Output**: Dict with `Total_Return` (float) and `Max_Drawdown` (float)
- **N.B.**: DCA `gap_days` is index-based (not calendar days) - e.g., `gap_days=20` means every 20th element in price_series

### Strategy Definition
- Strategies operate on a **single year's January-onward price series** starting after the *first January trading day*
- Use `df.groupby('year')` then filter `month >= 1` for each year
- Minimum viable series: 30+ trading days (checked in main.py)

## Key Dependencies
- **yfinance**: Data source (handles Chinese tickers like "0050.TW")
- **pandas**: Data manipulation (MultiIndex handling, reindex operations)
- **numpy/scipy**: Calculations (optimization with newton/brentq in jan_stats.py)
- **matplotlib**: Visualization (imported but not prominently used yet)
- **Note**: No requirements.txt exists - infer from imports

## Common Pitfalls & Solutions
| Issue | Solution |
|-------|----------|
| yfinance returns MultiIndex columns | Always check `isinstance(df.columns, pd.MultiIndex)` and flatten before processing |
| Missing Close prices for non-trading days | Use `.ffill()` then `.bfill()` after reindex to full_calendar |
| Date timezone inconsistency | Call `pd.to_datetime(df.index).normalize()` to remove tz/time info |
| DCA gap_days confusion | Interpret as "every nth element" not "every n calendar days" |
| Stale cached data | `load_price_data(refresh=True)` to force yfinance download |

## Extending the Codebase
- **New strategies**: Add function to `src/backtest/engine.py` returning metric dict; use in `main.py` loop
- **New analyses**: Follow `jan_stats.py` pattern - import load_price_data, group by year
- **Data changes**: Modify `TICKER`, `START_DATE` in load_data.py; cached files are named by ticker + date range
- **New metrics**: Add to `compute_metrics()` dict (e.g., Sharpe ratio) - current metrics are minimal (Return, MaxDD only)

## File Organization
```
src/
  data/         → Data loading & preprocessing (source of truth for pandas pattern)
  backtest/     → Strategy logic & backtesting engine
  analysis/     → Analysis scripts (can duplicate load_data patterns)
  strategies/   → Empty; reserved for future modular strategy classes
  viz/          → Empty; reserved for visualization utilities
tests/          → Empty; testing infrastructure not yet established
notebooks/      → Empty; reserves for interactive exploration
reports/        → Output destination for results & figures
```

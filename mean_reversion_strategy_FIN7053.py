"""
Mean-Reversion Swing Strategy on the Magnificent Seven

Implements:
- Bollinger Bands (20,2) mean-reversion entry
- ATR(14) trailing stop-loss
- 20-day realized volatility overlay
- 30% per-stock concentration cap
- 10% drawdown kill-switch (flatten for N days)
- Transaction cost and capacity modeling
- Robustness checks: cost sensitivity, parameter heatmap, 12m/3m walk-forward

Data Source: Alpaca Data API
Author: Tanesh Singhal
FIN 7053 Final Project
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pandas.tseries.offsets import DateOffset

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

# CONFIGURATION
API_KEY    = "PKTDIG1HA8UFMFZN9RBO"
API_SECRET = "iPxj1NthenscfCi4UnjSSGNWENH37fS7IkfVtbgy"
# Universe: Magnificent Seven stocks
SYMBOLS    = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"]
START_DATE = "2021-01-01"
END_DATE   = datetime.today().strftime("%Y-%m-%d")
TIMEFRAME  = TimeFrame.Day

# Default parameters: can be overwritten
default_params = {
    'bb_period':      20,     # Bollinger lookback window
    'bb_std':         2,      # Number of standard deviations
    'atr_period':     14,     # ATR lookback window
    'vol_period':     20,     # Realized volatility lookback
    'vol_threshold':  0.30,   # Max allowable vol to trade
    'cap_limit':      0.30,   # Max weight per stock
    'kill_window':    10,     # Days flat after drawdown breach
    'kill_threshold': 0.10,   # Drawdown % to trigger kill-switch
    'cost_per_side':  0.0002  # Transaction cost per side (2 bps)
}

# Initialize Alpaca client
dc = StockHistoricalDataClient(API_KEY, API_SECRET)


def fetch_ohlcv(symbols, start, end):
    """
    Fetch OHLCV data for given symbols and date range.
    Returns high, low, close, volume DataFrames.
    """
    highs, lows, closes, vols = [], [], [], []
    for sym in symbols:
        req = StockBarsRequest(
            symbol_or_symbols=[sym], timeframe=TIMEFRAME,
            start=pd.to_datetime(start), end=pd.to_datetime(end),
            adjustment='raw', feed=DataFeed.IEX
        )
        df = dc.get_stock_bars(req).df.reset_index().set_index('timestamp')
        highs.append(df['high'].rename(sym))
        lows.append(df['low'].rename(sym))
        closes.append(df['close'].rename(sym))
        vols.append(df['volume'].rename(sym))

    # Combine into aligned DataFrames
    high   = pd.concat(highs, axis=1).ffill()
    low    = pd.concat(lows,  axis=1).ffill()
    prices = pd.concat(closes,axis=1).ffill()
    volume = pd.concat(vols,  axis=1).ffill()

    # Drop timezone info for simplicity
    for df in (high, low, prices, volume):
        df.index = df.index.tz_convert(None)
    return high, low, prices, volume


# Load market data
high, low, prices, volume = fetch_ohlcv(SYMBOLS, START_DATE, END_DATE)

# Capacity estimate: 1% of ADV per stock in USD
adv_shares    = volume.mean()
avg_price     = prices.mean()
capacity_1pct = adv_shares * avg_price * 0.01
print("Approx capacity at 1% ADV per stock (USD):")
print(capacity_1pct)


def run_strategy(params):
    """
    Executes the full backtest with given parameters.
    Returns:
      strat    : daily strategy returns
      cum_eq   : cumulative equity series
      ann_ret  : annualized return
      ann_vol  : annualized volatility
      shrp     : Sharpe ratio
      max_dd   : maximum drawdown
    """
    # Unpack parameters
    bb_p   = params['bb_period']
    bb_s   = params['bb_std']
    atr_p  = params['atr_period']
    vol_p  = params['vol_period']
    vol_t  = params['vol_threshold']
    cap_l  = params['cap_limit']
    kill_w = params['kill_window']
    kill_t = params['kill_threshold']
    cost   = params['cost_per_side']

    # Compute indicators
    sma      = prices.rolling(bb_p).mean()                 # Bollinger center
    std_dev  = prices.rolling(bb_p).std()                  # Bollinger width
    lower_bb = sma - bb_s * std_dev                        # Lower band

    prev = prices.shift(1)
    tr1  = high - low
    tr2  = (high - prev).abs()
    tr3  = (low  - prev).abs()
    tr   = pd.DataFrame(np.maximum.reduce([tr1.values, tr2.values, tr3.values]),
                        index=prices.index, columns=SYMBOLS)
    atr  = tr.rolling(atr_p).mean()                       # ATR

    real_vol = prices.pct_change().rolling(vol_p).std()    # Realized vol

    # Generate raw position signals
    pos_df = pd.DataFrame(0, index=prices.index, columns=SYMBOLS)
    for sym in SYMBOLS:
        pos = 0
        stop = np.nan
        for i in range(len(prices)):
            p = prices[sym].iloc[i]
            v = real_vol[sym].iloc[i]
            # Entry if below lower band and vol filter
            if pos==0 and p < lower_bb[sym].iloc[i] and v < vol_t:
                pos, stop = 1, p - atr[sym].iloc[i]
            # Stop-loss breach
            elif pos==1 and p <= stop:
                pos, stop = 0, np.nan
            # Exit if back to mean
            elif pos==1 and p >= sma[sym].iloc[i]:
                pos, stop = 0, np.nan
            # Update trailing stop
            elif pos==1:
                stop = max(stop, p - atr[sym].iloc[i])
            pos_df.iloc[i, pos_df.columns.get_loc(sym)] = pos

    # Shift signals for next-day execution
    daily_pos = pos_df.shift(1).fillna(0)

    # Apply concentration cap and redistribute excess
    raw_w      = daily_pos.div(daily_pos.sum(axis=1), axis=0).fillna(0)
    capped     = raw_w.clip(upper=cap_l)
    excess     = raw_w.sum(axis=1) - capped.sum(axis=1)
    mask_uncap = capped < cap_l
    scale      = mask_uncap.div(mask_uncap.sum(axis=1), axis=0).fillna(0)
    weights    = capped.add(scale.mul(excess, axis=0))

    # Calculate daily returns net of transaction cost
    dret   = prices.pct_change().fillna(0)
    turn0  = weights.diff().abs().sum(axis=1)
    strat0 = (weights.shift(1) * dret).sum(axis=1) - turn0 * cost
    cum0   = (1 + strat0).cumprod()
    draw0  = (cum0.cummax() - cum0) / cum0.cummax()

    # Kill-switch: flat for kill_w days after drawdown breach
    w_final = weights.copy()
    breaches = draw0[draw0 > kill_t].index
    for day in breaches:
        end = day + DateOffset(days=kill_w)
        w_final.loc[day:end, :] = 0

    # Final returns
    turn  = w_final.diff().abs().sum(axis=1)
    strat = (w_final.shift(1) * dret).sum(axis=1) - turn * cost

    # Performance metrics
    cum_eq  = (1 + strat).cumprod()
    ann_ret = cum_eq.iloc[-1]**(252/len(cum_eq)) - 1
    ann_vol = strat.std() * np.sqrt(252)
    shrp    = ann_ret / ann_vol if ann_vol else np.nan
    draw    = (cum_eq.cummax() - cum_eq) / cum_eq.cummax()
    max_dd  = float(draw.max())

    return strat, cum_eq, ann_ret, ann_vol, shrp, max_dd


# Execute backtest
strat, cum_eq, ann_ret, ann_vol, sharpe, max_dd = run_strategy(default_params)
print(f"\nStrategy Performance: Ret={ann_ret:.2%}, Vol={ann_vol:.2%}, Sharpe={sharpe:.2f}, MaxDD={max_dd:.2%}")

# Plot cumulative equity curve
plt.figure(figsize=(10,6))
plt.plot(cum_eq, label='Equity Curve')
plt.title('Full Strategy Equity Curve')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Transaction-cost sensitivity
print("\nCost Sensitivity:")
for bps in [2,5,10,20]:
    p = default_params.copy()
    p['cost_per_side'] = bps/1e4
    _, _, r2, v2, s2, d2 = run_strategy(p)
    print(f"{bps}bps: Ret={r2:.2%}, Vol={v2:.2%}, Sharpe={s2:.2f}, MaxDD={d2:.2%}")

# Parameter-sensitivity heatmap
bb_vals  = [10,20,30]
atr_vals = [7,14,21]
heat     = pd.DataFrame(index=atr_vals, columns=bb_vals, dtype=float)
for bb in bb_vals:
    for ap in atr_vals:
        p = default_params.copy(); p['bb_period']=bb; p['atr_period']=ap
        _, _, _, _, sh, _ = run_strategy(p)
        heat.loc[ap, bb] = sh
print("\nSharpe Heatmap (ATR vs BB):")
print(heat)
plt.figure(figsize=(6,4))
plt.imshow(heat, aspect='auto', origin='lower')
plt.colorbar(label='Sharpe')
plt.xticks(range(len(bb_vals)), bb_vals)
plt.yticks(range(len(atr_vals)), atr_vals)
plt.xlabel('BB Period')
plt.ylabel('ATR Period')
plt.title('Sharpe Heatmap')
plt.tight_layout()
plt.show()

# 12m/3m walk-forward validation
all_oos = []
start = strat.index[0]
while start + DateOffset(months=15) <= strat.index[-1]:
    tr_end = start + DateOffset(months=12)
    oo_end = tr_end + DateOffset(months=3)
    mask = (strat.index > tr_end) & (strat.index <= oo_end)
    segment = strat.loc[mask]
    if len(segment) > 0:
        mret = (1 + segment).resample('ME').prod() - 1
        all_oos.append(mret)
    start = oo_end

all_oos = pd.concat(all_oos)
stats  = all_oos.describe()
pos_pct = (all_oos > 0).mean() * 100
print("\n12m/3m OOS Monthly Return Stats:")
print(stats)
print(f"% positive months: {pos:.2f}%")

plt.figure(figsize=(8,4))
plt.hist(all_oos, bins=int(stats['count']), edgecolor='k')
plt.title('OOS Monthly Return Distribution (12m/3m)')
plt.xlabel('Monthly Return')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

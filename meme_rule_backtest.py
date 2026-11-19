"""Backtest: "buy pressure above X and a volume surge -> did the meme coin go up over the next 15/60 minutes?"

Data: dexscreener_meme_history.csv (the dashboard's saved Dexscreener snapshots: the ~20 coins it showed at
each moment, with last-hour buy pressure, volume and liquidity) plus 5-minute GeckoTerminal candles for every
pool, so coins that dumped and dropped off the list are still measured (no survivorship bias).

Per snapshot:
  buy pressure = (buys - sells) / (buys + sells) over the last hour (Dexscreener transaction counts)
  volume surge = USD volume of the 15 minutes before the snapshot / average 15 minutes of the 45 before that
  entry        = open of the first 5-minute candle starting at or after the snapshot
  15m / 60m    = close of the candle ending 15 / 60 minutes after entry
  collapse     = the lowest low within 60 minutes is 50% or more below entry
One trade per coin per hour: later snapshots of the same coin inside an open trade are skipped.

Costs per round trip (taken off every trade):
  DEX fee        2 x FEE (0.25% each way: PumpSwap / Raydium standard)
  price impact   2 x ((1 + size / (liquidity / 2))^2 - 1), the constant-product pool formula
  bots / fees    2 x EXTRA (0.5% each way: priority fees, slippage and MEV sandwiches)
"""
import numpy as np
import pandas as pd

FEE = 0.0025
EXTRA = 0.005
BAR = pd.Timedelta(minutes=5)
HORIZONS = {'15m': pd.Timedelta(minutes=15), '60m': pd.Timedelta(minutes=60)}
COLLAPSE = -0.50
SURGE_LOOKBACK = pd.Timedelta(minutes=15)
SURGE_BASE = pd.Timedelta(minutes=45)


def round_trip_cost(size_usd, liquidity_usd, fee=FEE, extra=EXTRA):
    """Fraction of the position lost to fees, price impact and bots on the way in and out."""
    pool_usd = np.asarray(liquidity_usd, dtype=float) / 2
    with np.errstate(divide='ignore', invalid='ignore'):
        impact = np.where(pool_usd > 0, (1 + size_usd / pool_usd) ** 2 - 1, np.inf)
    return 2 * fee + 2 * impact + 2 * extra


def candle_features(candles, at):
    """Entry, 15m/60m returns, worst drop in 60m and the volume surge for a snapshot at `at`.

    candles: 5m bars indexed by bar open time (UTC-naive) with open/high/low/close and volume_usd.
    Returns None when the candles do not cover the snapshot and the next 60 minutes.
    """
    start = at.ceil('5min')
    if start not in candles.index:
        return None
    end60 = start + HORIZONS['60m']
    window = candles.loc[start:end60 - BAR]
    if len(window) < int(HORIZONS['60m'] / BAR):
        return None
    entry = float(candles.at[start, 'open'])
    if not entry > 0:
        return None
    out = {'entry': entry, 'worst_60m': float(window['low'].min() / entry - 1)}
    for name, span in HORIZONS.items():
        out[f'ret_{name}'] = float(candles.at[start + span - BAR, 'close'] / entry - 1)
    before = candles.loc[at.floor('5min') - SURGE_LOOKBACK - SURGE_BASE: at.floor('5min') - BAR]
    recent = before.loc[at.floor('5min') - SURGE_LOOKBACK:, 'volume_usd'].sum()
    base = before.loc[:at.floor('5min') - SURGE_LOOKBACK - BAR, 'volume_usd']
    full = len(before) == int((SURGE_LOOKBACK + SURGE_BASE) / BAR)
    base_15 = base.sum() / (len(base) * BAR / SURGE_LOOKBACK) if len(base) else np.nan
    out['surge'] = float(recent / base_15) if full and base_15 > 0 else (np.inf if full and recent > 0 else np.nan)
    return out


def one_trade_per_hour(frame, hold=HORIZONS['60m']):
    """Keep a coin's first qualifying snapshot, then skip that coin until the trade is closed."""
    keep, busy_until = [], {}
    for idx, row in frame.sort_values('Timestamp').iterrows():
        pair = row['Pair Address']
        if pair in busy_until and row['Timestamp'] < busy_until[pair]:
            continue
        busy_until[pair] = row['Timestamp'] + hold
        keep.append(idx)
    return frame.loc[keep]


def evaluate(frame, size_usd, horizon='60m'):
    """Summary of trades (rows already filtered by a rule) after costs for one position size."""
    if frame.empty:
        return {'Trades': 0, 'Coins': 0}
    cost = round_trip_cost(size_usd, frame['Liquidity USD'])
    net = frame[f'ret_{horizon}'] - cost
    return {
        'Trades': int(len(frame)),
        'Coins': int(frame['Pair Address'].nunique()),
        'Up before costs': float((frame[f'ret_{horizon}'] > 0).mean()),
        'Win after costs': float((net > 0).mean()),
        'Avg gross': float(frame[f'ret_{horizon}'].mean()),
        'Median gross': float(frame[f'ret_{horizon}'].median()),
        'Avg cost': float(np.mean(cost)),
        'Avg net': float(net.mean()),
        'Median net': float(net.median()),
        'Collapsed 50%+': float((frame['worst_60m'] <= COLLAPSE).mean()),
        'Best': float(net.max()),
        'Worst': float(net.min()),
    }


def rule_grid(frame, size_usd, pressures=(None, 0.0, 0.1, 0.2, 0.3, 0.5), surges=(None, 1.5, 2.0, 3.0),
              min_liquidity=0.0, horizon='60m'):
    """Every buy-pressure x volume-surge rule, one trade per coin per hour, after costs."""
    rows = []
    base = frame[frame['Liquidity USD'] >= min_liquidity]
    for pressure in pressures:
        for surge in surges:
            picked = base
            if pressure is not None:
                picked = picked[picked['Buy Pressure'] > pressure]
            if surge is not None:
                picked = picked[picked['surge'] >= surge]
            picked = one_trade_per_hour(picked)
            rows.append({'Buy pressure >': 'any' if pressure is None else pressure,
                         'Volume surge ≥': 'any' if surge is None else surge,
                         **evaluate(picked, size_usd, horizon)})
    return pd.DataFrame(rows)

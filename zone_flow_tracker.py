"""Zone Flow Tracker: forward test of the Zone Battle + Highest Flow Z Rise/Drop strategy.

Strategy (the best variant from the 2026-11-14 backtests; NOT proven profitable):
  BUY  when a zone battle ends as a Strong Buyer Victory AND the Highest Flow Z Rise's
       Flow + Momentum + Volatility contribution (its Direction Score) beats the size of the
       Highest Flow Z Drop's.
  SELL when a zone battle ends as a Strong Seller Victory AND the drop's beats the rise's.
  Exit "A": hold until the opposite confirmed signal, then reverse. Fill at the next candle's open.
  Costs 0.05% per side.
The highest rise / drop are taken over the Ghana day so far (as the Historical Z-Score Component
Analysis shows them) on 5m / 15m / 1h, and over the trailing 24 candles on 4h / 1d, where a day
holds too few candles. Only closed candles are used.

Forward tracking: for each symbol and timeframe, only signals AFTER tracking began are logged
(zone_flow_tracker_log.csv), so the record is a genuine out-of-sample test.
"""
import json
import os

import numpy as np
import pandas as pd

from qqq_nq_signals import demand_supply_score, historical_z_components

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, 'zone_flow_tracker_log.csv')
STATE_PATH = os.path.join(HERE, 'zone_flow_tracker_state.json')
TIMEFRAMES = ['5m', '15m', '1h', '4h', '1d']
TF_DELTA = {'5m': pd.Timedelta(minutes=5), '15m': pd.Timedelta(minutes=15), '1h': pd.Timedelta(hours=1),
            '4h': pd.Timedelta(hours=4), '1d': pd.Timedelta(days=1)}
DAY_GROUPED = {'5m', '15m', '1h'}
TRAILING = 24
OBS = 10
FEE = 0.0005
SOURCES = {
    'BTC/USDT': {'label': 'BTC/USDT perpetual (Binance, Derivative tab)', 'market': 'usdm', 'fvg': 3, 'flow': 'proxy'},
    'BTC/USD': {'label': 'BTC/USD (Yahoo, US Indices tab)', 'market': None, 'fvg': 2, 'flow': 'proxy'},
    # Same rules, but every Flow Z (zone battle Fz and the Highest Flow Rise / Drop) is built from Binance
    # taker buy vs sell volume instead of candle colour.
    'BTC/USDT (real flow)': {'label': 'BTC/USDT perpetual, REAL taker flow', 'market': 'usdm-real', 'fvg': 3,
                             'flow': 'real', 'base': 'BTC/USDT'},
}
DEFAULT_SYMBOLS = list(SOURCES)
LOG_COLUMNS = ['Symbol', 'Timeframe', 'Signal Time (UTC)', 'Side', 'Zone Type', 'Zone Bottom', 'Zone Top',
               'Zone Entry (UTC)', 'Battle Score', 'Rise Total', 'Drop Total', 'Signal Price',
               'Fill Time (UTC)', 'Fill Price', 'Recorded (UTC)']
KEY = ['Symbol', 'Timeframe', 'Signal Time (UTC)', 'Side']


def _utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')


def _fmt(ts):
    return _utc(ts).strftime('%Y-%m-%d %H:%M') if ts is not None and pd.notna(ts) else ''


# ----------------------------------------------------------------------------- signals

def real_flow_z(history):
    """The dashboard's Flow Z formula (20-candle flow, z over 50) on Binance taker buy vs sell volume."""
    buy = history['taker_buy'].astype(float)
    sell = history['volume'].astype(float) - buy
    flow = (buy.rolling(20).sum() - sell.rolling(20).sum()) / history['volume'].rolling(20).sum().replace(0, np.nan)
    return (flow - flow.rolling(50).mean()) / flow.rolling(50).std().replace(0, np.nan)


def flow_totals(history, timeframe, flow='proxy'):
    """Rise / drop totals (Flow + Momentum + Volatility contribution, in points) of the highest
    Flow Z rise and drop so far in the group, at every candle. Drop total is returned as a size (>= 0).
    flow='real' uses taker buy vs sell volume for Flow Z instead of candle colour."""
    comps = historical_z_components(history)
    if flow == 'real':
        comps['Flow (z)'] = real_flow_z(history)
    ch = comps.diff()
    flow = ch['Flow (z)']
    total = pd.Series(np.asarray(demand_supply_score(
        flow, ch['Momentum (z)'], ch['Volatility (z)'], history['close'].diff()), dtype=float), index=history.index)
    usable = flow.notna() & flow.ne(0) & total.notna()
    up = flow.where(usable & (flow > 0))
    down = flow.where(usable & (flow < 0))
    out = pd.DataFrame(index=history.index)
    if timeframe in DAY_GROUPED:
        day = pd.DatetimeIndex(history.index).normalize()
        at_max = up.notna() & (up >= up.groupby(day).cummax())
        at_min = down.notna() & (down <= down.groupby(day).cummin())
        out['Rise Total'] = total.where(at_max).groupby(day).ffill()
        out['Drop Total'] = -total.where(at_min).groupby(day).ffill()
    else:
        u, d, t = up.to_numpy(), down.to_numpy(), total.to_numpy()
        rise, drop = np.full(len(t), np.nan), np.full(len(t), np.nan)
        for i in range(len(t)):
            lo = max(0, i - TRAILING + 1)
            wu, wd = u[lo:i + 1], d[lo:i + 1]
            if np.isfinite(wu).any():
                rise[i] = t[lo + int(np.nanargmax(wu))]
            if np.isfinite(wd).any():
                drop[i] = -t[lo + int(np.nanargmin(wd))]
        out['Rise Total'], out['Drop Total'] = rise, drop
    return out


def confirmed_signals(history, timeframe, calc, fvg, now=None, flow='proxy'):
    """Every confirmed BUY / SELL in `history`, with the next candle's open as the fill (if it exists)."""
    cols = LOG_COLUMNS[:-1]
    if history is None or len(history) < 100:
        return pd.DataFrame(columns=cols)
    kwargs = {'flow': 'real'} if flow == 'real' else {}
    res, _ = calc(history, observation_bars=OBS, max_zones=10 ** 9, fvg_candles=fvg, now=now, **kwargs)
    if res.empty:
        return pd.DataFrame(columns=cols)
    res = res[res['Result'] != 'Battle Unresolved']
    totals = flow_totals(history, timeframe, flow)
    rise, drop = totals['Rise Total'].to_numpy(), totals['Drop Total'].to_numpy()
    opens, closes, index = history['open'].to_numpy(), history['close'].to_numpy(), history.index
    rows = []
    for _, z in res.iterrows():
        pos = index.get_loc(z['Verdict Time'])
        r, d = rise[pos], drop[pos]
        if not (np.isfinite(r) and np.isfinite(d)):
            continue
        buyer = z['Result'] == 'Strong Buyer Victory'
        if (buyer and r > d) or (not buyer and d > r):
            has_fill = pos + 1 < len(history)
            rows.append({
                'Symbol': None, 'Timeframe': timeframe,
                'Signal Time (UTC)': _fmt(index[pos] + TF_DELTA[timeframe]),
                'Side': 'BUY' if buyer else 'SELL', 'Zone Type': z['Zone Type'],
                'Zone Bottom': z['Zone Bottom'], 'Zone Top': z['Zone Top'],
                'Zone Entry (UTC)': _fmt(z['Zone Entry']), 'Battle Score': z['Battle Score'],
                'Rise Total': r, 'Drop Total': d, 'Signal Price': closes[pos],
                'Fill Time (UTC)': _fmt(index[pos + 1]) if has_fill else '',
                'Fill Price': opens[pos + 1] if has_fill else np.nan,
            })
    return pd.DataFrame(rows, columns=cols).sort_values('Signal Time (UTC)').reset_index(drop=True)


# ----------------------------------------------------------------------------- trading the signals

def trades_from_signals(signals, last_price=None):
    """Exit A (flip on the opposite confirmed signal). Returns (closed trades, open position or None)."""
    trades, side, entry, entry_time = [], 0, None, None
    for _, s in signals.dropna(subset=['Fill Price']).sort_values('Signal Time (UTC)').iterrows():
        new = 1 if s['Side'] == 'BUY' else -1
        if new == side:
            continue
        if side:
            gross = side * (s['Fill Price'] / entry - 1)
            trades.append({'Side': 'LONG' if side == 1 else 'SHORT', 'Entry Time (UTC)': entry_time,
                           'Entry': entry, 'Exit Time (UTC)': s['Fill Time (UTC)'], 'Exit': s['Fill Price'],
                           'Net %': (gross - 2 * FEE) * 100})
        side, entry, entry_time = new, s['Fill Price'], s['Fill Time (UTC)']
    open_pos = None
    if side:
        unreal = (side * (last_price / entry - 1) - 2 * FEE) * 100 if last_price else np.nan
        open_pos = {'Side': 'LONG' if side == 1 else 'SHORT', 'Entry Time (UTC)': entry_time, 'Entry': entry,
                    'Unrealized %': unreal}
    return pd.DataFrame(trades, columns=['Side', 'Entry Time (UTC)', 'Entry', 'Exit Time (UTC)', 'Exit', 'Net %']), open_pos


def summarize(trades):
    if trades.empty:
        return {'Trades': 0, 'Win %': np.nan, 'Total %': 0.0, 'Max DD %': 0.0}
    eq = np.cumprod(1 + trades['Net %'].to_numpy() / 100)
    return {'Trades': int(len(trades)), 'Win %': float((trades['Net %'] > 0).mean() * 100),
            'Total %': float((eq[-1] - 1) * 100), 'Max DD %': float(((eq / np.maximum.accumulate(eq)) - 1).min() * 100)}


# ----------------------------------------------------------------------------- forward log

def load_state(path=None):
    path = path or STATE_PATH
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def save_state(state, path=None):
    path = path or STATE_PATH
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(state, fh, indent=2)
    os.replace(tmp, path)


def load_log(path=None):
    path = path or LOG_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=LOG_COLUMNS)
    return pd.read_csv(path, dtype={'Signal Time (UTC)': str, 'Fill Time (UTC)': str}, keep_default_na=False,
                       na_values=[''])


def record(symbol, timeframe, signals, last_price, now=None, log_path=None, state_path=None):
    """Log signals made after tracking began for (symbol, timeframe). Returns the number of new signals."""
    now = _utc(now or pd.Timestamp.now(tz='UTC'))
    state = load_state(state_path)
    key = f'{symbol}|{timeframe}'
    entry = state.setdefault(key, {'started': _fmt(now)})
    entry.update({'last_update': _fmt(now), 'last_price': None if last_price is None else float(last_price)})
    fresh = signals[signals['Signal Time (UTC)'] >= entry['started']].copy() if not signals.empty else signals
    added = 0
    if not fresh.empty:
        fresh['Symbol'] = symbol
        fresh['Recorded (UTC)'] = _fmt(now)
        log = load_log(log_path)
        mine = log[(log['Symbol'] == symbol) & (log['Timeframe'] == timeframe)]
        known = set(zip(mine['Signal Time (UTC)'], mine['Side']))
        added = sum((t, s) not in known for t, s in zip(fresh['Signal Time (UTC)'], fresh['Side']))
        if not mine.empty:   # keep the first recording time; fill in a fill price that was pending
            first = dict(zip(zip(mine['Signal Time (UTC)'], mine['Side']), mine['Recorded (UTC)']))
            fresh['Recorded (UTC)'] = [first.get((t, s), r) for t, s, r in
                                       zip(fresh['Signal Time (UTC)'], fresh['Side'], fresh['Recorded (UTC)'])]
        keys = set(zip(fresh['Signal Time (UTC)'], fresh['Side']))
        drop = (log['Symbol'] == symbol) & (log['Timeframe'] == timeframe) & pd.Series(
            [(t, s) in keys for t, s in zip(log['Signal Time (UTC)'], log['Side'])], index=log.index, dtype=bool)
        log = pd.concat([log[~drop], fresh.reindex(columns=LOG_COLUMNS)], ignore_index=True)
        log = log.sort_values(['Symbol', 'Timeframe', 'Signal Time (UTC)'], kind='stable')
        path = log_path or LOG_PATH
        log.to_csv(path + '.tmp', index=False)
        os.replace(path + '.tmp', path)
    entry['signals'] = int(len(load_log(log_path).query('Symbol == @symbol and Timeframe == @timeframe')))
    save_state(state, state_path)
    return added


def real_flow_history(symbol, timeframe, limit=1000, fetch_klines=None):
    """Binance klines with taker volume, plus the dashboard's atr14 / z_score / momentum columns."""
    if fetch_klines is None:
        from real_flow import fetch_taker_klines as fetch_klines
    d = fetch_klines(symbol, timeframe, limit)
    if d is None or d.empty:
        return d
    d = d.copy()
    tr = pd.concat([d.high - d.low, (d.high - d.close.shift()).abs(), (d.low - d.close.shift()).abs()], axis=1).max(axis=1)
    d['atr14'] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d['z_score'] = (d.close - d.close.rolling(20).mean()) / d.close.rolling(20).std()
    d['momentum'] = d.close.pct_change(30)
    return d


def fetch_for(fetch_and_analyze, symbol, timeframe):
    source = SOURCES.get(symbol, {'market': 'usdm'})
    if source['market'] == 'usdm-real':
        return real_flow_history(source['base'], timeframe)
    if source['market'] == 'usdm':
        return fetch_and_analyze(symbol, market='usdm', timeframe=timeframe, silent=True, limit=1000)
    return fetch_and_analyze(symbol, timeframe=timeframe, silent=True)


def update_all(fetch_and_analyze, calc, symbols=DEFAULT_SYMBOLS, timeframes=TIMEFRAMES, now=None,
               log_path=None, state_path=None):
    """Fetch, find confirmed signals, and log the new ones. Returns {(symbol, tf): status}."""
    now = _utc(now or pd.Timestamp.now(tz='UTC'))
    results = {}
    for symbol in symbols:
        source = SOURCES.get(symbol, {'fvg': 3, 'flow': 'proxy'})
        for tf in timeframes:
            try:
                history = fetch_for(fetch_and_analyze, symbol, tf)
                if history is None or len(history) < 100:
                    results[(symbol, tf)] = 'not enough candles'
                    continue
                signals = confirmed_signals(history, tf, calc, source['fvg'], now, source.get('flow', 'proxy'))
                added = record(symbol, tf, signals, float(history['close'].iloc[-1]), now, log_path, state_path)
                results[(symbol, tf)] = f'{added} new signal(s)'
            except Exception as exc:
                results[(symbol, tf)] = f'failed: {exc}'
    return results


def forward_overview(log_path=None, state_path=None, symbols=DEFAULT_SYMBOLS, timeframes=TIMEFRAMES):
    """One row per (symbol, timeframe) with the forward record so far."""
    state, log = load_state(state_path), load_log(log_path)
    rows = []
    for symbol in symbols:
        for tf in timeframes:
            info = state.get(f'{symbol}|{tf}')
            mine = log[(log['Symbol'] == symbol) & (log['Timeframe'] == tf)]
            trades, open_pos = trades_from_signals(mine, (info or {}).get('last_price'))
            stats = summarize(trades)
            rows.append({
                'Symbol': symbol, 'Timeframe': tf,
                'Tracking Since (UTC)': (info or {}).get('started', 'not started'),
                'Signals': int(len(mine)), 'Closed Trades': stats['Trades'],
                'Win %': stats['Win %'], 'Forward Total %': stats['Total %'], 'Max DD %': stats['Max DD %'],
                'Position': open_pos['Side'] if open_pos else 'FLAT',
                'Entry': open_pos['Entry'] if open_pos else np.nan,
                'Entry Time (UTC)': open_pos['Entry Time (UTC)'] if open_pos else '',
                'Unrealized %': open_pos['Unrealized %'] if open_pos else np.nan,
                'Last Price': (info or {}).get('last_price'),
                'Last Update (UTC)': (info or {}).get('last_update', ''),
            })
    return pd.DataFrame(rows)

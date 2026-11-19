"""Session Flow Recorder: the Highest Flow Z Rise and Drop of every trading session.

For BTC/USD (Yahoo, as in the US Indices tab) and the top Binance crypto perpetuals, on 5m candles,
it records each session's Demand & Supply Analysis — Highest Flow Z Rise and Highest Flow Z Drop:
Flow / Momentum / Volatility Z change, price change, the 60/25/15 flow / momentum / volatility
contributions and the Direction Score, using the same formulas as the US Indices tab
(historical_z_components over the full fetched history, then highest_flow_zone).

Sessions follow New York time (so they move with US daylight saving). A trading day starts when
New York closes at 17:00 New York time, which is the start of the Asia session:
    Asia      17:00 -> 03:00 New York   (22:00 -> 08:00 Ghana in winter, 21:00 -> 07:00 in summer)
    London    03:00 -> 08:00
    New York  08:00 -> 17:00
    Day       17:00 -> 17:00            (everything since the Asia open)

Bias: the rise is stronger when its Flow + Momentum contribution beats the drop's (in size).
That is the user's own reading of the two cards. It is recorded, not yet tested.
"""
import os
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from historical_flow_analysis import highest_flow_zone
from qqq_nq_signals import historical_z_components

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION_HISTORY_PATH = os.path.join(HERE, 'session_flow_history.csv')
NY_TZ = 'America/New_York'
DAY_START_HOUR = 17
SESSIONS = (('Asia', 17, 3), ('London', 3, 8), ('New York', 8, 17))
SESSION_ORDER = {'Asia': 0, 'London': 1, 'New York': 2, 'Day': 3}
TIMEFRAME_DELTA = {'1m': pd.Timedelta(minutes=1), '5m': pd.Timedelta(minutes=5),
                   '15m': pd.Timedelta(minutes=15), '1h': pd.Timedelta(hours=1)}
KEY = ['Trading Day', 'Session', 'Symbol']
SIDE_FIELDS = ['Time (UTC)', 'Flow Z Change', 'Momentum Z Change', 'Volatility Z Change', 'Price Change',
               'Flow Contribution', 'Momentum Contribution', 'Volatility Contribution', 'Direction Score',
               'Zone Bottom', 'Zone Top', 'Close']
COLUMNS = (KEY + ['Status', 'Start (UTC)', 'End (UTC)', 'Bars', 'Open', 'Last', 'Change %',
                  'Bias', 'Rise Strength', 'Drop Strength', 'Margin']
           + [f'Rise {field}' for field in SIDE_FIELDS] + [f'Drop {field}' for field in SIDE_FIELDS]
           + ['Timeframe', 'Updated (UTC)'])


def _utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')


def trading_day(ts):
    """The New York trading day a moment belongs to (it starts at 17:00 New York the evening before)."""
    ny = _utc(ts).tz_convert(NY_TZ)
    return (ny + timedelta(hours=24 - DAY_START_HOUR)).date()


def session_windows(day):
    """{session: (start UTC, end UTC)} for one trading day."""
    day = pd.Timestamp(day).date()

    def at(d, hour):
        return pd.Timestamp(datetime.combine(d, time(hour))).tz_localize(NY_TZ).tz_convert('UTC')

    previous = day - timedelta(days=1)
    windows = {
        'Asia': (at(previous, 17), at(day, 3)),
        'London': (at(day, 3), at(day, 8)),
        'New York': (at(day, 8), at(day, 17)),
    }
    windows['Day'] = (windows['Asia'][0], windows['New York'][1])
    return windows


def current_session(now=None):
    now = _utc(now or pd.Timestamp.now(tz='UTC'))
    day = trading_day(now)
    for name, (start, end) in session_windows(day).items():
        if name != 'Day' and start <= now < end:
            return day, name, start, end
    return day, None, None, None


def _prepare(history, timeframe, now):
    data = history.copy()
    data.index = pd.to_datetime(data.index, utc=True)
    data = data[~data.index.duplicated(keep='last')].sort_index()
    return data.loc[data.index + TIMEFRAME_DELTA[timeframe] <= now]  # closed candles only


def session_rows(history, symbol, day, timeframe='5m', now=None):
    """One row per session of `day` (plus 'Day'), from candles that had closed by `now`."""
    now = _utc(now or pd.Timestamp.now(tz='UTC'))
    if history is None or history.empty or not {'open', 'high', 'low', 'close', 'volume',
                                                 'momentum', 'atr14', 'z_score'}.issubset(history.columns):
        return []
    data = _prepare(history, timeframe, now)
    if data.empty:
        return []
    components = historical_z_components(data)   # over the full history, so the first session bar has its warm-up
    changes = components.diff()
    previous_close = data['close'].shift(1)
    events = pd.DataFrame({
        'Time (UTC)': data.index,
        'Flow Z Change': changes['Flow (z)'].to_numpy(),
        'Momentum Z Change': changes['Momentum (z)'].to_numpy(),
        'Volatility Z Change': changes['Volatility (z)'].to_numpy(),
        'Price Change': (data['close'] - previous_close).to_numpy(),
        'Zone Bottom': data['low'].to_numpy(), 'Zone Top': data['high'].to_numpy(),
        'Close': data['close'].to_numpy(),
    }, index=data.index)
    usable = events['Flow Z Change'].notna() & events['Flow Z Change'].ne(0) & events['Price Change'].notna()
    rows, updated = [], now.strftime('%Y-%m-%d %H:%M')
    for name, (start, end) in session_windows(day).items():
        in_window = (data.index >= start) & (data.index < end)
        if not in_window.any():
            continue
        bars = data.loc[in_window]
        window_events = events.loc[in_window & usable.to_numpy()]
        rise = highest_flow_zone(window_events, 'rise')
        drop = highest_flow_zone(window_events, 'drop')
        row = {
            'Trading Day': str(day), 'Session': name, 'Symbol': symbol,
            'Status': 'Complete' if now >= end else 'Live',
            'Start (UTC)': start.strftime('%Y-%m-%d %H:%M'), 'End (UTC)': end.strftime('%Y-%m-%d %H:%M'),
            'Bars': int(len(bars)), 'Open': float(bars['open'].iloc[0]), 'Last': float(bars['close'].iloc[-1]),
            'Timeframe': timeframe, 'Updated (UTC)': updated,
        }
        row['Change %'] = (row['Last'] / row['Open'] - 1) * 100
        for label, event in (('Rise', rise), ('Drop', drop)):
            for field in SIDE_FIELDS:
                value = event[field] if event is not None else np.nan
                if field == 'Time (UTC)' and event is not None:
                    value = _utc(value).strftime('%Y-%m-%d %H:%M')
                row[f'{label} {field}'] = value
        rise_strength = (row['Rise Flow Contribution'] + row['Rise Momentum Contribution']) if rise is not None else np.nan
        drop_strength = -(row['Drop Flow Contribution'] + row['Drop Momentum Contribution']) if drop is not None else np.nan
        row['Rise Strength'], row['Drop Strength'] = rise_strength, drop_strength
        if np.isfinite(rise_strength) and np.isfinite(drop_strength):
            row['Margin'] = rise_strength - drop_strength
            row['Bias'] = 'BUY' if row['Margin'] > 0 else 'SELL' if row['Margin'] < 0 else 'EVEN'
        else:
            row['Margin'], row['Bias'] = np.nan, 'NO DATA'
        rows.append(row)
    return rows


def load_history(path=None):
    path = path or SESSION_HISTORY_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype={'Trading Day': str})


def upsert(rows, path=None):
    """Replace rows with the same (Trading Day, Session, Symbol); keep everything else. Returns the row count written."""
    if not rows:
        return 0
    path = path or SESSION_HISTORY_PATH
    new = pd.DataFrame(rows).reindex(columns=COLUMNS)
    history = load_history(path)
    if not history.empty:
        keys = set(map(tuple, new[KEY].astype(str).to_numpy()))
        keep = ~history[KEY].astype(str).apply(tuple, axis=1).isin(keys)
        history = history.loc[keep]
        new = pd.concat([history.reindex(columns=COLUMNS), new], ignore_index=True)
    new['_o'] = new['Session'].map(SESSION_ORDER)
    new = new.sort_values(['Trading Day', '_o', 'Symbol'], kind='stable').drop(columns='_o')
    tmp = path + '.tmp'
    new.to_csv(tmp, index=False)
    os.replace(tmp, path)
    return len(rows)


def record_sessions(symbols, fetch, timeframe='5m', now=None, path=None):
    """Fetch each symbol and record today's (and yesterday's, to finish it) sessions.

    fetch(symbol) -> candles with the dashboard's factor columns. Returns {symbol: status}.
    """
    now = _utc(now or pd.Timestamp.now(tz='UTC'))
    today = trading_day(now)
    days = [today - timedelta(days=1), today]
    results, rows = {}, []
    for symbol in symbols:
        try:
            history = fetch(symbol)
            symbol_rows = [r for day in days for r in session_rows(history, symbol, day, timeframe, now)]
            rows += symbol_rows
            results[symbol] = f'{len(symbol_rows)} session row(s)' if symbol_rows else 'no candles in session'
        except Exception as exc:
            results[symbol] = f'failed: {exc}'
    upsert(rows, path)
    return results

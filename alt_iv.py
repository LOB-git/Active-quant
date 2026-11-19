"""30-day implied volatility for coins without a DVOL index (SOL, XRP, AVAX, HYPE, TRX).

Source: Deribit's USDC-settled options (the same exchange as DVOL). For each expiry the at-the-money
implied volatility is the average call/put mark IV at the strike nearest that expiry's forward
(Deribit's `underlying_price` for the expiry). The 30-day figure interpolates total variance
(IV² × days) between the expiries either side of 30 days, the way volatility indices do.

Differences from DVOL, shown in the panel:
  * At-the-money only. DVOL also uses out-of-the-money options (the skew), so it usually reads a
    little higher: on 2026-11-19 BTC DVOL was ~45% while this method on BTC gave ~43.5%.
  * Deribit keeps no history of it, so every reading is saved to iv_snapshots.csv. The date picker
    can only show these coins from the day snapshots started (the alert runner saves one every hour).

BNB and DOGE are not included: their only listed options (Binance) run out to 8 days, too short and
too thin for a 30-day figure.
"""
import os

import numpy as np
import pandas as pd
import requests

ALT_COINS = ('SOL', 'XRP', 'AVAX', 'HYPE', 'TRX')
TARGET_DAYS = 30
MIN_DAYS = 2            # expiries closer than this are too noisy (same/next-day options)
SNAPSHOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'iv_snapshots.csv')
SNAPSHOT_COLUMNS = ['Time (UTC)', 'Coin', 'IV 30d', 'Near Expiry Days', 'Far Expiry Days', 'Source']
MIDNIGHT_WINDOW = pd.Timedelta(minutes=90)   # a snapshot this close to 00:00 UTC counts as "IV at 00:00"
LATE_DAY = pd.Timedelta(hours=22)            # a day's "close" needs a snapshot after 22:00 UTC
DERIBIT = 'https://www.deribit.com/api/v2/public/'


def option_chain(coin, get=requests.get):
    """Deribit book summary of every option on COIN_USDC: expiry, strike, side, mark_iv, forward."""
    response = get(DERIBIT + 'get_book_summary_by_currency', params={'currency': 'USDC', 'kind': 'option'},
                   timeout=30)
    response.raise_for_status()
    rows = pd.DataFrame(response.json().get('result') or [])
    if rows.empty:
        return rows
    rows = rows[rows['instrument_name'].str.startswith(f'{coin}_USDC-')].copy()
    if rows.empty:
        return rows
    parts = rows['instrument_name'].str.split('-', expand=True)
    rows['expiry'] = pd.to_datetime(parts[1], format='%d%b%y', utc=True) + pd.Timedelta(hours=8)  # 08:00 UTC
    rows['strike'] = parts[2].str.replace('d', '.', regex=False).astype(float)
    rows['side'] = parts[3]
    return rows[['instrument_name', 'expiry', 'strike', 'side', 'mark_iv', 'underlying_price', 'open_interest']]


def atm_term(chain, now=None):
    """One row per expiry at least MIN_DAYS away: days to expiry and at-the-money IV (fraction)."""
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    rows = []
    for expiry, group in chain.groupby('expiry'):
        days = (expiry - now).total_seconds() / 86_400
        valid = group[group['mark_iv'] > 0]
        if days < MIN_DAYS or valid.empty:
            continue
        forward = float(valid['underlying_price'].median())
        strike = valid.loc[(valid['strike'] - forward).abs().idxmin(), 'strike']
        atm = valid[valid['strike'] == strike]
        rows.append({'expiry': expiry, 'days': days, 'strike': float(strike), 'forward': forward,
                     'iv': float(atm['mark_iv'].mean()) / 100})
    return pd.DataFrame(rows).sort_values('days').reset_index(drop=True) if rows else pd.DataFrame()


def iv_at_days(term, target=TARGET_DAYS):
    """IV for `target` days by linear interpolation of total variance between the bracketing expiries.

    Returns (iv, near_days, far_days); outside the listed expiries the nearest one is used. None if empty.
    """
    if term is None or term.empty:
        return None
    below, above = term[term['days'] <= target], term[term['days'] >= target]
    if below.empty or above.empty:
        row = term.iloc[(term['days'] - target).abs().argmin()]
        return float(row['iv']), float(row['days']), float(row['days'])
    near, far = below.iloc[-1], above.iloc[0]
    if far['days'] == near['days']:
        return float(near['iv']), float(near['days']), float(far['days'])
    w_near, w_far = near['iv'] ** 2 * near['days'], far['iv'] ** 2 * far['days']
    w = w_near + (w_far - w_near) * (target - near['days']) / (far['days'] - near['days'])
    return float(np.sqrt(w / target)), float(near['days']), float(far['days'])


def live_iv(coin, get=requests.get, now=None):
    """{'iv', 'near_days', 'far_days', 'time', 'term'} for a coin right now, or None."""
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    term = atm_term(option_chain(coin, get), now)
    found = iv_at_days(term)
    if found is None:
        return None
    iv, near, far = found
    return {'iv': iv, 'near_days': near, 'far_days': far, 'time': now, 'term': term}


# ---------- snapshots (the only history these coins have) ----------

def record_snapshot(coin, reading, path=None, min_gap=pd.Timedelta(minutes=10)):
    """Append a live reading to the snapshot file unless one for this coin was saved within `min_gap`."""
    path = path or SNAPSHOT_PATH
    if not reading:
        return False
    stamp = pd.Timestamp(reading['time']).tz_convert('UTC')
    stored = load_snapshots(path)
    mine = stored[stored['Coin'] == coin]
    if len(mine) and (stamp - mine['Time (UTC)'].max()).total_seconds() < min_gap.total_seconds():
        return False
    row = pd.DataFrame([{
        'Time (UTC)': stamp.isoformat(), 'Coin': coin, 'IV 30d': round(reading['iv'], 6),
        'Near Expiry Days': round(reading['near_days'], 3), 'Far Expiry Days': round(reading['far_days'], 3),
        'Source': 'Deribit USDC options, 30-day ATM',
    }], columns=SNAPSHOT_COLUMNS)
    row.to_csv(path, mode='a', header=not os.path.exists(path) or os.path.getsize(path) == 0, index=False)
    return True


def load_snapshots(path=None):
    path = path or SNAPSHOT_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    frame['Time (UTC)'] = pd.to_datetime(frame['Time (UTC)'], utc=True, format='ISO8601')
    return frame.reindex(columns=SNAPSHOT_COLUMNS).sort_values('Time (UTC)').reset_index(drop=True)


def snapshot_series(snapshots, coin):
    """Daily implied volatility from snapshots, shaped like fetch_dvol's output.

    Value per UTC day = its last snapshot, kept only when taken after 22:00 (a real end-of-day reading).
    attrs['open'][day] = the snapshot nearest 00:00 UTC within 90 minutes (IV at the start of the day).
    """
    mine = snapshots[snapshots['Coin'] == coin].dropna(subset=['IV 30d'])
    if mine.empty:
        series = pd.Series(dtype=float, name='dvol', index=pd.DatetimeIndex([], tz='UTC'))
        series.attrs['open'] = {}
        return series
    times, values = mine['Time (UTC)'], mine['IV 30d'].astype(float)
    days = times.dt.floor('D')
    last = mine.assign(day=days).groupby('day').tail(1)
    closes = last[last['Time (UTC)'] - last['day'] >= LATE_DAY]
    series = pd.Series(closes['IV 30d'].astype(float).to_numpy(), index=pd.DatetimeIndex(closes['day']), name='dvol')
    opens = {}
    for day in pd.DatetimeIndex(days.unique()).union(pd.DatetimeIndex(days.unique()) + pd.Timedelta(days=1)):
        gap = (times - day).abs()
        if len(gap) and gap.min() <= MIDNIGHT_WINDOW:
            opens[day] = float(values.loc[gap.idxmin()])
    series.attrs['open'] = opens
    return series


def with_live(series, reading):
    """The snapshot series with today's live reading as its latest value (for the headline numbers)."""
    if not reading:
        return series
    day = pd.Timestamp(reading['time']).tz_convert('UTC').floor('D')
    out = series.copy()
    out.loc[day] = reading['iv']
    out = out.sort_index()
    out.attrs['open'] = dict(series.attrs.get('open', {}))
    return out

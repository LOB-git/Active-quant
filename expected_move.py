"""One-day expected move from options, and a backtest of how often price stayed inside it.

Live move (SPY, QQQ, DIA, ... from Yahoo option chains):
  * Skips today's expiry. Same-day (0DTE) options have hours left, so their volatility is
    noisy and on 2026-11-18 SPY's 0DTE implied ~15% while the next day's implied ~23%.
  * Reads the move from the at-the-money straddle instead of Yahoo's impliedVolatility
    column. Yahoo's spot is delayed, which makes its call and put IVs disagree (SPY 9% vs 21%
    at the same strike). Put-call parity gives the true forward: F = K + C - P.
  * Straddle -> move: s, the standard deviation of the return to expiry, is solved from the
    Black-Scholes price of that straddle (call + put at the strike nearest the forward).
    Daily move = s / sqrt(trading days to expiry), counting what is left of today's session.

Backtest: Yahoo has no option price history, so the implied side is tested with the
30-day volatility indices built from the same options (VIX for SPY, VXN for QQQ/NQ, VXD for
DIA/YM, GVZ for gold, Deribit DVOL for BTC), next to the 20-day historical fallback.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

NY = 'America/New_York'
SESSION_OPEN, SESSION_CLOSE = pd.Timedelta(hours=9, minutes=30), pd.Timedelta(hours=16)
SESSION_LENGTH = SESSION_CLOSE - SESSION_OPEN
ONE_SIGMA = float(norm.cdf(1) - norm.cdf(-1))   # 0.6827
TWO_SIGMA_OUT = float(2 * norm.cdf(-2))         # 0.0455
MAX_EXPIRIES_TRIED = 3

YAHOO_DAILY_SYMBOL = {'BTC/USD': 'BTC-USD', 'XAUUSD=X': 'GC=F'}
IMPLIED_INDEX = {
    'SPY': '^VIX', 'QQQ': '^VXN', 'NQ=F': '^VXN', 'DIA': '^VXD', 'YM=F': '^VXD',
    'XAUUSD=X': '^GVZ', 'BTC/USD': 'DVOL:BTC',
}


# ---------- live expected move from the option chain ----------

def trading_days_to_expiry(now, expiry):
    """US equity sessions left until the 16:00 New York close on `expiry`.

    Counts the unfinished part of today's session plus every weekday after today up to the
    expiry (exchange holidays are not removed).
    """
    now = pd.Timestamp(now)
    now = now.tz_localize(NY) if now.tzinfo is None else now.tz_convert(NY)
    today = now.normalize()
    expiry_day = pd.Timestamp(expiry).tz_localize(None).normalize()
    if expiry_day.date() < today.date():
        return 0.0
    later = int(np.busday_count((today + pd.Timedelta(days=1)).date(), (expiry_day + pd.Timedelta(days=1)).date()))
    today_left = 0.0
    if today.dayofweek < 5:
        elapsed = min(max(now - today - SESSION_OPEN, pd.Timedelta(0)), SESSION_LENGTH)
        today_left = 1 - elapsed / SESSION_LENGTH
    return later + today_left


def _price(row):
    bid, ask = row.get('bid', np.nan), row.get('ask', np.nan)
    if pd.notna(bid) and pd.notna(ask) and bid > 0 and ask >= bid:
        return (bid + ask) / 2, 'bid/ask mid'
    last = row.get('lastPrice', np.nan)
    return (float(last), 'last trade') if pd.notna(last) and last > 0 else (np.nan, None)


def straddle_move(calls, puts, spot):
    """Standard deviation of the return to expiry implied by the at-the-money straddle.

    Returns {'move', 'forward', 'strike', 'straddle', 'quote'} or None.
    """
    if calls is None or puts is None or calls.empty or puts.empty or not spot or spot <= 0:
        return None
    both = calls.merge(puts, on='strike', suffixes=('_c', '_p'))
    both = both[(both['strike'] - spot).abs() <= spot * 0.05]
    rows = []
    for _, row in both.iterrows():
        call, call_quote = _price({k[:-2]: v for k, v in row.items() if k.endswith('_c')})
        put, put_quote = _price({k[:-2]: v for k, v in row.items() if k.endswith('_p')})
        if np.isfinite(call) and np.isfinite(put):
            quote = 'bid/ask mid' if call_quote == put_quote == 'bid/ask mid' else 'last trade'
            rows.append((float(row['strike']), call, put, quote))
    if not rows:
        return None
    frame = pd.DataFrame(rows, columns=['strike', 'call', 'put', 'quote'])
    parity = frame.loc[(frame['call'] - frame['put']).abs().idxmin()]
    forward = parity['strike'] + parity['call'] - parity['put']
    atm = frame.loc[(frame['strike'] - forward).abs().idxmin()]
    straddle = atm['call'] + atm['put']
    strike = atm['strike']
    if forward <= 0 or straddle <= abs(forward - strike) or straddle >= forward + strike:
        return None
    try:
        move = brentq(lambda s: _bs_straddle(forward, strike, s) - straddle, 1e-6, 5.0)
    except ValueError:
        return None
    return {'move': float(move), 'forward': float(forward), 'strike': float(strike),
            'straddle': float(straddle), 'quote': atm['quote']}


def _bs_straddle(forward, strike, s):
    """Black-Scholes call + put (zero rates) for return standard deviation s to expiry."""
    d1 = (np.log(forward / strike) + s * s / 2) / s
    call = forward * norm.cdf(d1) - strike * norm.cdf(d1 - s)
    return 2 * call - (forward - strike)


def options_daily_move(expirations, load_chain, spot, now=None):
    """Daily one-standard-deviation move from the nearest expiry after today.

    load_chain(expiry) -> (calls, puts). Returns a dict with 'daily_move' (fraction),
    'expiry', 'trading_days', 'forward', 'strike', 'straddle', 'quote', or None.
    """
    now = pd.Timestamp.now(tz=NY) if now is None else pd.Timestamp(now)
    today = (now.tz_localize(NY) if now.tzinfo is None else now.tz_convert(NY)).date()
    later = [expiry for expiry in expirations or [] if pd.Timestamp(expiry).date() > today]
    for expiry in later[:MAX_EXPIRIES_TRIED]:
        days = trading_days_to_expiry(now, expiry)
        if days <= 0:
            continue
        try:
            calls, puts = load_chain(expiry)
        except Exception:
            continue
        result = straddle_move(calls, puts, spot)
        if result is None:
            continue
        daily = result['move'] / np.sqrt(days)
        if 0.0005 < daily < 0.25:
            return {**result, 'daily_move': float(daily), 'expiry': expiry, 'trading_days': float(days)}
    return None


# ---------- backtest of the one-day range ----------

def range_hits(close, sigma, high=None, low=None):
    """Score a one-day range of centre x (1 ± sigma), set at each previous close.

    `sigma` is the daily move known at each day's close (a fraction); it is applied to the next
    day. Returns a dict of hit rates, or None with fewer than 20 days.
    """
    frame = pd.DataFrame({'close': close, 'sigma': sigma})
    if high is not None:
        frame['high'], frame['low'] = high, low
    frame['centre'] = frame['close'].shift(1)
    frame['sigma'] = frame['sigma'].shift(1)
    frame = frame.dropna(subset=['close', 'centre', 'sigma'])
    frame = frame[frame['sigma'] > 0]
    if len(frame) < 20:
        return None
    z = (frame['close'] / frame['centre'] - 1) / frame['sigma']
    result = {
        'Days': int(len(z)),
        'Close inside': float((z.abs() <= 1).mean()),
        'Closed above': float((z > 1).mean()),
        'Closed below': float((z < -1).mean()),
        'Beyond 2x range': float((z.abs() > 2).mean()),
        'Actual ÷ predicted': float(np.sqrt((z ** 2).mean())),
        'Avg range': float(frame['sigma'].mean()),
    }
    if high is not None and frame[['high', 'low']].notna().all(axis=None):
        upper, lower = frame['centre'] * (1 + frame['sigma']), frame['centre'] * (1 - frame['sigma'])
        result['High & low inside'] = float(((frame['high'] <= upper) & (frame['low'] >= lower)).mean())
    else:
        result['High & low inside'] = np.nan
    return result


def historical_sigma(close, window=20):
    """Standard deviation of the last `window` daily log returns (the dashboard's fallback)."""
    return np.log(close).diff().rolling(window).std()


def index_sigma(index_close, days_per_year=252):
    """Daily move implied by a volatility index quoted in annual percent (VIX 25 -> 25%/sqrt(252))."""
    return index_close / 100 / np.sqrt(days_per_year)


def verdict(close_inside):
    if close_inside >= ONE_SIGMA + 0.05:
        return 'too wide: price closed inside it more often than the 68% a one-standard-deviation range should'
    if close_inside <= ONE_SIGMA - 0.05:
        return 'too narrow: price closed outside it more often than the 32% a one-standard-deviation range should'
    return 'well sized: price closed inside it about 68% of the time, as a one-standard-deviation range should'


def range_accuracy(prices, implied=None, implied_label=None, implied_days_per_year=252, window=20):
    """Rows for the accuracy table: the historical fallback, and the implied index if given.

    prices: daily frame with close (and optionally high, low), date-indexed.
    implied: daily volatility index closes (annual percent), date-indexed.
    """
    rows = []
    hist = range_hits(prices['close'], historical_sigma(prices['close'], window),
                      prices.get('high'), prices.get('low'))
    if hist:
        rows.append({'Method': f'Historical ({window}-day)', **hist})
    if implied is not None and len(implied):
        joined = prices.join(implied.rename('index'), how='inner')
        hits = range_hits(joined['close'], index_sigma(joined['index'], implied_days_per_year),
                          joined.get('high'), joined.get('low'))
        if hits:
            rows.append({'Method': f'Implied ({implied_label})', **hits})
    return pd.DataFrame(rows)


def daily_frame(history):
    """Yahoo daily history -> date-indexed close/high/low (time zones dropped)."""
    if history is None or history.empty:
        return pd.DataFrame(columns=['close', 'high', 'low'])
    if isinstance(history.columns, pd.MultiIndex):
        history = history.droplevel(1, axis=1) if history.columns.nlevels > 1 else history
    frame = history.rename(columns=str.lower)[['close', 'high', 'low']].astype(float)
    index = pd.DatetimeIndex(frame.index)
    frame.index = (index.tz_localize(None) if index.tz is not None else index).normalize()
    return frame[~frame.index.duplicated(keep='last')].dropna(subset=['close'])

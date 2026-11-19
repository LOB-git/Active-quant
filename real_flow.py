"""Real money flow from Binance USD-M perpetuals: taker (market-order) buy vs sell volume.

Each Binance kline reports how much of its volume was bought by takers (aggressive market buys).
The rest was sold by takers. So, per candle, in USDT:
    buy  = taker buy quote volume
    sell = quote volume - buy
    delta = buy - sell                     (net money that crossed the spread)
    CVD   = running sum of delta           (cumulative volume delta)
Real Flow Z mirrors the dashboard's Flow Z formula (20-candle flow, z-scored over 50 candles) but uses
this real delta instead of "volume of green candles minus volume of red candles".

This is INFORMATION about who is pushing, not a trading signal: tested on BTC / ETH / SOL 1h
(Jan 2024 - Nov 2026), taker-flow rules had no edge after fees.
"""
import numpy as np
import pandas as pd

from binance_futures import perpetual_symbol

INTERVAL_MS = {'5m': 300_000, '15m': 900_000, '1h': 3_600_000, '4h': 14_400_000}
WINDOWS = {'5m': pd.Timedelta(minutes=5), '15m': pd.Timedelta(minutes=15), '1h': pd.Timedelta(hours=1),
           '4h': pd.Timedelta(hours=4), '24h': pd.Timedelta(hours=24)}


def window_fits(window, interval):
    """True when the `window` of net flow holds whole `interval` candles (5m flow needs 5m candles)."""
    return WINDOWS[window] >= pd.Timedelta(milliseconds=BINANCE_INTERVAL_MS[interval])


def market_id(symbol):
    """'BTC/USDT' -> 'BTCUSDT' (Binance's raw kline id)."""
    return perpetual_symbol(symbol).split(':')[0].replace('/', '')


def fetch_taker_klines(symbol, interval='5m', limit=500, exchange=None):
    """Raw Binance klines with taker volumes. Index: candle open time (UTC-naive)."""
    if exchange is None:
        from binance_futures import usdm_exchange
        exchange = usdm_exchange()
    rows = exchange.fapiPublicGetKlines({'symbol': market_id(symbol), 'interval': interval, 'limit': int(limit)})
    return klines_frame(rows)


BINANCE_INTERVAL_MS = {'1m': 60_000, '5m': 300_000, '15m': 900_000, '30m': 1_800_000, '1h': 3_600_000,
                       '2h': 7_200_000, '4h': 14_400_000, '12h': 43_200_000, '1d': 86_400_000}


def attach_taker_volume(history, symbol, timeframe, exchange=None):
    """Add Binance's 'taker_buy' (base units, like 'volume') to candles from fetch_and_analyze.

    Fetches the raw klines covering the same span and joins them on candle open time. Candles Binance
    does not return keep NaN. Returns a copy; raises if Binance gives nothing back.
    """
    if history is None or history.empty:
        return history
    if exchange is None:
        from binance_futures import usdm_exchange
        exchange = usdm_exchange()
    step = BINANCE_INTERVAL_MS[timeframe]
    index = pd.DatetimeIndex(history.index)
    index = index.tz_convert('UTC').tz_localize(None) if index.tz is not None else index
    since = int(index[0].value // 1_000_000)
    end = int(index[-1].value // 1_000_000)
    rows = []
    while since <= end:
        page = exchange.fapiPublicGetKlines({'symbol': market_id(symbol), 'interval': timeframe,
                                             'startTime': since, 'limit': 1500})
        if not page:
            break
        rows += page
        following = int(page[-1][0]) + step
        if following <= since:
            break
        since = following
    taker = klines_frame(rows)
    if taker.empty:
        raise RuntimeError(f'Binance returned no taker volume for {symbol} {timeframe}')
    out = history.copy()
    out['taker_buy'] = taker['taker_buy'].reindex(index).to_numpy()
    return out


def klines_frame(rows):
    if not rows:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trades',
                                     'taker_buy', 'taker_buy_quote', 'close_time'])
    k = pd.DataFrame(rows)
    frame = pd.DataFrame({
        'open': k[1].astype(float).values, 'high': k[2].astype(float).values, 'low': k[3].astype(float).values,
        'close': k[4].astype(float).values, 'volume': k[5].astype(float).values,
        'quote_volume': k[7].astype(float).values, 'trades': k[8].astype(float).values,
        'taker_buy': k[9].astype(float).values, 'taker_buy_quote': k[10].astype(float).values,
        'close_time': pd.to_datetime(k[6].astype('int64').values + 1, unit='ms'),
    }, index=pd.to_datetime(k[0].astype('int64').values, unit='ms'))
    return frame[~frame.index.duplicated(keep='last')].sort_index()


def _z(series, window=50):
    return (series - series.rolling(window).mean()) / series.rolling(window).std().replace(0, np.nan)


def flow_frame(klines):
    """Per-candle real flow, CVD, Real Flow Z and the old proxy Flow Z for comparison."""
    f = pd.DataFrame(index=klines.index)
    f['close'] = klines['close']
    f['buy_usdt'] = klines['taker_buy_quote']
    f['sell_usdt'] = klines['quote_volume'] - klines['taker_buy_quote']
    f['delta_usdt'] = f['buy_usdt'] - f['sell_usdt']
    f['cvd_usdt'] = f['delta_usdt'].cumsum()
    real = f['delta_usdt'].rolling(20).sum() / klines['quote_volume'].rolling(20).sum().replace(0, np.nan)
    up = klines['close'] >= klines['open']
    vol = klines['volume']
    proxy = (vol.where(up, 0).rolling(20).sum() - vol.where(~up, 0).rolling(20).sum()) / vol.rolling(20).sum().replace(0, np.nan)
    f['real_flow'] = real
    f['real_flow_z'] = _z(real)
    f['proxy_flow_z'] = _z(proxy)
    return f


CROSS_WINDOW = 200   # candles on the chart; each line is scaled to its own min..max over them
CROSS_WATCH = 3      # candles after a cross with bigger-than-usual moves in the backtest


def price_cvd_gap(flow, window=CROSS_WINDOW):
    """Where the price and CVD lines sit relative to each other on a chart of the last `window` candles.

    P~ = (price - min) / (max - min), C~ = (CVD - min) / (max - min) over the trailing window, gap G = P~ - C~.
    G = 0: lines touch. 'cross' = G changed sign on this candle (lines crossed); cross_dir = +1 when price
    went above CVD, -1 when it went below. Causal: each value only uses candles up to that one.

    Backtest (Binance BTC/ETH/SOL perps, 5m-1h, Jan 2024 - Nov 2026): in the 1-3 candles after a cross,
    moves were 8-26% bigger than after candles of the same size without a cross, but direction was a coin flip.
    """
    price, cvd = flow['close'], flow['cvd_usdt']
    lo_p, hi_p = price.rolling(window).min(), price.rolling(window).max()
    lo_c, hi_c = cvd.rolling(window).min(), cvd.rolling(window).max()
    gap = (price - lo_p) / (hi_p - lo_p).replace(0, np.nan) - (cvd - lo_c) / (hi_c - lo_c).replace(0, np.nan)
    side = np.sign(gap)
    cross = (side * side.shift()) < 0
    return pd.DataFrame({'gap': gap, 'cross': cross, 'cross_dir': side.where(cross, 0.0)}, index=flow.index)


def cross_watch(gaps, watch=CROSS_WATCH):
    """Bigger-move watch from the latest closed candle: active while the last cross is <= `watch` candles old."""
    valid = gaps.dropna(subset=['gap'])
    if valid.empty:
        return {'gap': np.nan, 'last_cross': None, 'candles_since': None, 'active': False, 'remaining': 0}
    crosses = np.flatnonzero(valid['cross'].to_numpy())
    if not len(crosses):
        return {'gap': float(valid['gap'].iloc[-1]), 'last_cross': None, 'candles_since': None, 'active': False,
                'remaining': 0}
    since = len(valid) - 1 - int(crosses[-1])
    return {'gap': float(valid['gap'].iloc[-1]), 'last_cross': valid.index[crosses[-1]],
            'last_dir': float(valid['cross_dir'].iloc[crosses[-1]]), 'candles_since': since,
            'active': since < watch, 'remaining': max(watch - since, 0)}


def window_flow(flow, closed_until, window):
    """Net taker flow over the last `window` of closed candles: (net USDT, buy share of volume)."""
    part = flow[(flow.index >= closed_until - window) & (flow.index < closed_until)]
    total = part['buy_usdt'].sum() + part['sell_usdt'].sum()
    if total <= 0:
        return np.nan, np.nan
    return float(part['delta_usdt'].sum()), float(part['buy_usdt'].sum() / total)


def summary(klines, now=None):
    """Numbers for the panel, from closed candles only."""
    now = pd.Timestamp.now(tz='UTC').tz_localize(None) if now is None else pd.Timestamp(now).tz_localize(None) \
        if pd.Timestamp(now).tzinfo else pd.Timestamp(now)
    closed = klines[klines['close_time'] <= now]
    if len(closed) < 2:
        return None
    flow = flow_frame(closed)
    end = closed['close_time'].iloc[-1]
    out = {'flow': flow, 'as_of': end, 'price': float(closed['close'].iloc[-1])}
    for name, span in WINDOWS.items():
        out[f'net_{name}'], out[f'buy_share_{name}'] = window_flow(flow, end, span)
    last = flow.iloc[-1]
    out['real_flow_z'], out['proxy_flow_z'] = float(last['real_flow_z']), float(last['proxy_flow_z'])
    both = flow[['real_flow_z', 'proxy_flow_z']].dropna()
    out['agreement'] = float((np.sign(both['real_flow_z']) == np.sign(both['proxy_flow_z'])).mean()) if len(both) else np.nan
    first = closed[closed.index >= end - WINDOWS['24h']]
    out['change_24h'] = float(closed['close'].iloc[-1] / first['open'].iloc[0] - 1) if len(first) else np.nan
    out['gaps'] = price_cvd_gap(flow)
    out['cross_watch'] = cross_watch(out['gaps'])
    return out

"""GeckoTerminal OHLCV for DEX pools, shaped like the Binance candles the engines expect.

Dexscreener finds the pairs but serves no candles. GeckoTerminal serves candles for the
same pool addresses, under a public rate limit of only a few calls per minute.
"""

import threading
import time
from collections import deque
from urllib.parse import quote

import pandas as pd
import requests


API = 'https://api.geckoterminal.com/api/v2'
HEADERS = {'Accept': 'application/json;version=20230302', 'User-Agent': 'quant-dashboard/1.0'}
TIMEFRAMES = {
    '5m': ('minute', 5), '15m': ('minute', 15), '1h': ('hour', 1),
    '4h': ('hour', 4), '12h': ('hour', 12), '1d': ('day', 1),
}
BAR_FREQUENCY = {
    '5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h', '12h': '12h', '1d': '1D',
}
PAGE_SIZE = 1000  # The most candles one call can return.
MAX_PAGES = 12  # About 41 days of 5m bars. Each page uses up the scarce rate limit.
CALLS_PER_WINDOW = 5  # Measured: a sixth call inside a minute can return HTTP 429.
WINDOW_SECONDS = 60.0
RATE_LIMIT_RETRIES = 4
RATE_LIMIT_WAIT_SECONDS = 30.0
COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

# Dexscreener chain ids that differ from GeckoTerminal network ids. Others are identical.
CHAIN_TO_NETWORK = {
    'ethereum': 'eth', 'polygon': 'polygon_pos', 'avalanche': 'avax', 'fantom': 'ftm',
    'cronos': 'cro', 'sui': 'sui-network', 'gnosis': 'xdai', 'moonbeam': 'glmr',
    'moonriver': 'movr', 'harmony': 'one', 'worldchain': 'world-chain', 'seiv2': 'sei-evm',
    'polygonzkevm': 'polygon-zkevm', 'zora': 'zora-network', 'manta': 'manta-pacific',
}

# The latest failure per asset. fetch_and_analyze hides exceptions behind a None result.
last_errors = {}

_call_times = deque()
_call_lock = threading.Lock()


def asset_id(chain, pair_address, token_address):
    """Build the id for one coin in one pool. Sui token addresses contain colons."""
    return f'{CHAIN_TO_NETWORK.get(chain, chain)}:{pair_address}:{token_address}'


def parse_asset_id(asset):
    parts = str(asset).split(':', 2)
    if len(parts) != 3 or not all(parts):
        raise ValueError(f'{asset} is not a network:pool:token DEX asset id.')
    return parts


def wait_for_rate_limit(clock=time.monotonic, sleep=time.sleep):
    """Block until a call fits in the window. All Streamlit sessions share one IP."""
    with _call_lock:
        now = clock()
        while _call_times and now - _call_times[0] >= WINDOW_SECONDS:
            _call_times.popleft()
        if len(_call_times) >= CALLS_PER_WINDOW:
            sleep(WINDOW_SECONDS - (now - _call_times[0]))
            _call_times.popleft()
        _call_times.append(clock())


def candles_from_rows(rows, timeframe, now=None):
    """Convert API rows to UTC-naive candles on an unbroken bar grid.

    Pools report no candle when nothing traded, but the engines count bars as time.
    Those gaps, and any quiet stretch up to now, become flat zero-volume candles.

    Volume arrives in USD. It is stored in coin units, as on Binance, so that the
    shared helpers' volume x close gives USD back exactly.
    """
    if not rows:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name='date'))
    frame = pd.DataFrame(rows, columns=COLUMNS).apply(pd.to_numeric, errors='coerce')
    frame = frame.dropna().drop_duplicates('timestamp').sort_values('timestamp')
    frame = frame.loc[frame[['open', 'high', 'low', 'close']].gt(0).all(axis=1)]
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS, index=pd.DatetimeIndex([], name='date'))
    frame['date'] = pd.to_datetime(frame['timestamp'], unit='s')
    frame = frame.set_index('date')
    frame['volume'] = frame['volume'] / frame['close']

    bar = pd.Timedelta(BAR_FREQUENCY[timeframe])
    moment = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    moment = moment.tz_localize('UTC') if moment.tzinfo is None else moment.tz_convert('UTC')
    first, latest = frame.index[0], frame.index[-1]
    last_bar = max(latest, first + ((moment.tz_localize(None) - first) // bar) * bar)
    grid = pd.date_range(first, last_bar, freq=bar, name='date')
    filled = frame.reindex(grid)
    filled['close'] = filled['close'].ffill()
    for column in ('open', 'high', 'low'):
        filled[column] = filled[column].fillna(filled['close'])
    filled['volume'] = filled['volume'].fillna(0.0)
    filled['timestamp'] = (grid - pd.Timestamp('1970-01-01')) // pd.Timedelta(seconds=1)
    return filled[COLUMNS]


def request_page(asset, timeframe, before=None, session=requests, sleep=time.sleep,
                 throttle=wait_for_rate_limit):
    """Return one page of rows, newest first. Waits out rate limiting before failing."""
    network, pool, token = parse_asset_id(asset)
    period, aggregate = TIMEFRAMES[timeframe]
    params = {
        'aggregate': aggregate, 'limit': PAGE_SIZE, 'currency': 'usd',
        # The token address decides the priced side: 'base' can return SOL or WETH instead.
        'token': token,
    }
    if before is not None:
        params['before_timestamp'] = int(before)
    url = f'{API}/networks/{quote(network, safe="")}/pools/{quote(pool, safe="")}/ohlcv/{period}'
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        throttle()
        response = session.get(url, params=params, headers=HEADERS, timeout=20)
        if response.status_code != 429:
            break
        if attempt < RATE_LIMIT_RETRIES:
            sleep(RATE_LIMIT_WAIT_SECONDS)
    if response.status_code == 429:
        raise RuntimeError('GeckoTerminal rate limit reached (HTTP 429). Wait a minute and refresh.')
    if response.status_code == 404:
        raise RuntimeError(f'GeckoTerminal does not list this pool on {network}.')
    response.raise_for_status()
    return ((response.json().get('data') or {}).get('attributes') or {}).get('ohlcv_list') or []


def fetch_dex_candles(asset, timeframe='1h', limit=None, start_date=None, end_date=None,
                      now=None, session=requests, sleep=time.sleep, throttle=wait_for_rate_limit):
    """Return candles for one DEX coin, paging backward from now.

    Without a start date one page is read, so sections that want fewer bars can share it.
    Older pages stop quietly when refused: the public API only serves the last 180 days.
    """
    last_errors.pop(asset, None)
    try:
        start = pd.Timestamp(start_date) if start_date else None
        wanted = int(limit) if limit else PAGE_SIZE
        rows, before = [], None
        for _ in range(MAX_PAGES):
            try:
                page = request_page(asset, timeframe, before, session, sleep, throttle)
            except requests.HTTPError:
                if not rows:
                    raise
                break
            if not page:
                break
            rows.extend(page)
            oldest = min(int(row[0]) for row in page)
            # A short page is the pool's whole remaining history: skip the wasted call.
            if len(page) < PAGE_SIZE or (before is not None and oldest >= before):
                break
            before = oldest
            if start is not None:
                if pd.Timestamp(oldest, unit='s') <= start:
                    break
            elif len(rows) >= wanted:
                break
        frame = candles_from_rows(rows, timeframe, now)
    except Exception as error:
        last_errors[asset] = f'{type(error).__name__}: {str(error)[:200]}'
        raise
    if start is not None:
        frame = frame.loc[frame.index >= start]
    if end_date:
        frame = frame.loc[frame.index < pd.Timestamp(end_date)]
    return frame if start is not None else frame.tail(wanted)

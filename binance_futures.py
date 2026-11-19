"""Binance USD-M perpetual and European options data for the Derivative Crypto tab."""

import re
import time
from datetime import timedelta
from functools import lru_cache

import ccxt
import numpy as np
import pandas as pd
import requests


OPTIONS_API = 'https://eapi.binance.com/eapi/v1'
PAGE_SIZE = 1500
DEFAULT_CANDLES = 1500  # About 62 days of 1h bars, matching Yahoo's 60d US history.
OPTION_HORIZON_DAYS = 60  # Same expiry window as the US Polygon GEX calculation.
STABLE_BASES = {'USDC', 'FDUSD', 'TUSD', 'USDP', 'BUSD', 'DAI', 'USDE', 'USD1', 'RLUSD'}


def http_status(error):
    match = re.search(r'\b(403|418|429|451)\b', str(error))
    return match.group(1) if match else None


def with_retries(call, attempts=3, delay=2.0):
    """Retry brief network faults. Bans and region blocks are not retried."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except ccxt.NetworkError as error:
            if attempt == attempts or http_status(error) in {'403', '418', '451'}:
                raise
            time.sleep(delay * attempt)


def describe_exchange_failure(error):
    """Explain an exchange failure in terms the dashboard user can act on."""
    status = http_status(error)
    if status == '418':
        return 'Your IP is temporarily banned for too many requests (HTTP 418). Wait before scanning again.'
    if status == '429' or isinstance(error, ccxt.RateLimitExceeded):
        return 'The exchange rate limit was reached (HTTP 429). Wait a minute and scan again.'
    if status in {'403', '451'}:
        return f'The exchange refused access from this location (HTTP {status}). A VPN or different network may be needed.'
    if isinstance(error, ccxt.RequestTimeout):
        return 'The request timed out. Check your connection and scan again.'
    if isinstance(error, ccxt.NetworkError):
        return f'A network error occurred ({type(error).__name__}). Check your connection and scan again.'
    return f'{type(error).__name__}: {str(error)[:200]}'


@lru_cache(maxsize=1)
def usdm_exchange():
    """Share one client: every new ccxt instance re-downloads the market list."""
    return ccxt.binanceusdm({'enableRateLimit': True})


def perpetual_symbol(symbol):
    """Convert BTC/USDT, BTCUSDT, BTC-USDT or BTC/USDT:USDT to the ccxt perp symbol."""
    text = str(symbol).strip().upper().replace('-', '/')
    if text.endswith(':USDT'):
        text = text[:-5]
    if '/' not in text:
        if not text.endswith('USDT') or len(text) <= 4:
            raise ValueError(f'{symbol} is not a USDT-margined Binance symbol.')
        text = f'{text[:-4]}/USDT'
    if not text.endswith('/USDT'):
        raise ValueError(f'{symbol} is not a USDT-margined Binance symbol.')
    return f'{text}:USDT'


def fetch_usdm_candles(symbol, timeframe='1h', limit=None, start_date=None, end_date=None,
                       exchange=None):
    """Return UTC-naive OHLCV, paging forward from the requested start.

    The page size actually served is not assumed: paging only stops when the
    exchange returns nothing new or the current candle has been reached.
    """
    client = exchange or usdm_exchange()
    market = perpetual_symbol(symbol)
    candles = int(limit) if limit else DEFAULT_CANDLES
    bar_ms = int(client.parse_timeframe(timeframe) * 1000)
    now = int(client.milliseconds())
    since = (
        client.parse8601(f'{start_date}T00:00:00Z') if start_date
        else now - bar_ms * candles
    )
    rows = []
    while since <= now:
        page = client.fetch_ohlcv(market, timeframe, since=since, limit=PAGE_SIZE)
        if not page:
            break
        rows.extend(page)
        following = int(page[-1][0]) + bar_ms
        if following <= since:
            break
        since = following
    columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not rows:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], name='date'))
    frame = pd.DataFrame(rows, columns=columns).drop_duplicates('timestamp').sort_values('timestamp')
    frame['date'] = pd.to_datetime(frame['timestamp'], unit='ms')
    frame = frame.set_index('date')
    if end_date:
        frame = frame.loc[frame.index < pd.Timestamp(end_date)]
    return frame if start_date else frame.tail(candles)


def is_crypto_perpetual(market):
    """Binance also lists equity, commodity and FX perps; keep only crypto contracts."""
    info = market.get('info') or {}
    return (
        bool(market.get('swap')) and bool(market.get('linear'))
        and market.get('quote') == 'USDT' and market.get('active') is not False
        and str(market.get('symbol', '')).endswith(':USDT')
        and info.get('underlyingType') == 'COIN'
        and market.get('base') not in STABLE_BASES
    )


def top_usdm_symbols(count=10, exchange=None):
    """Rank crypto perps by 24h quote volume. Binance publishes no market cap."""
    client = exchange or usdm_exchange()
    markets = client.load_markets()
    tickers = client.fetch_tickers()
    ranked = sorted(
        (
            (float(ticker.get('quoteVolume') or 0.0), symbol)
            for symbol, ticker in tickers.items()
            if symbol in markets and is_crypto_perpetual(markets[symbol])
        ),
        reverse=True,
    )
    return [symbol.replace(':USDT', '') for _, symbol in ranked[:count]]


def intraday_money_flow(symbol, interval='5m', now=None, exchange=None):
    """Today's UTC (Ghana) inflow and outflow notional from up versus down candles."""
    moment = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    candles = fetch_usdm_candles(
        symbol, timeframe=interval, start_date=moment.strftime('%Y-%m-%d'), exchange=exchange,
    )
    if candles.empty:
        return None
    notional = candles['close'] * candles['volume']
    is_up = candles['close'] >= candles['open']
    inflow = float(notional[is_up].sum())
    outflow = float(notional[~is_up].sum())
    return {'inflow': inflow, 'outflow': outflow, 'net_flow': inflow - outflow, 'bars': len(candles)}


def options_gex_profile(contracts, marks, open_interest):
    """Sum gamma x open interest x contract size by strike; puts count negative."""
    gamma = {row['symbol']: float(row['gamma']) for row in marks}
    contracts_open = {row['symbol']: float(row['sumOpenInterest']) for row in open_interest}
    rows = [
        {
            'strike': float(contract['strikePrice']),
            'gamma_exposure': (
                gamma[contract['symbol']] * contracts_open[contract['symbol']]
                * float(contract.get('unit', 1))
                * (-1 if contract['side'] == 'PUT' else 1)
            ),
        }
        for contract in contracts
        if contract['symbol'] in gamma and contract['symbol'] in contracts_open
    ]
    if not rows:
        return pd.Series(dtype=float, name='gamma_exposure')
    profile = pd.DataFrame(rows).groupby('strike')['gamma_exposure'].sum()
    return profile if np.any(profile.to_numpy() != 0) else pd.Series(dtype=float, name='gamma_exposure')


def fetch_options_gex(base_asset, now=None, session=requests):
    """Return the live options GEX profile, or raise ValueError with the reason.

    All-or-nothing: a partly loaded chain would misstate the zero-gamma level.
    """
    moment = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    horizon = moment + timedelta(days=OPTION_HORIZON_DAYS)

    def get(path, **params):
        response = session.get(f'{OPTIONS_API}/{path}', params=params, timeout=15)
        response.raise_for_status()
        return response.json()

    contracts = [
        contract for contract in get('exchangeInfo').get('optionSymbols', [])
        if contract.get('underlying') == f'{base_asset}USDT'
        and contract.get('status') == 'TRADING'
        and moment.value // 1_000_000 <= int(contract['expiryDate']) <= horizon.value // 1_000_000
    ]
    if not contracts:
        raise ValueError(f'Binance lists no {base_asset} options expiring within {OPTION_HORIZON_DAYS} days.')
    open_interest = []
    for expiry in sorted({contract['symbol'].split('-')[1] for contract in contracts}):
        open_interest.extend(get('openInterest', underlyingAsset=base_asset, expiration=expiry))
    profile = options_gex_profile(contracts, get('mark'), open_interest)
    if profile.empty:
        raise ValueError(f'Binance {base_asset} options returned no gamma with open interest.')
    return profile

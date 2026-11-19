"""Forex Quant Edge Scan: the quant layer on major FX pairs and gold, sized in lots and pips."""

import numpy as np
import pandas as pd
import streamlit as st

from derivative_crypto_panel import render_quant_scan_results, render_quant_settings, run_quant_scan


FX_SCAN_KEY = 'fx_quant_scan'
# Yahoo ticker -> (base, quote, pip size, units per standard lot).
# XAUUSD is gold; the app's loader maps it to COMEX gold futures (GC=F), as Yahoo has no spot gold.
FX_PAIRS = {
    'EURUSD=X': ('EUR', 'USD', 0.0001, 100_000),
    'GBPUSD=X': ('GBP', 'USD', 0.0001, 100_000),
    'USDJPY=X': ('USD', 'JPY', 0.01, 100_000),
    'AUDUSD=X': ('AUD', 'USD', 0.0001, 100_000),
    'USDCAD=X': ('USD', 'CAD', 0.0001, 100_000),
    'USDCHF=X': ('USD', 'CHF', 0.0001, 100_000),
    'NZDUSD=X': ('NZD', 'USD', 0.0001, 100_000),
    'EURJPY=X': ('EUR', 'JPY', 0.01, 100_000),
    'GBPJPY=X': ('GBP', 'JPY', 0.01, 100_000),
    'EURGBP=X': ('EUR', 'GBP', 0.0001, 100_000),
    'XAUUSD': ('XAU', 'USD', 0.10, 100),
}
FX_MIN_BARS = 150
FX_FEE_BPS = 3.0  # Typical major-pair spread plus commission, round trip.


def display_name(symbol):
    return symbol.replace('=X', '')


def required_symbols(selected):
    """Add the USD pairs needed to convert each selected pair's quote currency into USD."""
    needed = list(selected)
    for symbol in selected:
        quote = FX_PAIRS[symbol][1]
        if quote == 'USD':
            continue
        helper = f'{quote}USD=X' if f'{quote}USD=X' in FX_PAIRS else f'USD{quote}=X'
        if helper in FX_PAIRS and helper not in needed:
            needed.append(helper)
    return needed


def usd_per_quote(histories):
    """{symbol: USD value of one unit of its quote currency}, from the latest closes."""
    latest = {symbol: float(frame['close'].iloc[-1]) for symbol, frame in histories.items() if len(frame)}
    rates = {}
    for symbol in histories:
        quote = FX_PAIRS.get(symbol, (None, 'USD'))[1]
        if quote == 'USD':
            rates[symbol] = 1.0
        elif latest.get(f'{quote}USD=X'):
            rates[symbol] = latest[f'{quote}USD=X']
        elif latest.get(f'USD{quote}=X'):
            rates[symbol] = 1.0 / latest[f'USD{quote}=X']
        else:
            rates[symbol] = np.nan  # Sizing is refused rather than guessed.
    return rates


def add_fx_columns(table):
    """Stop distance in pips, size in standard/mini/micro lots, and pip value per lot in USD."""
    result = table.copy()
    spec = result['Symbol'].map(lambda symbol: FX_PAIRS.get(symbol, (None, None, np.nan, np.nan)))
    pip = spec.map(lambda item: item[2])
    lot = spec.map(lambda item: item[3])
    result['Stop (pips)'] = (result['Entry'] - result['Stop']).abs() / pip
    result['Lots'] = result['Size (units)'] / lot
    result['Mini Lots'] = result['Lots'] * 10
    result['Micro Lots'] = result['Lots'] * 100
    result['Pip $ / Lot'] = pip * lot * pd.to_numeric(result.get('Quote to Account'), errors='coerce')
    result['Symbol'] = result['Symbol'].map(display_name)
    return result


def usd_exposure(table):
    """Count actionable trades that are long or short the US dollar."""
    long_usd = short_usd = 0
    for symbol, verdict in zip(table['Symbol'], table['Verdict']):
        if verdict not in ('LONG', 'SHORT'):
            continue
        base, quote = FX_PAIRS.get(symbol, FX_PAIRS.get(f'{symbol}=X', (None, None)))[:2]
        buys_usd = (base == 'USD') == (verdict == 'LONG') if 'USD' in (base, quote) else None
        if buys_usd is True:
            long_usd += 1
        elif buys_usd is False:
            short_usd += 1
    return long_usd, short_usd


def render_forex_quant_scan(fetch_history):
    st.divider()
    st.subheader('💱 Forex & Gold — Quant Edge Scan')
    st.caption(
        'The same Quant Edge Layer on major FX pairs and gold, using Yahoo Finance candles (about 60 days '
        'of intraday history). Sizes are converted to USD, so JPY, CAD, CHF and GBP-quoted pairs risk the '
        'right dollar amount. Forex has no volume on Yahoo, so the money-flow term is neutral.'
    )
    settings = render_quant_settings(prefix='fx_q', fee_default=FX_FEE_BPS)
    c1, c2, c3 = st.columns(3)
    timeframe = c1.selectbox('Forex scan timeframe', ['15m', '1h', '4h'], index=1, key='fx_qscan_tf',
                             help='4h leaves only about 300 candles from Yahoo, so backtests are thin.')
    selected = c2.multiselect('Pairs', list(FX_PAIRS), default=list(FX_PAIRS), key='fx_qscan_pairs',
                              format_func=display_name)
    only_actionable = c3.checkbox('Show only LONG / SHORT', value=False, key='fx_qscan_only')

    if st.button(f'Scan {len(selected)} Forex Pairs for Quant Edge', type='primary', key='fx_qscan_run',
                 disabled=not selected):
        run_quant_scan(FX_SCAN_KEY, required_symbols(selected), fetch_history, timeframe, settings,
                       workers=1, candles=None, min_bars=FX_MIN_BARS, round_the_clock=False,
                       conversions=usd_per_quote)

    scan = st.session_state.get(FX_SCAN_KEY)
    if scan is not None and not scan['table'].empty:
        long_usd, short_usd = usd_exposure(scan['table'])
        if long_usd + short_usd > 1:
            st.info(f'USD exposure across signals: {long_usd} trade(s) buy USD, {short_usd} sell USD. '
                    'Trades on the same side of USD tend to win or lose together.')
    render_quant_scan_results(
        FX_SCAN_KEY, 'fx_qscan', settings, only_actionable,
        empty_hint='Click Scan to run the quant layer on the forex pairs.', min_bars=FX_MIN_BARS,
        correlation_warning=(
            'Most pairs are priced against the US dollar, so several signals can be one USD bet. '
            'Check the USD exposure line above and size down if they all lean the same way.'
        ),
        rank_label='List #', decorate=add_fx_columns,
        extra_columns=('Stop (pips)', 'Lots', 'Mini Lots', 'Micro Lots', 'Pip $ / Lot'),
        extra_formats={'Stop (pips)': '{:,.1f}', 'Lots': '{:,.2f}', 'Mini Lots': '{:,.2f}',
                       'Micro Lots': '{:,.1f}', 'Pip $ / Lot': '${:,.2f}'},
    )
    st.caption(
        '1 standard lot = 100,000 units of the first currency (100 oz for gold); mini = 0.1 lot; '
        'micro = 0.01 lot. Round lots down to what your broker allows; if it rounds to 0, the trade is '
        'too large for your risk limit. Gold uses COMEX futures prices, which run slightly above spot. '
        'The fee default is 3 bps; raise it if your spread is wider.'
    )

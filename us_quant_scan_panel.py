"""Quant Edge Scan for the US Indices tab: the crypto quant layer applied to Yahoo data."""

import numpy as np
import streamlit as st

from derivative_crypto_panel import render_quant_scan_results, render_quant_settings, run_quant_scan


US_SCAN_KEY = 'us_quant_scan'
OVERVIEW_SYMBOLS = ['SPY', 'QQQ', 'NQ=F', 'DIA', 'YM=F', '^VIX', 'DX-Y.NYB']
# Only symbols the app's fetcher routes to Yahoo; anything else would be sent to Binance.
EXTRA_SYMBOLS = ['ES=F', 'GC=F', 'GLD', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA',
                 'AVGO', 'LLY', 'JPM', 'GBPUSD=X', '^FTSE']
# Dollars per index point: (E-mini / full contract, Micro contract).
CONTRACT_MULTIPLIERS = {'NQ=F': (20, 2), 'ES=F': (50, 5), 'YM=F': (5, 0.5), 'GC=F': (100, 10)}
NOT_TRADEABLE = {
    '^VIX': 'index only: trade via VX futures or VIX ETPs (UVXY/VIXY)',
    'DX-Y.NYB': 'index only: trade via DX futures or UUP',
    '^FTSE': 'index only: trade via FTSE futures or ISF.L',
}
EQUITY_INDEX = {'SPY', 'QQQ', 'NQ=F', 'DIA', 'YM=F', 'ES=F'}
US_MIN_BARS = 150  # Yahoo serves about 60 days of intraday history.


def add_us_contract_columns(table):
    """Convert unit size into futures contracts and flag symbols that are not directly tradeable."""
    result = table.copy()
    full = result['Symbol'].map(lambda symbol: CONTRACT_MULTIPLIERS.get(symbol, (np.nan, np.nan))[0])
    micro = result['Symbol'].map(lambda symbol: CONTRACT_MULTIPLIERS.get(symbol, (np.nan, np.nan))[1])
    result['E-mini Contracts'] = result['Size (units)'] / full
    result['Micro Contracts'] = result['Size (units)'] / micro
    for symbol, note in NOT_TRADEABLE.items():
        mask = result['Symbol'].eq(symbol)
        result.loc[mask, 'Reason'] = result.loc[mask, 'Reason'].astype(str) + f' ({note})'
    return result


def vix_conflict(table):
    """True when equities and VIX point the same way: they normally move opposite."""
    actionable = table[table['Verdict'].isin(['LONG', 'SHORT'])]
    equity_sides = set(actionable.loc[actionable['Symbol'].isin(EQUITY_INDEX), 'Verdict'])
    vix_side = set(actionable.loc[actionable['Symbol'].eq('^VIX'), 'Verdict'])
    return bool(equity_sides & vix_side)


def render_us_quant_scan(fetch_history):
    st.divider()
    st.subheader('🔎 US Indices & VIX — Quant Edge Scan')
    st.caption(
        'Runs the Quant Edge Layer (Hurst regime, IC, triple-barrier backtest, half-Kelly sizing) on the '
        'US overview symbols using Yahoo Finance candles (about 60 days of intraday history). '
        'Volatility is annualized from the actual bar rate, so overnight and weekend gaps are respected.'
    )
    settings = render_quant_settings(prefix='us_q')
    c1, c2, c3 = st.columns(3)
    timeframe = c1.selectbox('US scan timeframe', ['15m', '1h', '4h'], index=1, key='us_qscan_tf',
                             help='4h leaves only about 150–250 candles from Yahoo, so backtests are thin.')
    extras = c2.multiselect('Add more symbols', EXTRA_SYMBOLS, key='us_qscan_extra')
    only_actionable = c3.checkbox('Show only LONG / SHORT', value=False, key='us_qscan_only')
    symbols = OVERVIEW_SYMBOLS + [symbol for symbol in extras if symbol not in OVERVIEW_SYMBOLS]

    if st.button(f'Scan {len(symbols)} US Symbols for Quant Edge', type='primary', key='us_qscan_run'):
        # yfinance downloads share global state, so they run one at a time.
        run_quant_scan(US_SCAN_KEY, symbols, fetch_history, timeframe, settings,
                       workers=1, candles=None, min_bars=US_MIN_BARS, round_the_clock=False)

    scan = st.session_state.get(US_SCAN_KEY)
    if scan is not None and not scan['table'].empty and vix_conflict(scan['table']):
        st.warning('Equity indices and ^VIX point the same way. They normally move opposite, so treat both '
                   'signals with caution.')
    render_quant_scan_results(
        US_SCAN_KEY, 'us_qscan', settings, only_actionable,
        empty_hint='Click Scan to run the quant layer on the US indices and VIX.', min_bars=US_MIN_BARS,
        correlation_warning=(
            'SPY, QQQ, NQ, DIA, YM and ES are all the US stock market. Several LONGs (or SHORTs) at once '
            'are one larger bet; pick the strongest one or split your risk.'
        ),
        rank_label='List #', decorate=add_us_contract_columns,
        extra_columns=('E-mini Contracts', 'Micro Contracts'),
    )
    st.caption(
        'Futures sizes: NQ \\$20/pt (Micro \\$2), ES \\$50/pt (Micro \\$5), YM \\$5/pt (Micro \\$0.50), '
        'GC \\$100/oz (Micro \\$10). Round contracts down; if it rounds to 0, the trade is too large for '
        'your risk limit. Check fees for your broker in Quant Edge Settings.'
    )

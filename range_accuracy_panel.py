"""📏 How accurate is the one-day range? Backtest panel for the Volatility Quantum Analysis tab."""
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

import expected_move as em
from implied_volatility_panel import fetch_dvol

HISTORY_PERIOD = '3y'
STATE_KEY = 'range_accuracy'


def _yahoo_daily(symbol):
    return em.daily_frame(yf.Ticker(symbol).history(period=HISTORY_PERIOD, interval='1d', auto_adjust=False))


def _implied_series(source):
    """(daily index closes in annual percent, days per year) for an IMPLIED_INDEX source."""
    if source.startswith('DVOL:'):
        dvol = fetch_dvol(source.split(':', 1)[1], days=3 * 365)
        index = pd.DatetimeIndex(dvol.index).tz_localize(None).normalize()
        return pd.Series(dvol.to_numpy() * 100, index=index).groupby(level=0).last(), 365
    return _yahoo_daily(source)['close'], 252


@st.cache_data(ttl=3600, show_spinner=False)
def backtest_symbol(symbol):
    """Accuracy rows for one dashboard symbol (historical fallback, plus its implied index if any)."""
    prices = _yahoo_daily(em.YAHOO_DAILY_SYMBOL.get(symbol, symbol))
    source = em.IMPLIED_INDEX.get(symbol)
    implied, days, label = None, 252, None
    if source:
        try:
            implied, days = _implied_series(source)
            label = 'Deribit DVOL' if source.startswith('DVOL:') else source
        except Exception:
            implied = None
    table = em.range_accuracy(prices, implied, label, days)
    if not table.empty:
        table.insert(0, 'Symbol', symbol)
        table['From'] = prices.index.min().date() if len(prices) else None
    return table


def render_range_accuracy(symbol, symbols):
    st.subheader('📏 How Accurate Is the One-Day Range?')
    st.caption(
        'Replays the one-day range on about 3 years of daily data: each day the range is set at the previous '
        'close as close × (1 ± daily move), then the next close is checked. A correct one-standard-deviation '
        'range holds the close about **68%** of the time and is broken by more than 2× only about **4.6%**. '
        'Yahoo keeps no history of option prices, so the implied side is tested with the 30-day volatility '
        'index made from the same options (VIX for SPY, VXN for QQQ/NQ, VXD for DIA/YM, GVZ for gold, '
        'Deribit DVOL for BTC). The live range above uses next-day options, so it reacts faster than these.'
    )
    c1, c2 = st.columns(2)
    run_one = c1.button(f'Backtest the range for {symbol}', key='range_acc_one')
    run_all = c2.button(f'Backtest all {len(symbols)} symbols', key='range_acc_all')
    if run_one or run_all:
        wanted = symbols if run_all else [symbol]
        tables, failed = [], []
        bar = st.progress(0.0, text='Loading daily history…')
        for done, name in enumerate(wanted, start=1):
            try:
                table = backtest_symbol(name)
            except Exception:
                table = pd.DataFrame()
            (tables.append(table) if not table.empty else failed.append(name))
            bar.progress(done / len(wanted), text=f'Backtested {name}')
        bar.empty()
        st.session_state[STATE_KEY] = {
            'table': pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(), 'failed': failed,
        }

    result = st.session_state.get(STATE_KEY)
    if not result:
        st.info('Press a button to measure how often price stayed inside the range.')
        return
    if result['failed']:
        st.warning(f"No usable daily history for: {', '.join(result['failed'])}.")
    table = result['table']
    if table.empty:
        return

    pct = '{:.1%}'
    st.dataframe(
        table[['Symbol', 'Method', 'Close inside', 'High & low inside', 'Closed above', 'Closed below',
               'Beyond 2x range', 'Actual ÷ predicted', 'Avg range', 'Days', 'From']].style.format({
            'Close inside': pct, 'High & low inside': pct, 'Closed above': pct, 'Closed below': pct,
            'Beyond 2x range': pct, 'Actual ÷ predicted': '{:.2f}', 'Avg range': '±{:.2%}',
        }, na_rep='—').background_gradient(
            subset=['Close inside'], cmap='RdYlGn', vmin=em.ONE_SIGMA - 0.15, vmax=em.ONE_SIGMA + 0.15,
        ),
        width='stretch', hide_index=True,
    )
    for _, row in table.iterrows():
        st.markdown(f"- **{row['Symbol']} · {row['Method']}:** {em.verdict(row['Close inside'])}. "
                    f"{row['Beyond 2x range']:.1%} of days moved more than 2× the range (normal: 4.6%).")
    st.caption(
        '**Close inside** is the main score (68% = right size). **High & low inside** is stricter: the whole day, '
        'wicks included, stayed in the range. **Actual ÷ predicted** compares real moves with the range: 1.00 is '
        'perfect, below 1 means the range was too wide, above 1 too narrow. **Beyond 2x range** shows the fat '
        'tails: days far bigger than the range expected.'
    )

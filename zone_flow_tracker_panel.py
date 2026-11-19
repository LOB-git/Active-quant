"""🎯 Zone Flow Tracker tab: forward test of Zone Battle + Highest Flow Z Rise/Drop on every timeframe."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import zone_flow_tracker as zft

SIDE_STYLE = {'LONG': 'color: #16a34a; font-weight: 700', 'SHORT': 'color: #dc2626; font-weight: 700',
              'BUY': 'color: #16a34a; font-weight: 700', 'SELL': 'color: #dc2626; font-weight: 700'}


def _signed(value):
    if value is None or pd.isna(value):
        return ''
    return 'color: #16a34a' if value > 0 else 'color: #dc2626' if value < 0 else ''


def _equity_chart(trades, title):
    eq = np.cumprod(1 + trades['Net %'].to_numpy() / 100) * 100 - 100
    fig = go.Figure(go.Scatter(x=trades['Exit Time (UTC)'], y=eq, mode='lines+markers', line={'color': '#7c3aed'}))
    fig.add_hline(y=0, line_dash='dash', line_color='grey')
    fig.update_layout(title=title, height=320, margin={'l': 10, 'r': 10, 't': 40, 'b': 10}, yaxis_title='%')
    return fig


def _reference_backtest(fetch_and_analyze, calc, symbol):
    rows = []
    source = zft.SOURCES[symbol]
    for tf in zft.TIMEFRAMES:
        try:
            history = zft.fetch_for(fetch_and_analyze, symbol, tf)
            signals = zft.confirmed_signals(history, tf, calc, source['fvg'], flow=source.get('flow', 'proxy'))
            trades, open_pos = zft.trades_from_signals(signals, float(history['close'].iloc[-1]))
            stats = zft.summarize(trades)
            idx = pd.DatetimeIndex(history.index)
            rows.append({'Timeframe': tf, 'From (UTC)': f'{idx[0]:%Y-%m-%d}', 'To (UTC)': f'{idx[-1]:%Y-%m-%d}',
                         'Candles': len(history), 'Signals': len(signals), 'Trades': stats['Trades'],
                         'Win %': stats['Win %'], 'Total %': stats['Total %'], 'Max DD %': stats['Max DD %'],
                         'Buy & Hold %': (history['close'].iloc[-1] / history['open'].iloc[0] - 1) * 100})
        except Exception as exc:
            rows.append({'Timeframe': tf, 'From (UTC)': f'failed: {exc}'})
    return pd.DataFrame(rows)


def render_zone_flow_tracker(fetch_and_analyze, calc):
    st.subheader('🎯 Zone Flow Tracker — forward test on every timeframe')
    st.caption(
        '**BUY** when a Zone Battle ends as a **Strong Buyer Victory** and the Highest Flow Z **Rise**\'s Flow + '
        'Momentum + Volatility contribution beats the Highest Flow Z **Drop**\'s (in size). **SELL** when it ends as '
        'a **Strong Seller Victory** and the drop\'s beats the rise\'s. Hold until the opposite signal, then reverse; '
        'fill at the next candle\'s open; 0.05% fee per side. Highest rise / drop: Ghana day so far on 5m · 15m · 1h, '
        'last 24 candles on 4h · 1d. Closed candles only. **BTC/USDT (real flow)** runs the same rules with every '
        'Flow Z built from Binance taker buy vs sell volume instead of candle colour, so the two can be compared.'
    )
    st.warning(
        '**This is a watch list, not a trading signal.** In backtests (BTC, Jan 2024 – Nov 2026) this was the best of '
        'the variants tested but not reliably profitable: 1h made +24% while buy & hold made +56% (t = 0.57, −46% '
        'drawdown, negative without its 3 best trades); 5m lost −86%. Only signals **after tracking began** count '
        'below, so this table is a genuine out-of-sample test. Give it a few months before judging it.'
    )

    c1, c2 = st.columns([2, 3])
    if c1.button('Update now (all strategies, all timeframes)', key='zft_update'):
        with st.spinner(f'Fetching {len(zft.DEFAULT_SYMBOLS) * len(zft.TIMEFRAMES)} data sets and checking for new signals…'):
            results = zft.update_all(fetch_and_analyze, calc)
        problems = {f'{s} {tf}': r for (s, tf), r in results.items() if not r.endswith('signal(s)')}
        new = sum(int(r.split()[0]) for r in results.values() if r.endswith('signal(s)'))
        st.success(f'Updated {len(results) - len(problems)} of {len(results)}: {new} new signal(s).')
        if problems:
            st.caption('Problems: ' + '; '.join(f'{k}: {v}' for k, v in problems.items()))
    c2.caption('The alert runner (`alert runner.bat`) updates this every 5 minutes on its own, even with the '
               'dashboard closed. Tracking for each timeframe starts the first time it is updated.')

    overview = zft.forward_overview()
    started = overview[overview['Tracking Since (UTC)'] != 'not started']
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Timeframes tracking', f'{len(started)} / {len(overview)}')
    m2.metric('Signals logged', int(overview['Signals'].sum()))
    m3.metric('Closed trades', int(overview['Closed Trades'].sum()))
    m4.metric('Open positions', int((overview['Position'] != 'FLAT').sum()))
    if started.empty:
        st.info('Tracking has not started yet. Press **Update now** (or start the alert runner) to begin.')
    st.dataframe(
        overview.style.format({
            'Win %': '{:.0f}%', 'Forward Total %': '{:+.2f}%', 'Max DD %': '{:+.2f}%', 'Entry': '{:,.2f}',
            'Unrealized %': '{:+.2f}%', 'Last Price': '{:,.2f}',
        }, na_rep='—').map(lambda v: SIDE_STYLE.get(v, ''), subset=['Position'])
        .map(_signed, subset=['Forward Total %', 'Unrealized %']),
        width='stretch', hide_index=True,
    )

    s1, s2 = st.columns(2)
    symbol = s1.selectbox('Symbol', zft.DEFAULT_SYMBOLS, format_func=lambda s: zft.SOURCES[s]['label'], key='zft_sym')
    tf = s2.radio('Timeframe', zft.TIMEFRAMES, horizontal=True, key='zft_tf')
    row = overview[(overview['Symbol'] == symbol) & (overview['Timeframe'] == tf)].iloc[0]
    log = zft.load_log()
    mine = log[(log['Symbol'] == symbol) & (log['Timeframe'] == tf)]
    trades, open_pos = zft.trades_from_signals(mine, row['Last Price'])
    st.markdown(f'#### {symbol} · {tf}')
    if open_pos:
        o1, o2, o3 = st.columns(3)
        o1.metric('Open position', open_pos['Side'], f"since {open_pos['Entry Time (UTC)']} UTC",
                  delta_color='off', delta_arrow='off')
        o2.metric('Entry', f"{open_pos['Entry']:,.2f}")
        o3.metric('Unrealized (after fees)', f"{open_pos['Unrealized %']:+.2f}%" if pd.notna(open_pos['Unrealized %']) else '—')
    else:
        st.caption('No open position.')
    if not trades.empty:
        st.plotly_chart(_equity_chart(trades, f'Forward closed trades — cumulative net return ({symbol} {tf})'),
                        width='stretch')
        st.dataframe(trades.style.format({'Entry': '{:,.2f}', 'Exit': '{:,.2f}', 'Net %': '{:+.2f}%'})
                     .map(lambda v: SIDE_STYLE.get(v, ''), subset=['Side']).map(_signed, subset=['Net %']),
                     width='stretch', hide_index=True)
    if mine.empty:
        st.caption(f"No signals since tracking began ({row['Tracking Since (UTC)']}).")
    else:
        st.markdown('**Signals logged**')
        st.dataframe(mine.drop(columns=['Symbol']).iloc[::-1].style.format({
            'Zone Bottom': '{:,.2f}', 'Zone Top': '{:,.2f}', 'Battle Score': '{:+.2f}', 'Rise Total': '{:+.1f}',
            'Drop Total': '{:.1f}', 'Signal Price': '{:,.2f}', 'Fill Price': '{:,.2f}'}, na_rep='pending')
            .map(lambda v: SIDE_STYLE.get(v, ''), subset=['Side']), width='stretch', hide_index=True)

    with st.expander('Reference backtest on the loaded candles (historical, NOT forward)'):
        st.caption('Runs the same rules over the candles currently loaded for each timeframe (about 1,000 for the '
                   'Binance perpetual; Yahoo keeps 60 days of 5m / 15m). In-sample: useful context only.')
        if st.button(f'Run reference backtest for {symbol}', key='zft_ref'):
            with st.spinner('Backtesting every timeframe…'):
                st.session_state['zft_ref_result'] = (symbol, _reference_backtest(fetch_and_analyze, calc, symbol))
        ref = st.session_state.get('zft_ref_result')
        if ref and ref[0] == symbol:
            st.dataframe(ref[1].style.format({'Win %': '{:.0f}%', 'Total %': '{:+.2f}%', 'Max DD %': '{:+.2f}%',
                                              'Buy & Hold %': '{:+.2f}%'}, na_rep='—'),
                         width='stretch', hide_index=True)

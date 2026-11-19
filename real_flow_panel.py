"""💵 Real Money Flow panel: Binance taker buy vs sell volume (actual money crossing the spread)."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import real_flow as rf
from live_state import live_cached

COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT']
SCAN_SIZE = 15


def money(value):
    if value is None or not np.isfinite(value):
        return '—'
    sign = '+' if value >= 0 else '−'
    value = abs(value)
    if value >= 1e9:
        return f'{sign}${value / 1e9:,.2f}B'
    if value >= 1e6:
        return f'{sign}${value / 1e6:,.1f}M'
    return f'{sign}${value / 1e3:,.0f}K'


@st.cache_data(ttl=60, show_spinner=False)
def _load(symbol, interval, limit=500, fresh_slot=0):
    return rf.fetch_taker_klines(symbol, interval, limit)


load = live_cached(_load)  # refreshes with the dashboard's live updates


def flow_chart(summary, symbol, interval):
    flow = summary['flow'].tail(200)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.04,
                        specs=[[{'secondary_y': True}], [{}]])
    fig.add_trace(go.Scatter(x=flow.index, y=flow['close'], name='Price', line={'color': '#94a3b8'}),
                  row=1, col=1, secondary_y=False)
    cvd = flow['cvd_usdt'] - flow['cvd_usdt'].iloc[0]
    fig.add_trace(go.Scatter(x=flow.index, y=cvd / 1e6, name='CVD ($M, from chart start)',
                             line={'color': '#7c3aed', 'width': 2}), row=1, col=1, secondary_y=True)
    gaps = summary.get('gaps')
    if gaps is not None:
        marks = gaps.reindex(flow.index)
        marks = marks[marks['cross'].fillna(False).astype(bool)]
        if len(marks):
            up = marks['cross_dir'] > 0
            fig.add_trace(go.Scatter(
                x=marks.index, y=flow.loc[marks.index, 'close'], mode='markers', name='Price–CVD cross',
                marker={'symbol': 'diamond', 'size': 9, 'color': np.where(up, '#f59e0b', '#0ea5e9'),
                        'line': {'width': 1, 'color': '#111827'}},
                customdata=np.where(up, 'price went above CVD', 'price went below CVD'),
                hovertemplate='Price–CVD cross (%{customdata})<br>Bigger move likely next ' + str(rf.CROSS_WATCH)
                              + ' candles, direction unknown<extra></extra>'),
                row=1, col=1, secondary_y=False)
    fig.add_trace(go.Bar(x=flow.index, y=flow['delta_usdt'] / 1e6, name='Net taker flow per candle ($M)',
                         marker_color=np.where(flow['delta_usdt'] >= 0, '#16a34a', '#dc2626')), row=2, col=1)
    fig.update_layout(title=f'{symbol} real money flow ({interval} candles)', height=520, hovermode='x unified',
                      margin={'l': 10, 'r': 10, 't': 50, 'b': 10}, legend={'orientation': 'h', 'y': -0.12})
    fig.update_yaxes(title_text='Price', row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text='CVD $M', row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text='$M', row=2, col=1)
    return fig


def scan_row(symbol):
    try:
        s = rf.summary(rf.fetch_taker_klines(symbol, '1h', 80))
    except Exception:
        return None
    if s is None:
        return None
    return {'Symbol': symbol, 'Price': s['price'], '24h Change %': s['change_24h'] * 100,
            'Net Flow 1h': s['net_1h'], 'Net Flow 4h': s['net_4h'], 'Net Flow 24h': s['net_24h'],
            'Buy Share 24h %': s['buy_share_24h'] * 100, 'Real Flow Z': s['real_flow_z']}


def scan(symbols, workers=6):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = [r for r in pool.map(scan_row, symbols) if r]
    if not rows:  # every fetch failed (e.g. Binance blocks the host): nothing to sort
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values('Net Flow 24h', ascending=False, kind='stable').reset_index(drop=True)


def _signed(value):
    if value is None or not np.isfinite(value):
        return ''
    return 'color: #16a34a' if value > 0 else 'color: #dc2626' if value < 0 else ''


def render_cross_watch(watch, interval):
    """Bigger-move warning when the price and CVD lines have just crossed."""
    c1, c2 = st.columns([1, 2])
    gap = watch['gap']
    c1.metric('Price–CVD gap', f'{gap:+.2f}' if np.isfinite(gap) else '—',
              'lines touching' if np.isfinite(gap) and abs(gap) < 0.1 else
              ('price drawn above CVD' if gap > 0 else 'price drawn below CVD') if np.isfinite(gap) else None,
              delta_color='off', delta_arrow='off',
              help='G = price position − CVD position, each scaled to its own low..high over the last '
                   f'{rf.CROSS_WINDOW} candles (as the chart draws them). 0 = the lines touch; a sign change = they cross.')
    if watch['active']:
        side = 'above' if watch.get('last_dir', 0) > 0 else 'below'
        ago = 'on the last closed candle' if watch['candles_since'] == 0 else f"{watch['candles_since']} candle(s) ago"
        c2.warning(
            f"⚡ **Bigger-move watch ON** — price crossed {side} the CVD line {ago} "
            f"({watch['last_cross']:%H:%M} UTC). For the next {watch['remaining']} {interval} candle(s) expect moves about "
            '10–25% larger than usual. **Direction unknown**: widen stops / cut size; do not enter on the cross alone.'
        )
    else:
        last = (f"last cross {watch['candles_since']} candles ago ({watch['last_cross']:%m-%d %H:%M} UTC)"
                if watch['last_cross'] is not None else f'no cross in the last {rf.CROSS_WINDOW} candles')
        c2.info(f'Bigger-move watch off — {last}. It turns on for {rf.CROSS_WATCH} candles after the price and CVD '
                'lines cross (◆ on the chart).')


def render_real_flow(top_symbols=None):
    st.subheader('💵 Real Money Flow (Binance taker buy vs sell)')
    st.caption(
        'Every Binance candle reports how much volume was bought by **market orders** (takers) versus sold by them. '
        '**Net flow** = taker buys − taker sells in USDT: the money that actually crossed the spread. **CVD** adds it '
        'up over time. **Real Flow Z** uses the same formula as the dashboard\'s Flow Z (20 candles, z-scored over 50) '
        'but with this real flow instead of "volume of green candles − volume of red candles" (**Proxy Flow Z**).'
    )
    c1, c2 = st.columns([1, 2])
    symbol = c1.selectbox('Coin', COINS, key='rf_symbol')
    interval = c2.radio('Candles', list(rf.INTERVAL_MS), index=0, horizontal=True, key='rf_interval')
    render_flow_view(symbol, interval, key='rf_main')

    with st.expander(f'Top {SCAN_SIZE} perpetuals — where the money went in the last 24h'):
        if st.button(f'Scan top {SCAN_SIZE} perpetuals', key='rf_scan'):
            try:
                symbols = top_symbols(SCAN_SIZE) if top_symbols else COINS
            except Exception:
                symbols = COINS
            with st.spinner('Reading taker flow…'):
                st.session_state['rf_scan_result'] = (pd.Timestamp.now(tz='UTC'), scan(symbols))
        result = st.session_state.get('rf_scan_result')
        if result:
            at, table = result
            if table.empty:
                st.warning('No flow data came back. Binance may be unreachable from this server '
                           '(it blocks US hosts such as Streamlit Cloud); run the dashboard on your PC instead.')
                result = None
        if result:
            at, table = result
            st.caption(f'Scanned {at:%Y-%m-%d %H:%M} UTC, 1h candles, sorted by net taker flow over 24h.')
            st.dataframe(
                table.style.format({'Price': '{:,.6g}', '24h Change %': '{:+.2f}%', 'Net Flow 1h': money,
                                    'Net Flow 4h': money, 'Net Flow 24h': money, 'Buy Share 24h %': '{:.1f}%',
                                    'Real Flow Z': '{:+.2f}'}, na_rep='—')
                .map(_signed, subset=['24h Change %', 'Net Flow 1h', 'Net Flow 4h', 'Net Flow 24h', 'Real Flow Z']),
                width='stretch', hide_index=True)

    with st.expander('🧪 Backtest: lines close + last hour buy share above 51% → trade'):
        from flow_cvd_backtest_panel import render_flow_rule_backtest
        render_flow_rule_backtest()

    st.warning(
        '**Information, not a signal.** Tested on BTC / ETH / SOL hourly data (Jan 2024 – Nov 2026): trading on '
        'taker flow (4h–72h), CVD, absorption or funding had no edge after fees. On liquid coins, flow this visible '
        'is priced in within minutes. Use it to see who is pushing right now, not as a reason to enter.'
    )


def window_sources(summary, symbol, interval, loader=None):
    """Which summary to read each net-flow window from: the chart's own candles when the window holds whole
    candles, else one built from 5m candles (so 5m and 15m flow also show on 15m/1h/4h charts)."""
    sources, short = {}, None
    for name in rf.WINDOWS:
        if rf.window_fits(name, interval):
            sources[name] = summary
            continue
        if short is None:
            try:
                short = rf.summary((loader or load)(symbol, '5m', 300)) or False
            except Exception:
                short = False
        sources[name] = short or None
    return sources


def render_flow_view(symbol, interval, key='rf_view'):
    """Net flow, Real Flow Z, the price–CVD cross watch and the chart for one Binance perpetual.

    Shared by the Real Money Flow panel and the Risk & Leverage Watch List drop-down. Returns the summary
    (None when nothing could be loaded).
    """
    try:
        s = rf.summary(load(symbol, interval))
    except Exception as exc:
        st.warning(f'Could not load Binance taker data for {symbol}: {exc}')
        return None
    if s is None:
        st.warning('Not enough closed candles yet.')
        return None
    sources = window_sources(s, symbol, interval)
    m = st.columns(len(rf.WINDOWS))
    for col, name in zip(m, rf.WINDOWS):
        source = sources.get(name)
        net = source[f'net_{name}'] if source else np.nan
        share = source[f'buy_share_{name}'] if source else np.nan
        col.metric(f'Net taker flow, last {name}', money(net),
                   f'{share:.1%} of volume was buying' if np.isfinite(share) else None,
                   delta_color='off', delta_arrow='off',
                   help=None if source is s else 'Read from 5m candles: a window this short holds no complete '
                                                 f'{interval} candle.')
    z = st.columns(3)
    z[0].metric('Real Flow Z (now)', f"{s['real_flow_z']:+.2f}")
    z[1].metric('Proxy Flow Z (dashboard formula)', f"{s['proxy_flow_z']:+.2f}")
    z[2].metric('Same direction', f"{s['agreement']:.0%}" if np.isfinite(s['agreement']) else '—',
                help='Share of candles on this chart where Real Flow Z and the proxy Flow Z had the same sign.')
    if np.isfinite(s['real_flow_z']) and np.isfinite(s['proxy_flow_z']) and np.sign(s['real_flow_z']) != np.sign(s['proxy_flow_z']):
        st.info('Right now the proxy Flow Z and the real taker flow **disagree**: candle colours are saying one thing '
                'while market orders are doing the other.')
    render_cross_watch(s['cross_watch'], interval)
    st.plotly_chart(flow_chart(s, symbol, interval), width='stretch', key=f'{key}_chart')
    st.caption(f"Closed candles up to {s['as_of']:%Y-%m-%d %H:%M} UTC (cached 1 min). ◆ = price–CVD cross "
               '(orange: price went above CVD, blue: below). Backtest on Binance BTC/ETH/SOL perps, 5m–1h, '
               'Jan 2024 – Nov 2026: the 1–3 candles after a cross moved 8–26% more than same-size candles without '
               'one, but up and down equally often, and every long/short rule built on it lost after fees.')
    return s

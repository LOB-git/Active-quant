"""🕒 Session Flow Log tab: the recorded Highest Flow Z Rise / Drop of every trading session."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import session_flow_recorder as sfr

YAHOO_SYMBOLS = {'BTC/USD', 'BTC-USD'}
DEFAULT_SYMBOLS = ['BTC/USD']
DEFAULT_TOP = 10
BIAS_STYLE = {'BUY': 'color: #16a34a; font-weight: 700', 'SELL': 'color: #dc2626; font-weight: 700'}
VIEW_COLUMNS = [
    'Symbol', 'Status', 'Bias', 'Margin', 'Rise Strength', 'Drop Strength',
    'Rise Flow Contribution', 'Rise Momentum Contribution', 'Rise Volatility Contribution', 'Rise Price Change',
    'Rise Direction Score', 'Rise Time (UTC)',
    'Drop Flow Contribution', 'Drop Momentum Contribution', 'Drop Volatility Contribution', 'Drop Price Change',
    'Drop Direction Score', 'Drop Time (UTC)', 'Change %', 'Last', 'Bars', 'Updated (UTC)',
]


def session_symbols(fixed=DEFAULT_SYMBOLS, top_n=DEFAULT_TOP, top_symbols=None):
    """BTC/USD (Yahoo) plus the top N Binance crypto perpetuals, without duplicates."""
    symbols = list(fixed)
    if top_n:
        try:
            if top_symbols is None:
                from binance_futures import top_usdm_symbols as top_symbols
            symbols += list(top_symbols(int(top_n)))
        except Exception:
            pass
    return list(dict.fromkeys(symbols))


def session_fetcher(fetch_and_analyze, timeframe='5m'):
    """BTC/USD comes from Yahoo exactly as in the US Indices tab; the rest from Binance USD-M perpetuals."""
    def fetch(symbol):
        if symbol.upper() in YAHOO_SYMBOLS:
            return fetch_and_analyze(symbol, timeframe=timeframe, silent=True)
        return fetch_and_analyze(symbol, market='usdm', timeframe=timeframe, silent=True, limit=1000)
    return fetch


def _session_times_caption(now):
    day = sfr.trading_day(now)
    windows = sfr.session_windows(day)
    parts = [f"**{name}** {start:%H:%M}–{end:%H:%M}" for name, (start, end) in windows.items() if name != 'Day']
    return f"Trading day {day:%a %d %b} in Ghana time (UTC): " + ' · '.join(parts) + \
        ' — the day starts at the New York close (17:00 New York), which is the Asia open.'


def _strength_chart(view):
    data = view.dropna(subset=['Rise Strength', 'Drop Strength'])
    fig = go.Figure()
    fig.add_bar(x=data['Symbol'], y=data['Rise Strength'], name='Highest Rise: Flow + Momentum', marker_color='#16a34a')
    fig.add_bar(x=data['Symbol'], y=data['Drop Strength'], name='Highest Drop: |Flow + Momentum|', marker_color='#dc2626')
    fig.update_layout(barmode='group', height=360, margin={'l': 10, 'r': 10, 't': 40, 'b': 10},
                      title='Rise vs Drop strength (contribution points)', legend={'orientation': 'h', 'y': -0.25})
    return fig


def _side_cards(row, side):
    st.markdown(f"**Highest Flow Z {side}** — {row[f'{side} Time (UTC)']} UTC" if pd.notna(row[f'{side} Time (UTC)'])
                else f'**Highest Flow Z {side}** — none in this session')
    if pd.isna(row[f'{side} Time (UTC)']):
        return
    a = st.columns(4)
    a[0].metric('Flow Z Change', f"{row[f'{side} Flow Z Change']:+.3f}")
    a[1].metric('Momentum Z Change', f"{row[f'{side} Momentum Z Change']:+.3f}")
    a[2].metric('Volatility Z Change', f"{row[f'{side} Volatility Z Change']:+.3f}")
    a[3].metric('Price Change', f"{row[f'{side} Price Change']:+,.6g}")
    b = st.columns(4)
    b[0].metric('Flow Contribution', f"{row[f'{side} Flow Contribution']:+.1f}")
    b[1].metric('Momentum Contribution', f"{row[f'{side} Momentum Contribution']:+.1f}")
    b[2].metric('Volatility Contribution', f"{row[f'{side} Volatility Contribution']:+.1f}")
    b[3].metric('Direction Score', f"{row[f'{side} Direction Score']:+.1f}")


def render_session_flow_log(fetch_and_analyze, top_symbols=None):
    st.subheader('🕒 Session Flow Log — Highest Flow Z Rise vs Drop')
    now = pd.Timestamp.now(tz='UTC')
    st.caption(_session_times_caption(now))
    day, session, _, end = sfr.current_session(now)
    m1, m2, m3 = st.columns(3)
    m1.metric('Trading day', f'{day:%a %d %b}')
    m2.metric('Session now', session or '—')
    if end is not None:
        left = end - now
        m3.metric('Session ends in', f'{int(left.total_seconds() // 3600)}h {int(left.total_seconds() % 3600 // 60):02d}m',
                  f'{end:%H:%M} UTC', delta_color='off', delta_arrow='off')

    c1, c2 = st.columns([1, 3])
    top_n = int(c1.number_input('Top perpetuals', 0, 30, DEFAULT_TOP, 1, key='sfl_top'))
    c2.write('')
    if c2.button('Record now (BTC/USD + top perpetuals, 5m)', key='sfl_record'):
        symbols = session_symbols(DEFAULT_SYMBOLS, top_n, top_symbols)
        with st.spinner(f'Recording {len(symbols)} symbols…'):
            results = sfr.record_sessions(symbols, session_fetcher(fetch_and_analyze, '5m'), '5m')
        problems = {s: r for s, r in results.items() if not r.endswith('row(s)')}
        st.success(f'Recorded {len(results) - len(problems)} of {len(results)} symbols.')
        if problems:
            st.caption('Not recorded: ' + '; '.join(f'{s}: {r}' for s, r in problems.items()))

    history = sfr.load_history()
    if history.empty:
        st.info('Nothing recorded yet. Press **Record now**, or keep `alert runner.bat` running: it records every '
                '5 minutes, all day, even when the dashboard is closed.')
        return

    days = sorted(history['Trading Day'].astype(str).unique(), reverse=True)
    f1, f2, f3 = st.columns([1, 1, 2])
    pick_day = f1.selectbox('Trading day', days, key='sfl_day')
    sessions = ['Asia', 'London', 'New York', 'Day']
    default_session = sessions.index(session) if session and pick_day == str(day) else 3
    pick_session = f2.radio('Session', sessions, index=default_session, horizontal=True, key='sfl_session')
    view = history[(history['Trading Day'].astype(str) == pick_day) & (history['Session'] == pick_session)]
    symbols = list(view['Symbol'])
    chosen = f3.multiselect('Symbols', symbols, default=symbols, key=f'sfl_symbols_{pick_day}_{pick_session}')
    view = view[view['Symbol'].isin(chosen)]
    if view.empty:
        st.info(f'No {pick_session} rows recorded for {pick_day}.')
        return

    counts = view['Bias'].value_counts()
    b1, b2, b3 = st.columns(3)
    b1.metric('BUY bias', int(counts.get('BUY', 0)))
    b2.metric('SELL bias', int(counts.get('SELL', 0)))
    b3.metric('Live / Complete', f"{int((view['Status'] == 'Live').sum())} / {int((view['Status'] == 'Complete').sum())}")
    number = {c: '{:+.1f}' for c in VIEW_COLUMNS if 'Contribution' in c or 'Strength' in c or 'Score' in c}
    number.update({'Margin': '{:+.1f}', 'Rise Price Change': '{:+,.6g}', 'Drop Price Change': '{:+,.6g}',
                   'Change %': '{:+.2f}%', 'Last': '{:,.6g}'})
    st.dataframe(view.reindex(columns=VIEW_COLUMNS).style.format(number, na_rep='—')
                 .map(lambda v: BIAS_STYLE.get(v, ''), subset=['Bias']),
                 width='stretch', hide_index=True)
    st.caption(
        'Strength = Flow Contribution + Momentum Contribution of the session\'s Highest Flow Z Rise; for the Highest '
        'Flow Z Drop it is the size of the same sum. **Bias BUY** when the rise is stronger, **SELL** when the drop is. '
        'Margin = Rise Strength − Drop Strength. Live rows keep updating until the session ends.'
    )
    st.plotly_chart(_strength_chart(view), width='stretch')

    pick = st.selectbox('Detail', list(view['Symbol']), key='sfl_detail')
    row = view[view['Symbol'] == pick].iloc[0]
    st.markdown(f"### {pick} · {pick_session} · {pick_day}: **{row['Bias']}** (margin {row['Margin']:+.1f})"
                if pd.notna(row['Margin']) else f'### {pick} · {pick_session} · {pick_day}')
    left, right = st.columns(2)
    with left:
        _side_cards(row, 'Rise')
    with right:
        _side_cards(row, 'Drop')

    st.download_button('Download full session log (CSV)', history.to_csv(index=False).encode(),
                       file_name='session_flow_history.csv', mime='text/csv', key='sfl_download')
    first = days[-1]
    st.info(
        f'Recording since trading day {first} ({len(days)} day(s), {len(history)} rows). The BUY/SELL bias is your '
        'own reading of the Rise vs Drop cards and has **not been tested yet**. Once a few weeks are logged (or from '
        'past candles now), it can be checked: does a BUY bias in one session lead to price rising in the next?'
    )

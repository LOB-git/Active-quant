"""Volatility Quantum Analysis for the top Binance USD-M perpetuals.

Runs the same signal as the 🌌 Volatility Quantum Analysis tab (volatility outlook from
mean reversion, then 50 EMA trend and RSI momentum) on every perpetual, so the whole
market can be read at once.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st

OUTLOOK_LOOKBACK = 1000  # volatility readings the median is taken over (the scan loads 1000 candles)
SCAN_SIZE = 100
SCAN_MAX = 200
SCAN_CANDLES = 1000
SCAN_WORKERS = 6
MIN_BARS = 100
SCAN_KEY = 'vq_scan'
PREDICTION_ORDER = {'MOVE UP': 0, 'MOVE DOWN': 0, 'CONSOLIDATE / CHOP': 1, 'ERROR': 2}
COLUMNS = ['Volume Rank', 'Symbol', 'Price', 'Prediction', 'Volatility Outlook', 'Price Trend', 'Momentum',
           'RSI', 'Volatility', 'Vol Median', 'Vol Percentile', 'Expected Daily Move',
           'Range Low', 'Range High']


def _rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / loss)


def volatility_outlook(vol):
    """Volatility mean reversion: INCREASE when the latest reading is below the median of the
    last OUTLOOK_LOOKBACK readings, DECREASE when above.

    Replaces the old phase gradient of psi = vol + i * vol change, which in practice just meant
    "volatility just fell". Walk-forward on BTC/USDT (1d/4h/1h, 2020-2026) the phase gradient
    called the next 20 bars right about 50% of the time; this rule about 68%, in every year.

    Returns (outlook, median); ('NEUTRAL', nan) when there are no readings.
    """
    recent = np.asarray(vol, dtype=float)[-OUTLOOK_LOOKBACK:]
    recent = recent[~np.isnan(recent)]
    if not len(recent):
        return 'NEUTRAL', float('nan')
    median = float(np.median(recent))
    current = recent[-1]
    outlook = 'INCREASE' if current < median else 'DECREASE' if current > median else 'NEUTRAL'
    return outlook, median


def quantum_state(frame):
    """The tab's quantum signal for the latest candle, with the numbers behind it.

    Returns None when there is too little history.
    """
    if frame is None or 'close' not in frame or len(frame) < 50:
        return None
    df = frame[['close']].copy()
    df['ema_50'] = frame['ema_50'] if 'ema_50' in frame else frame['close'].ewm(span=50, adjust=False).mean()
    df['rsi'] = frame['rsi'] if 'rsi' in frame else _rsi(frame['close'])
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_ret'].rolling(20).std() * np.sqrt(252)  # Same scaling as the tab's surface.
    df = df.dropna()
    if len(df) < 12:
        return None

    vol = df['volatility'].to_numpy()
    outlook, median = volatility_outlook(vol)
    last = df.iloc[-1]
    trend = 'BULLISH' if last['close'] > last['ema_50'] else 'BEARISH'
    momentum = 'POSITIVE' if last['rsi'] > 50 else 'NEGATIVE'
    prediction = 'CONSOLIDATE / CHOP'
    if outlook == 'INCREASE' and trend == 'BULLISH' and momentum == 'POSITIVE':
        prediction = 'MOVE UP'
    elif outlook == 'INCREASE' and trend == 'BEARISH' and momentum == 'NEGATIVE':
        prediction = 'MOVE DOWN'
    return {
        'Price': float(last['close']), 'Prediction': prediction, 'Volatility Outlook': outlook,
        'Price Trend': trend, 'Momentum': momentum, 'RSI': float(last['rsi']),
        'Volatility': float(vol[-1]), 'Vol Median': median, 'Vol Percentile': float((vol <= vol[-1]).mean() * 100),
    }


def expected_daily_move(frame):
    """One-standard-deviation daily move from the last 20 daily closes (crypto trades every day).

    Falls back to scaling the candle volatility up to one day when there are fewer than
    10 daily returns (e.g. 15m candles). Returns a fraction, or None.
    """
    close = frame['close'].dropna()
    if len(close) < 3 or not isinstance(close.index, pd.DatetimeIndex):
        return None
    daily = np.log(close.resample('1D').last().dropna()).diff().dropna()
    if len(daily) >= 10:
        return float(daily.tail(20).std())
    bar_returns = np.log(close).diff().dropna()
    step = close.index.to_series().diff().median()
    if not len(bar_returns) or pd.isna(step) or step <= pd.Timedelta(0):
        return None
    return float(bar_returns.tail(200).std() * np.sqrt(pd.Timedelta('1D') / step))


def quantum_scan_row(symbol, frame):
    state = quantum_state(frame)
    if state is None:
        return None
    move = expected_daily_move(frame)
    state['Symbol'] = symbol
    state['Expected Daily Move'] = move
    state['Range Low'] = state['Price'] * (1 - move) if move is not None else np.nan
    state['Range High'] = state['Price'] * (1 + move) if move is not None else np.nan
    return state


def scan_volatility_quantum(symbols, fetch, timeframe, workers=SCAN_WORKERS, progress=None,
                            candles=SCAN_CANDLES, min_bars=MIN_BARS):
    """Fetch each symbol in parallel and compute its quantum row.

    Returns (table in volume order, failed symbols, {symbol: candles}).
    """
    histories, failed = {}, []

    def load(symbol):
        try:
            return symbol, fetch(symbol, timeframe=timeframe, silent=True, limit=candles)
        except Exception:
            return symbol, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(load, symbol) for symbol in symbols]
        for done, future in enumerate(as_completed(futures), start=1):
            symbol, frame = future.result()
            if frame is None or len(frame) < min_bars:
                failed.append(symbol)
            else:
                histories[symbol] = frame
            if progress:
                progress(done / len(symbols), symbol)

    rows = []
    for rank, symbol in enumerate(symbols, start=1):
        if symbol not in histories:
            continue
        try:
            row = quantum_scan_row(symbol, histories[symbol])
        except Exception:
            row = None
        if row is None:
            failed.append(symbol)
            continue
        row['Volume Rank'] = rank
        rows.append(row)
    table = pd.DataFrame(rows).reindex(columns=COLUMNS) if rows else pd.DataFrame(columns=COLUMNS)
    ordered = {symbol: histories[symbol] for symbol in symbols if symbol in histories}
    return table, sorted(set(failed), key=symbols.index), ordered


def rank_quantum_scan(table):
    """MOVE UP / MOVE DOWN first, then by volume rank."""
    if table.empty:
        return table
    order = table['Prediction'].map(PREDICTION_ORDER).fillna(3)
    return (table.assign(_order=order)
            .sort_values(['_order', 'Volume Rank'], kind='stable')
            .drop(columns='_order').reset_index(drop=True))


def _color_prediction(value):
    return {'MOVE UP': 'color: #16a34a; font-weight: 600', 'MOVE DOWN': 'color: #dc2626; font-weight: 600'}.get(value, '')


def render_volatility_quantum_scan(top_symbols, fetch, plot_surface):
    """top_symbols(count) -> symbols; fetch(symbol, timeframe, silent, limit) -> candles;
    plot_surface(df, symbol) -> (figure, data) or None."""
    st.subheader('🔭 Top 100 Derivatives — Volatility Quantum Scan')
    st.caption(
        'Runs the quantum volatility signal above on the most active Binance USD-M perpetuals by 24-hour '
        f'USDT volume, using {SCAN_CANDLES} candles each. Volatility Outlook is INCREASE when volatility is '
        'below its median over those candles (it tends to revert up). **MOVE UP** needs that, price above '
        'the 50 EMA and RSI above 50; **MOVE DOWN** needs that, price below the 50 EMA and '
        'RSI below 50. Everything else is CONSOLIDATE / CHOP. Takes about 30–60 seconds for 100 perpetuals.'
    )
    c1, c2, c3 = st.columns(3)
    timeframe = c1.selectbox('Scan timeframe', ['15m', '1h', '4h'], index=1, key='vq_scan_tf')
    count = int(c2.number_input('Number of perpetuals', 10, SCAN_MAX, SCAN_SIZE, 10, key='vq_scan_count'))
    only_moves = c3.checkbox('Show only MOVE UP / MOVE DOWN', key='vq_scan_only_moves')

    if st.button(f'Run Volatility Quantum Scan on the Top {count} Perpetuals', key='vq_scan_run'):
        try:
            symbols = top_symbols(count)
        except Exception as exc:
            st.error(f'Could not load the Binance perpetual list: {exc}')
            symbols = []
        if symbols:
            bar = st.progress(0.0, text='Scanning…')
            table, failed, histories = scan_volatility_quantum(
                symbols, fetch, timeframe,
                progress=lambda share, symbol: bar.progress(share, text=f'Scanned {symbol}'),
            )
            bar.empty()
            st.session_state[SCAN_KEY] = {
                'table': rank_quantum_scan(table), 'failed': failed, 'histories': histories,
                'timeframe': timeframe, 'count': len(symbols),
                'scanned_at': pd.Timestamp.now(tz='UTC'),
            }

    scan = st.session_state.get(SCAN_KEY)
    if not scan:
        st.info('Press the button to scan the top perpetuals.')
        return
    table = scan['table']
    st.caption(
        f"Scanned {scan['count']} perpetuals on {scan['timeframe']} at "
        f"{scan['scanned_at']:%Y-%m-%d %H:%M} UTC (Ghana time)."
        + (f" Skipped {len(scan['failed'])} with too little data: {', '.join(scan['failed'])}." if scan['failed'] else '')
    )
    if table.empty:
        st.warning('No perpetual returned enough data.')
        return

    counts = table['Prediction'].value_counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('MOVE UP', int(counts.get('MOVE UP', 0)))
    m2.metric('MOVE DOWN', int(counts.get('MOVE DOWN', 0)))
    m3.metric('CONSOLIDATE / CHOP', int(counts.get('CONSOLIDATE / CHOP', 0)))
    m4.metric('Volatility below median',int((table['Volatility Outlook'] == 'INCREASE').sum()))

    shown = table[table['Prediction'].isin(['MOVE UP', 'MOVE DOWN'])] if only_moves else table
    if shown.empty:
        st.info('No perpetual has a MOVE UP or MOVE DOWN prediction right now.')
    else:
        st.dataframe(
            shown.style.format({
                'Price': '{:,.6g}', 'RSI': '{:.1f}', 'Volatility': '{:.2%}', 'Vol Median': '{:.2%}',
                'Vol Percentile': '{:.0f}%', 'Expected Daily Move': '±{:.2%}',
                'Range Low': '{:,.6g}', 'Range High': '{:,.6g}',
            }, na_rep='—').map(_color_prediction, subset=['Prediction']),
            width='stretch', hide_index=True,
        )
        st.download_button(
            'Download scan (CSV)', shown.to_csv(index=False).encode(),
            file_name=f"volatility_quantum_scan_{scan['scanned_at']:%Y%m%d_%H%M}.csv",
            mime='text/csv', key='vq_scan_download',
        )
    st.caption(
        'Volatility: annualised the same way as the surface above. Vol Median: the level the outlook compares '
        'against. Vol Percentile: where today\'s volatility sits '
        'in the loaded history (high = stretched). Expected Daily Move: one standard deviation of the last 20 daily '
        'returns, and Range Low/High apply it to the price. It is a rough range, not a guarantee.'
    )

    pick = st.selectbox('Show the volatility surface for', shown['Symbol'] if not shown.empty else table['Symbol'],
                        key='vq_scan_pick')
    if pick and st.button(f'Draw surface for {pick}', key='vq_scan_draw'):
        result = plot_surface(scan['histories'][pick].copy(), pick)
        if result and result[0] is not None:
            st.pyplot(result[0])
        else:
            st.warning(f'Could not draw the volatility surface for {pick}.')

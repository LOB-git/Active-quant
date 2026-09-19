"""Optional live smoke test: real data, exact candle matching and UI render path."""
import importlib.util
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from streamlit.testing.v1 import AppTest
from qqq_nq_signals import build_qqq_nq_signals, signals_on_ghana_day
from qqq_nq_daily_alerts import prepare_daily_alerts


def overview_daily_alerts(dashboard, frames):
    """QA source: use the overview's builder and exact bar-close/Momentum inputs."""
    alerts = []
    for symbol in ('QQQ', 'NQ=F', '^VIX'):
        history = frames[symbol].copy()
        z = pd.to_numeric(history['z_score'], errors='coerce').fillna(0.)
        momentum = pd.to_numeric(history['momentum'], errors='coerce')
        momentum_z = (momentum - momentum.rolling(50).mean()) / momentum.rolling(50).std().replace(0, float('nan'))
        history.index = pd.DatetimeIndex(history.index) + pd.Timedelta(hours=1)
        z.index = momentum_z.index = history.index
        alerts.extend(dashboard.build_daily_zscore_alert_history(
            history, z, symbol, '1h', threshold=1., momentum_z_series=momentum_z,
            as_of=pd.Timestamp.now(tz='UTC'),
        ))
    clean, issue = prepare_daily_alerts(alerts)
    if issue:
        raise RuntimeError(issue)
    print('Daily source rows by symbol:', clean.groupby('Symbol').size().to_dict(), flush=True)
    return alerts


def main():
    spec = importlib.util.spec_from_file_location('dashboard', Path(__file__).resolve().parents[1] / 'Untitled-1.py')
    dashboard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dashboard)
    frames = {symbol: dashboard.fetch_and_analyze(symbol, timeframe='1h', silent=True)
              for symbol in ('QQQ', 'NQ=F', '^VIX')}
    for symbol, frame in frames.items():
        if frame is None or frame.empty:
            raise RuntimeError(f'Live data unavailable for {symbol}')
        print(symbol, 'candles:', len(frame), 'latest:', frame.index[-1], flush=True)
    alerts = overview_daily_alerts(dashboard, frames)
    result = build_qqq_nq_signals(frames['QQQ'], frames['NQ=F'], daily_alerts=alerts)
    matched = result['QQQ Ready'].eq(True) & result['NQ Ready'].eq(True)
    print('Comparable completed candles:', matched.sum(), flush=True)
    print('States:', result['Signal'].value_counts().to_dict(), flush=True)
    if not matched.any():
        raise RuntimeError('No comparable real-data candles were available for UI QA')
    best_day = pd.Timestamp(result.loc[matched, 'Signal Time (Ghana)'].iloc[-1]).date()
    sample = signals_on_ghana_day(result, best_day)
    print('Latest comparable Ghana day:', best_day, 'rows:', len(sample), flush=True)
    print(sample[['QQQ Score', 'NQ Score', 'Signal', 'Reason']].tail(8).to_string(), flush=True)
    app = AppTest.from_string('''
import streamlit as st
from qqq_nq_panel import render_qqq_nq_panel
def fetch(symbol, timeframe, silent):
    return st.session_state['live_sources'][symbol]
render_qqq_nq_panel(fetch)
''', default_timeout=20)
    app.session_state['live_sources'] = frames
    app.session_state['daily_ghana_zscore_alerts'] = alerts
    app.session_state['daily_ghana_zscore_alert_timeframe'] = '1h'
    app.run().button(key='qqq_nq_generate').click().run()
    app.date_input[0].set_value(best_day).run()
    assert len(app.exception) == 0, str(app.exception)
    assert len(app.error) == 0, str(app.error)
    assert len(app.dataframe) >= 1
    assert 'VIX Daily Z' in app.dataframe[-1].value.columns
    assert len(app.get('plotly_chart')) == 1
    print('Real-data chart and table render paths: PASS', flush=True)


if __name__ == '__main__':
    main()

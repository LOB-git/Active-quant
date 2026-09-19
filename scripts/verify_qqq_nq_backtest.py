"""Run the default 30-day QQQ/NQ backtest with real candles and verify its UI."""
import importlib.util
from pathlib import Path
import sys
from datetime import timedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from streamlit.testing.v1 import AppTest
from qqq_nq_backtest import backtest_qqq_nq
from scripts.verify_qqq_nq import overview_daily_alerts


def main():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location('dashboard', root / 'Untitled-1.py')
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    frames = {}
    for symbol, tf in [('QQQ', '1h'), ('NQ=F', '1h'), ('^VIX', '1h'), ('NQ=F', '5m')]:
        frame = app_module.fetch_and_analyze(symbol, timeframe=tf, silent=True)
        if frame is None or frame.empty:
            raise RuntimeError(f'Live history unavailable: {symbol} {tf}')
        frames[(symbol, tf)] = frame
        print(symbol, tf, 'rows', len(frame), 'latest', frame.index[-1], flush=True)
    now = pd.Timestamp.now(tz='UTC')
    start, end = now.date() - timedelta(days=30), now.date()
    alerts = overview_daily_alerts(app_module, {symbol: frames[(symbol, '1h')] for symbol in ('QQQ', 'NQ=F', '^VIX')})
    result = backtest_qqq_nq(
        frames[('QQQ', '1h')], frames[('NQ=F', '1h')], frames[('NQ=F', '5m')], frames[('NQ=F', '1h')],
        start=start, end_exclusive=end + timedelta(days=1), as_of=now,
        zone_calculator=app_module.calculate_liquidity_zones,
        daily_alerts=alerts,
    )
    print('Period:', start, 'to', end, flush=True)
    print('Summary:', result['summary'], flush=True)
    audit, trades = result['audit'], result['trades']
    if not audit.empty:
        print('Decisions:', audit.groupby(['Status', 'Reason']).size().to_dict(), flush=True)
    if not trades.empty:
        print(trades[['Entry Time (Ghana)', 'Direction', 'Exit Reason', 'Net R', 'Net Points']].tail(6).to_string(), flush=True)
    app = AppTest.from_string('''
import streamlit as st
from qqq_nq_backtest_panel import render_qqq_nq_backtest
def fetch(symbol, timeframe, silent):
    return st.session_state['live_frames'][(symbol, timeframe)]
render_qqq_nq_backtest(fetch, st.session_state['zone_calculator'])
''', default_timeout=45)
    app.session_state['live_frames'] = frames
    app.session_state['zone_calculator'] = app_module.calculate_liquidity_zones
    app.session_state['daily_ghana_zscore_alerts'] = alerts
    app.session_state['daily_ghana_zscore_alert_timeframe'] = '1h'
    app.run().button(key='qqq_nq_bt_run').click().run()
    assert len(app.exception) == 0, str(app.exception)
    assert len(app.error) == 0, str(app.error)
    assert 'qqq_nq_bt_result' in app.session_state.filtered_state
    if not trades.empty:
        assert len(app.get('plotly_chart')) == 1
        assert trades['Daily Alert Confirmation'].isin(['BUY', 'SELL']).all()
        for label in ('QQQ', 'NQ', 'VIX'):
            assert trades[f'{label} Alert Time (Ghana)'].le(trades['Entry Time (Ghana)']).all()
            assert trades[f'{label} Alert Time (Ghana)'].dt.date.eq(trades['Entry Time (Ghana)'].dt.date).all()
    app.session_state['daily_ghana_zscore_alerts'] = alerts[:-1]
    app.run()
    assert any('stored daily alert source changed' in message.value for message in app.info)
    app.session_state['daily_ghana_zscore_alerts'] = alerts
    app.run()
    app.number_input(key='qqq_nq_bt_slip').set_value(1.0).run()
    assert any('settings changed' in message.value for message in app.info)
    print('Real-data backtest, equity curve, trade table and stale-setting guard: PASS', flush=True)
    output = root / 'backtest_results'
    output.mkdir(exist_ok=True)
    stamp = now.strftime('%Y%m%d_%H%M%S')
    trades.to_csv(output / f'qqq_nq_daily_filtered_{stamp}_trades.csv', index=False)
    audit.to_csv(output / f'qqq_nq_daily_filtered_{stamp}_decisions.csv', index=False)
    result['signals'].to_csv(output / f'qqq_nq_daily_filtered_{stamp}_signals.csv')
    pd.DataFrame(alerts).to_csv(output / f'qqq_nq_daily_filtered_{stamp}_source_alerts.csv', index=False)
    pd.DataFrame([dict(result['summary'], Start=start, End=end, Timeframe='1h', Threshold=10,
                       Target='Nearest 1h liquidity zone', ATR_Multiple=1., Slippage_Per_Fill_Points=.25,
                       Round_Trip_Fees_Points=.5, Daily_Alert_Timeframe='1h', Daily_Z_Threshold=1.,
                       Filter='Positive QQQ/NQ with negative VIX for BUY; inverse for SELL',
                       Source='Overview daily alert builder, reconstructed for QA')]).to_csv(
                           output / f'qqq_nq_daily_filtered_{stamp}_summary.csv', index=False)
    print('Saved verified run:', output, stamp, flush=True)


if __name__ == '__main__':
    main()

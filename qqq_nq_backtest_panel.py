"""Interactive backtest controls and results beside the QQQ/NQ signal panel."""

from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qqq_nq_backtest import backtest_qqq_nq
from qqq_nq_daily_alerts import (
    EVIDENCE_COLUMNS, stored_alert_inputs, prepare_daily_alerts, alert_source_fingerprint,
)


def backtest_equity_figure(trades, start):
    # Contract: cumulative net R through exits, Ghana time, includes initial zero.
    # One blue step line; markers preserve sparse trade histories. Costs included.
    dates = pd.to_datetime(trades['Exit Time (Ghana)'], utc=True).dt.tz_localize(None).tolist()
    dates.insert(0, pd.Timestamp(start))
    values = [0.0] + trades['Cumulative R'].tolist()
    fig = go.Figure(go.Scatter(
        x=dates, y=values, name='Net cumulative R', mode='lines+markers',
        line=dict(color='#4C78A8', shape='hv', width=2), marker=dict(size=6),
    ))
    fig.add_hline(y=0, line_color='#808080', line_width=1)
    fig.update_layout(
        title='QQQ / NQ confirmation backtest — closed-trade equity',
        xaxis_title='Exit time — Ghana (GMT/UTC)', yaxis_title='Cumulative net R', height=360,
    )
    return fig


def render_qqq_nq_backtest(fetch_history, zone_calculator):
    st.markdown('#### Backtest QQQ + NQ Confirmation Signals')
    timeframe = st.session_state.get('qqq_nq_timeframe', '1h')
    threshold = float(st.session_state.get('qqq_nq_threshold', 10.0))
    st.caption(
        f'Signal timeframe: {timeframe} · score threshold: ±{threshold:g} · traded asset: NQ=F. '
        'Uses the same QQQ/NQ function and stored 1-hour VIX/NQ/QQQ daily alert filter above. '
        'Change the scoring timeframe/threshold there.'
    )
    today = pd.Timestamp.now(tz='UTC').date()
    earliest = today - timedelta(days=59)
    c1, c2, c3 = st.columns(3)
    with c1:
        start = st.date_input('Backtest start (Ghana)', today - timedelta(days=30),
                              min_value=earliest, max_value=today, key='qqq_nq_bt_start')
    with c2:
        end = st.date_input('Backtest end (Ghana, inclusive)', today,
                            min_value=earliest, max_value=today, key='qqq_nq_bt_end')
    with c3:
        target_label = st.selectbox('Take-profit rule',
                                    ['Nearest 1-hour liquidity zone', 'Fixed reward / risk'],
                                    key='qqq_nq_bt_target')
    target_mode = 'liquidity' if target_label.startswith('Nearest') else 'fixed_rr'
    risk_col, reward_col, slip_col, fee_col = st.columns(4)
    with risk_col:
        atr_multiple = st.number_input('Stop: 5m ATR(14) multiple', .25, 10.0, 1.0, .25, key='qqq_nq_bt_atr')
    with reward_col:
        rr = st.number_input('Reward / risk', .25, 10.0, 2.0, .25,
                              disabled=target_mode == 'liquidity', key='qqq_nq_bt_rr')
    with slip_col:
        slippage = st.number_input('Slippage per fill (NQ points)', 0.0, 20.0, .25, .25, key='qqq_nq_bt_slip')
    with fee_col:
        commission = st.number_input('Round-trip fees (NQ points)', 0.0, 20.0, .5, .25, key='qqq_nq_bt_fee')
    alert_inputs = stored_alert_inputs(st.session_state)
    daily_rows, source_issue = prepare_daily_alerts(**alert_inputs)
    config = (timeframe, threshold, start, end, target_mode, atr_multiple, rr, slippage, commission,
              alert_source_fingerprint(alert_inputs))
    with st.expander('Backtest execution rules'):
        st.write(
            'Only new BUY/SELL confirmations enter. BUY goes long NQ=F; SELL goes short. '
            'Entry is the 5-minute open at the exact signal time, with assumed adverse slippage. '
            'The stop uses the simple average of 14 completed 5-minute true ranges before entry. '
            'One position stays open until stop, target or the test ends; signals during it are skipped.'
        )
        st.write(
            'Daily confirmation uses the latest stored event for each of VIX, NQ and QQQ available '
            'at or before the signal time, from that Ghana day only. BUY requires positive QQQ/NQ '
            'and negative VIX; SELL requires negative QQQ/NQ and positive VIX. '
            'Missing or conflicting alerts prevent entry. New confirmations are detected after '
            'applying this filter. The stored table is historical reconstruction, not proof of '
            'real-time delivery latency; future-dated events cannot confirm earlier signals.'
        )
        st.write(
            'Liquidity targets use only 1-hour candles completed by entry time and stay fixed. '
            'The nearest zone edge ahead of price becomes the target; no eligible zone means a skip. '
            'The fixed reward/risk option defaults to 2:1. Stop and target are checked on 5-minute bars, '
            'including the entry bar. When both are touched, the stop counts first. '
            'A stop crossed at an opening gap fills at the worse opening price. '
            'Slippage is deducted at entry and exit; the round-trip fee is deducted once.'
        )
        st.caption(
            'Cost defaults are editable assumptions. Net points represent one unit of NQ price exposure; '
            'net R divides each trade by its initial stop distance. Drawdown uses closed trades, '
            'not intratrade equity. Positions still open at the cutoff close at the last available '
            'completed 5-minute close. Missing 5-minute intervals are flagged in the trade log.'
        )

    if st.button('Run QQQ + NQ Backtest', type='primary', key='qqq_nq_bt_run'):
        st.session_state.pop('qqq_nq_bt_result', None)
        if start > end:
            st.error('The start date must be on or before the end date.')
        elif source_issue:
            st.error(source_issue)
        else:
            try:
                sources = {}
                requests = [('QQQ', timeframe), ('NQ=F', timeframe), ('NQ=F', '5m')]
                if target_mode == 'liquidity':
                    requests.append(('NQ=F', '1h'))
                with st.spinner('Loading signal and execution history, then simulating trades...'):
                    for symbol, tf in dict.fromkeys(requests):
                        frame = fetch_history(symbol, timeframe=tf, silent=True)
                        if frame is None or frame.empty:
                            raise ValueError(f'No {tf} candle history returned for {symbol}. Try again after data is available.')
                        sources[(symbol, tf)] = frame
                    result = backtest_qqq_nq(
                        sources[('QQQ', timeframe)], sources[('NQ=F', timeframe)], sources[('NQ=F', '5m')],
                        sources.get(('NQ=F', '1h')), timeframe=timeframe, threshold=threshold,
                        start=start, end_exclusive=end + timedelta(days=1),
                        zone_calculator=zone_calculator, atr_multiple=atr_multiple,
                        target_mode=target_mode, reward_risk=rr,
                        slippage_points=slippage, commission_points=commission,
                        **alert_inputs,
                    )
                st.session_state['qqq_nq_bt_result'] = {
                    'config': config, 'result': result, 'run_at': pd.Timestamp.now(tz='UTC'),
                }
            except Exception as exc:
                st.error(f'The backtest could not run: {exc}')

    stored = st.session_state.get('qqq_nq_bt_result')
    if not stored:
        st.info('Choose the Ghana date range and click Run QQQ + NQ Backtest to evaluate profitability.')
        return
    if stored['config'] != config:
        st.info('Backtest settings changed or the stored daily alert source changed. Run again to update the results.')
        return
    result = stored['result']
    summary, trades, audit = result['summary'], result['trades'], result['audit']
    st.caption(
        f'Tested {start:%d %b %Y} through {end:%d %b %Y} Ghana · '
        f'5m history: {result["execution_start"]:%d %b %H:%M} to '
        f'{result["execution_end"]:%d %b %H:%M} · run {stored["run_at"]:%d %b %H:%M} Ghana.'
    )
    st.caption('Signal warmup uses the fetched history before the start date; unavailable factors cannot trigger trades.')
    scoped_alerts = daily_rows.loc[
        daily_rows['Ghana Time'].ge(pd.Timestamp(start, tz='UTC'))
        & daily_rows['Ghana Time'].lt(result['effective_end'])
    ]
    st.caption(
        f'Stored daily alerts in this range: {len(scoped_alerts)}. '
        'Days/assets without a stored alert cannot confirm trades; absence is not a neutral reading.'
    )
    if result['execution_start'].date() > start or result['execution_end'] < result['effective_end'] - pd.Timedelta(minutes=5):
        st.info('The execution feed does not cover every requested boundary. See the recorded coverage and skipped signals below.')
    metrics = st.columns(6)
    metrics[0].metric('Trades', summary['Trades'])
    metrics[1].metric('Win rate', f'{summary["Win Rate %"]:.1f}%' if len(trades) else '—')
    metrics[2].metric('Net R', f'{summary["Net R"]:+.2f}R')
    metrics[3].metric('Net NQ points', f'{summary["Net Points"]:+,.2f}')
    pf = summary['Profit Factor (points)']
    metrics[4].metric('Profit factor', '—' if pd.isna(pf) else ('∞' if np.isinf(pf) else f'{pf:.2f}'))
    metrics[5].metric('Closed drawdown', f'{summary["Max Closed Drawdown R"]:.2f}R')
    st.write(f'New confirmations: {summary["New confirmations"]} · skipped: {summary["Skipped"]}.')
    st.caption('This filter can leave very few trades. A small historical sample does not establish signal accuracy or future profitability.')
    if trades.empty:
        st.info('No trades completed under these settings. The decision log explains any skipped confirmations.')
    else:
        if summary['Trades with bar gaps']:
            st.warning(
                f'{summary["Trades with bar gaps"]} trade(s) span gaps in the 5-minute data. '
                'Price movements inside those gaps cannot be checked; inspect the flagged trades.'
            )
        st.plotly_chart(backtest_equity_figure(trades, start), width='stretch', key='qqq_nq_bt_equity')
        st.markdown('##### NQ backtest trades — Ghana time')
        st.dataframe(trades.style.format({
            name: '{:,.2f}' for name in trades.select_dtypes(include='number').columns if name != 'Trade'
        }, na_rep='—'), width='stretch', hide_index=True)
        st.download_button('Download backtest trades', trades.to_csv(index=False).encode(),
                            f'QQQ_NQ_backtest_{start}_{end}_{timeframe}.csv', 'text/csv', key='qqq_nq_bt_export')
    with st.expander('Signal decision log — traded and skipped'):
        if audit.empty:
            st.write('No new BUY/SELL confirmations fell inside the selected Ghana dates.')
        else:
            st.dataframe(audit, width='stretch', hide_index=True)
        if not result['signals'].empty:
            st.write('All confirmation states in the backtest range:')
            st.dataframe(result['signals']['Signal'].value_counts().rename('Candles'), width='stretch')
            columns = ['Signal Time (Ghana)', 'Base Signal', 'Signal', 'New Alert',
                       'Daily Alert Confirmation', *EVIDENCE_COLUMNS, 'Reason', 'Daily Alert Reason']
            evidence = result['signals'][columns]
            st.dataframe(evidence, width='stretch', hide_index=True)
            st.download_button('Download all signal decisions (including WAIT)',
                               evidence.to_csv(index=False).encode(),
                               f'QQQ_NQ_daily_confirmations_{start}_{end}.csv', 'text/csv',
                               key='qqq_nq_bt_signals_export')

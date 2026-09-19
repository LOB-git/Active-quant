"""QQQ/NQ signal view, rendered immediately above the US Zone Battle section."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qqq_nq_signals import build_qqq_nq_signals, signals_on_ghana_day
from qqq_nq_daily_alerts import EVIDENCE_COLUMNS, prepare_daily_alerts, stored_alert_inputs


def confirmation_figure(day, threshold):
    # Chart contract: compare independent direction scores and their confirmation
    # through one Ghana day. Native Plotly, full width, fixed -100..100 scale;
    # blue/orange asset lines, neutral confirmation, marker shape + dash distinction.
    # Sparse days use points instead of implying a continuous trend.
    figure = go.Figure()
    x = pd.to_datetime(day['Signal Time (Ghana)'], utc=True).dt.tz_localize(None)
    for column, color, dash in [
        ('QQQ Score', '#4C78A8', 'solid'),
        ('NQ Score', '#F28E2B', 'dash'),
        ('Confirmation Score', '#B8BCC6', 'dot'),
    ]:
        figure.add_trace(go.Scatter(
            x=x, y=day[column], name=column,
            mode='lines+markers' if day[column].notna().sum() >= 8 else 'markers',
            line=dict(color=color, dash=dash, width=2),
            connectgaps=False, marker=dict(size=5),
            hovertemplate='%{x|%d %b %H:%M} Ghana<br>Score: %{y:+.2f}<extra>%{fullData.name}</extra>',
        ))
    for signal, shape, color in [('BUY', 'triangle-up', '#4C78A8'), ('SELL', 'triangle-down', '#F28E2B')]:
        mask = day['Signal'].eq(signal) & day['New Alert']
        figure.add_trace(go.Scatter(
            x=x[mask], y=day.loc[mask, 'Confirmation Score'], mode='markers',
            name=f'New {signal} alert', marker=dict(symbol=shape, size=13, color=color),
        ))
    for value in (-threshold, 0, threshold):
        figure.add_hline(y=value, line_color='#808080', line_dash='dash', line_width=1)
    figure.update_layout(
        title='QQQ and NQ demand/supply confirmation',
        xaxis_title='Signal time — Ghana (GMT/UTC)', yaxis_title='Direction score',
        yaxis=dict(range=[-105, 105]), height=440,
        legend=dict(orientation='h', y=1.18), margin=dict(t=100),
    )
    return figure


def render_qqq_nq_panel(fetch_history):
    st.divider()
    st.subheader('QQQ + NQ Quantitative Confirmation Signals')
    st.caption(
        'Compare the demand/supply formula from the historical Ghana-day alerts for QQQ and '
        'NQ=F, confirmed by VIX, NQ=F and QQQ from the stored Daily Historical Z-Score Alerts '
        '— Ghana Time table. The daily alert monitor uses 1-hour candles.'
    )
    controls = st.columns(2)
    with controls[0]:
        timeframe = st.selectbox(
            'QQQ / NQ timeframe', ['5m', '15m', '1h', '4h'], index=2, key='qqq_nq_timeframe'
        )
    with controls[1]:
        threshold = st.number_input(
            'Minimum direction score (each asset)', min_value=1.0, max_value=100.0,
            value=10.0, step=1.0, key='qqq_nq_threshold',
        )
    if st.button('Generate QQQ + NQ Signals', type='primary', key='qqq_nq_generate'):
        histories, errors = {}, []
        with st.spinner('Loading QQQ and NQ=F candle histories...'):
            for symbol in ('QQQ', 'NQ=F'):
                try:
                    frame = fetch_history(symbol, timeframe=timeframe, silent=True)
                    if frame is None or frame.empty:
                        errors.append(f'{symbol}: no candle history returned')
                    else:
                        histories[symbol] = frame.copy()
                except Exception as exc:
                    errors.append(f'{symbol}: {exc}')
        # Replace the previous attempt even when one source fails: never show it as fresh.
        st.session_state['qqq_nq_source'] = {
            'timeframe': timeframe, 'histories': histories, 'errors': errors,
            'loaded_at': pd.Timestamp.now(tz='UTC'),
        }

    with st.expander('Quantitative function and signal rules'):
        st.latex(r'S_i(t)=100\left[0.60\tanh(\Delta F_i)+0.25\tanh(\Delta M_i)'
                 r'+0.15\,\mathbf{1}_{\Delta V_i>0}\,\mathrm{sign}(\Delta P_i)'
                 r'\tanh(|\Delta V_i|)\right]')
        st.latex(r'C(t)=\begin{cases}\mathrm{sign}(S_Q)\min(|S_Q|,|S_N|), & S_Q S_N>0'
                 r'\\0, & S_Q S_N\leq0\end{cases}')
        st.write(
            'F, M and V are the historical Flow, Momentum and Volatility Z components. '
            'Changes are measured against the preceding candle in each asset’s own history. '
            'The 20-candle flow ratio and 50-candle Z windows match Historical Z-Score Component Analysis. '
            'Both assets must have a Flow Z rise/drop event and all scoring factors available.'
        )
        st.write(
            'BUY: both direction scores ≥ the threshold, plus positive QQQ and NQ daily alerts '
            'and a negative VIX daily alert. SELL: both scores ≤ minus the threshold, plus '
            'negative QQQ and NQ daily alerts and a positive VIX daily alert. '
            'CONFLICT: scores have opposite signs. WAIT: incomplete data or insufficient agreement. '
            'The plotted confirmation score is the weaker base direction score; BUY/SELL markers '
            'also require the daily alert filter. '
            'New Alert marks the start of a confirmed run; the table retains every candle.'
        )
        st.write(
            'Daily source: the exact stored US Indices table, not a separate Z-score calculation. '
            'It keeps the first positive and first negative event each day with aligned Momentum Z. '
            'At each signal time, use the latest already-available event for each asset on that Ghana '
            'day; a later event replaces the earlier direction. Nothing carries into the next day. '
            'Daily alert timestamps already represent candle close. No future alerts are used. '
            'These are event states, not continuously updated Z-score readings. '
            'If an alert arrives between analysis candles, confirmation is evaluated at the next '
            'completed, matching QQQ/NQ candle, not backdated to that alert.'
        )
        st.caption(
            'Use the backtest below to evaluate this confirmation rule. '
            'Scores are model values, not probabilities. Adding confirmation does not guarantee '
            'higher accuracy or profitability.'
        )
        st.markdown(
            'QQQ is a [Nasdaq-100 ETF](https://www.invesco.com/qqq-etf/en/home.html); '
            'NQ is a [Nasdaq-100 futures contract](https://www.cmegroup.com/markets/equities/nasdaq.html). '
            'Their trading activity and candle coverage differ, so their rolling flow scores can differ.'
        )

    alert_inputs = stored_alert_inputs(st.session_state)
    daily_rows, source_issue = prepare_daily_alerts(**alert_inputs)
    if source_issue:
        st.warning(source_issue + ' BUY/SELL signals remain disabled until this source is ready.')
    else:
        st.caption(
            f'Daily source: {len(daily_rows)} stored QQQ/NQ/VIX events · 1h · '
            f'Z and Momentum Z threshold: ±{float(alert_inputs["alert_threshold"]):g}. '
            'Refresh Indices Data updates this source; Generate QQQ + NQ Signals reloads scoring candles.'
        )
        built_at = st.session_state.get('daily_ghana_zscore_alert_built_at')
        if built_at is not None:
            st.caption(f'Daily alert source cutoff: {pd.Timestamp(built_at):%d %b %Y %H:%M} Ghana.')
    source = st.session_state.get('qqq_nq_source')
    if not source:
        st.info('Select a timeframe and click Generate QQQ + NQ Signals here to load both assets.')
        return
    if source['timeframe'] != timeframe:
        st.info(f'Select Generate QQQ + NQ Signals to load the chosen {timeframe} timeframe.')
        return
    if source['errors']:
        st.error('Unable to compare both assets. ' + '; '.join(source['errors']))
        return
    try:
        signals = build_qqq_nq_signals(
            source['histories']['QQQ'], source['histories']['NQ=F'],
            timeframe, threshold, as_of=source['loaded_at'], **alert_inputs,
        )
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f'The QQQ/NQ history could not be analyzed: {exc}')
        return
    if signals.empty:
        st.info('No completed candles are available for this comparison.')
        return
    dates = sorted(set(pd.to_datetime(signals['Signal Time (Ghana)'], utc=True).dt.date))
    selected = st.date_input(
        'QQQ / NQ Ghana date', value=dates[-1], min_value=dates[0], max_value=dates[-1],
        key=f'qqq_nq_date_{timeframe}_{dates[0]}_{dates[-1]}',
    )
    day = signals_on_ghana_day(signals, selected)
    st.caption(
        f'{selected:%d %b %Y} · {timeframe} · retrieved {source["loaded_at"]:%d %b %H:%M} Ghana. '
        'Signal time = candle timestamp + timeframe. Only completed candles are evaluated. '
        'Daily Z-score alerts also use candle-close time. The separate demand/supply rise/drop '
        'table labels candle starts; compare that table using Candle Time below.'
    )
    if day.empty:
        st.info('No completed candles fall on this Ghana calendar day. Choose another date.')
        return
    matched = day['QQQ Ready'].eq(True) & day['NQ Ready'].eq(True)
    summary = st.columns(4)
    summary[0].metric('Comparable candles', int(matched.sum()))
    summary[1].metric('BUY candles', int(day['Signal'].eq('BUY').sum()))
    summary[2].metric('SELL candles', int(day['Signal'].eq('SELL').sum()))
    summary[3].metric('Conflicts', int(day['Signal'].eq('CONFLICT').sum()))
    latest = day.iloc[-1]
    st.write(f'Last available state on this date: **{latest["Signal"]}** — {latest["Reason"]}.')
    blocked = day['Base Signal'].isin(['BUY', 'SELL']) & day['Signal'].eq('WAIT')
    st.caption(f'{int(blocked.sum())} base BUY/SELL candle(s) held at WAIT by the daily alert filter.')
    with st.expander('Stored VIX / NQ / QQQ daily alerts — selected Ghana day'):
        if daily_rows.empty:
            st.info(source_issue or 'No stored alerts are available.')
        else:
            source_day = daily_rows.loc[
                daily_rows['Ghana Time'].dt.date.eq(selected)
                & daily_rows['Ghana Time'].le(source['loaded_at'])
            ]
            if source_day.empty:
                st.info('No completed stored daily alerts for these assets on this date.')
            else:
                st.dataframe(source_day, width='stretch', hide_index=True)
            st.caption('This is the source-day preview. Each signal below uses only the rows available by its own timestamp.')
    if not matched.any():
        st.info(
            'No comparable QQQ/NQ candles with valid factors on this date. '
            'Check the timestamps and reasons below. Different hourly candle grids can prevent '
            'exact matching; try 15m for finer matching. More history is needed during Z-score warmup.'
        )
    if day[['QQQ Score', 'NQ Score']].notna().any().any():
        st.plotly_chart(confirmation_figure(day, threshold), width='stretch', key='qqq_nq_chart')
    st.markdown('##### All QQQ / NQ signals — selected Ghana day')
    columns = [
        'Candle Time (Ghana)', 'Signal Time (Ghana)', 'Signal', 'New Alert', 'Base Signal',
        'QQQ Score', 'NQ Score', 'Confirmation Score', 'QQQ Flow Event', 'NQ Flow Event',
        'Daily Alert Confirmation', *EVIDENCE_COLUMNS,
        'QQQ Flow Z Change', 'NQ Flow Z Change', 'QQQ Price', 'NQ Price', 'Reason', 'Daily Alert Reason',
    ]
    table = day.reset_index()[columns]
    numeric = table.select_dtypes(include='number').columns
    st.dataframe(
        table.style.format({col: '{:+.2f}' for col in numeric}, na_rep='Unavailable'),
        width='stretch', hide_index=True, height=min(650, 38 * len(table) + 40),
    )
    st.download_button(
        'Download this day’s QQQ / NQ signals', table.to_csv(index=False).encode('utf-8'),
        file_name=f'QQQ_NQ_signals_{selected}_{timeframe}.csv', mime='text/csv', key='qqq_nq_download',
    )

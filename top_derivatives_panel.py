"""Top 100 Binance derivatives scan with the US Historical Z-Score Component Analysis."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from derivative_crypto_panel import format_price
from alert_quant_verdict import render_alert_quant_verdict
from historical_flow_analysis import analyze_ghana_flow_events, highest_flow_zone


SCAN_SIZE = 100
SCAN_COLUMNS = [
    'Rank', 'symbol', 'price', 'momentum', 'z_score', 'rsi_15m', 'funding_rate', 'open_interest',
    '24h_volume', 'money_flow_signal', 'vol_ratio', 'tps', 'entry_score', 'entry_signal', 'quantum_verdict',
]
ALERT_COLUMNS = [
    'Asset', 'Date', 'Timeframe', 'Alert', 'Ghana Time', 'Flow Z', 'Flow Z Change', 'Momentum Z Change',
    'Trend Z Change', 'Volatility Z Change', 'Price Change', 'Zone Bottom', 'Zone Top',
    'Direction Score', 'Demand/Supply Alert',
]
ZONE_COLUMNS = [
    'Asset', 'Ghana Time', 'Zone Bottom', 'Zone Top', 'Close', 'Flow Z Change', 'Momentum Z Change',
    'Volatility Z Change', 'Price Change', 'Flow Contribution', 'Momentum Contribution',
    'Volatility Contribution',
]


def rank_scan(results):
    """Order scan results by 24h USDT volume, the basis of the top 100 selection."""
    ranked = results.sort_values('24h_volume', ascending=False, kind='stable').reset_index(drop=True)
    ranked.insert(0, 'Rank', np.arange(1, len(ranked) + 1))
    return ranked


def flow_alert_label(score):
    """US alert-table wording for the shared -100..100 direction score."""
    if not np.isfinite(score):
        return 'Insufficient factor data'
    if score >= 10:
        return 'Demand Alert'
    if score <= -10:
        return 'Supply Alert'
    return 'Unresolved Alert'


def component_drop_events(components):
    """Every downward move of each component within the selected day, latest first."""
    events = []
    for column in components.columns:
        series = components[column].dropna()
        changes = series.diff().dropna()
        for moment in changes[changes < 0].index:
            previous, current = float(series.shift(1).loc[moment]), float(series.loc[moment])
            events.append({
                'Component': column, 'Drop Time': pd.Timestamp(moment), 'Previous Value': previous,
                'Current Value': current, 'Drop Size': previous - current,
            })
    if not events:
        return pd.DataFrame(columns=['Component', 'Drop Time', 'Previous Value', 'Current Value', 'Drop Size'])
    return pd.DataFrame(events).sort_values('Drop Time', ascending=False).head(50)


def build_historical_z_view(history, asset, timeframe, selected_date):
    """Calculate the US section for one Ghana day from the asset's complete history.

    Alerts compare the day's first candle with the preceding candle. The highest rise
    and drop use moves within the selected day, as the US section does.
    """
    components, _, alerts = analyze_ghana_flow_events(history, asset, selected_date)
    zone_events = alerts
    if not alerts.empty:
        alerts = alerts.copy()
        alerts['Timeframe'] = timeframe
        alerts['Demand/Supply Alert'] = alerts['Direction Score'].map(flow_alert_label)
        zone_events = alerts.loc[alerts['Ghana Time'] != components.index[0]]
    return {
        'components': components,
        'alerts': alerts,
        'drops': component_drop_events(components),
        'highest_drop': highest_flow_zone(zone_events, 'drop'),
        'highest_rise': highest_flow_zone(zone_events, 'rise'),
    }


def render_zone(zone, label, caption):
    st.subheader(f'Demand & Supply Analysis — Highest Flow Z {label}')
    st.caption(caption)
    metrics = st.columns(5)
    metrics[0].metric('Classification', zone['Classification'])
    metrics[1].metric('Direction Score', f"{zone['Direction Score']:+.1f}")
    metrics[2].metric('Analyzed Component', 'Flow (z)')
    metrics[3].metric(f'{label} Size', f"{zone['Event Size']:.2f}σ")
    metrics[4].metric('Volatility Regime', zone['Volatility Regime'])
    st.dataframe(
        pd.DataFrame([zone])[ZONE_COLUMNS].style.format({
            'Zone Bottom': format_price, 'Zone Top': format_price, 'Close': format_price,
            'Flow Z Change': '{:+.3f}', 'Momentum Z Change': '{:+.3f}', 'Volatility Z Change': '{:+.3f}',
            'Price Change': '{:+.6f}', 'Flow Contribution': '{:+.1f}', 'Momentum Contribution': '{:+.1f}',
            'Volatility Contribution': '{:+.1f}',
        }),
        width='stretch', hide_index=True,
    )


def render_historical_z_section(fetch_history, color_metrics, symbols):
    state = st.session_state
    st.divider()
    st.subheader('Historical Z-Score Component Analysis')
    st.caption(
        'This chart shows the historical Z-scores for each component of the QS score for a single asset. '
        'This helps identify which factors are strengthening or weakening over time. A value of +2 means '
        'the factor is 2 standard deviations above its own recent history.'
    )
    if state.get('td_asset') not in symbols:
        state.pop('td_asset', None)
    asset = st.selectbox('Select Asset for Z-Score Chart', symbols, key='td_asset')
    timeframe = st.selectbox(
        'Select timeframe for historical components', ['5m', '15m', '1h', '4h', '12h', '1d'],
        index=2, key='td_z_timeframe',
    )
    config = (asset, timeframe)
    requested = st.button(f'Generate Historical Z-Chart for {asset}', key='td_z_generate')
    if requested or state.get('td_z_config') != config:
        with st.spinner(f'Calculating historical Z-scores for {asset}...'):
            history = fetch_history(asset, timeframe=timeframe, silent=True)
        if history is None or history.empty:
            state.pop('td_z_data', None)
            state.pop('td_z_config', None)
            st.error(f'Could not fetch Binance data to generate the Z-chart for {asset}.')
            return
        state['td_z_data'], state['td_z_config'] = history, config
    if state.get('td_z_config') != config:
        return

    history = state['td_z_data']
    dates = sorted(np.unique(pd.DatetimeIndex(history.index).date))
    with st.popover('Select analysis date'):
        selected = st.date_input(
            'Date to analyze', value=dates[-1], min_value=dates[0], max_value=dates[-1],
            key=f'td_z_date_{asset}_{timeframe}',
        )
    st.caption(f'Showing component analysis for {selected:%B %d, %Y}.')
    view = build_historical_z_view(history, asset, timeframe, selected)
    components = view['components']
    if components.empty:
        st.info('No historical Z-score data is available for the selected date.')
        return
    if components.notna().sum().sum() == 0:
        st.info(f'{asset} does not yet have enough {timeframe} history for its 50-candle Z-scores on this date.')
        return

    figure = go.Figure()
    for column in components.columns:
        figure.add_trace(go.Scatter(x=components.index, y=components[column], mode='lines', name=column))
    figure.add_hline(y=0, line_width=1, line_dash='dash', line_color='grey')
    figure.update_layout(
        title=f'Historical Z-Score Components for {asset} — {selected:%Y-%m-%d}',
        yaxis_title='Z-Score (Standard Deviations from Mean)', template='plotly_dark',
    )
    st.plotly_chart(figure, width='stretch', key='td_z_chart')

    st.subheader('All Historical Demand & Supply Rise/Drop Alerts — Ghana Day')
    st.caption(
        f'All Flow Z rise and drop events for {asset} during the Ghana calendar day {selected:%Y-%m-%d}, '
        f'using {timeframe} candles. The first event after Ghana midnight is compared with the preceding candle.'
    )
    alerts = view['alerts']
    if alerts.empty:
        st.info('No valid Flow Z rise or drop alert exists for this asset during the selected date.')
    else:
        counts = alerts['Alert'].value_counts()
        metrics = st.columns(3)
        metrics[0].metric('Total Alerts', len(alerts))
        metrics[1].metric('Rise Alerts', int(counts.get('Flow Z Rise Alert', 0)))
        metrics[2].metric('Drop Alerts', int(counts.get('Flow Z Drop Alert', 0)))
        changes = ['Flow Z', 'Flow Z Change', 'Momentum Z Change', 'Trend Z Change',
                   'Volatility Z Change', 'Price Change', 'Direction Score']
        st.dataframe(
            alerts.sort_values('Ghana Time')[ALERT_COLUMNS].style.format({
                'Flow Z': '{:+.2f}', 'Flow Z Change': '{:+.2f}', 'Momentum Z Change': '{:+.2f}',
                'Trend Z Change': '{:+.2f}', 'Volatility Z Change': '{:+.2f}', 'Price Change': '{:+,.6f}',
                'Zone Bottom': format_price, 'Zone Top': format_price, 'Direction Score': '{:+.1f}',
            }).map(color_metrics, subset=changes),
            width='stretch', hide_index=True,
        )
        render_alert_quant_verdict(history, alerts, selected, 'td_aq', price_format=format_price)

    drops = view['drops']
    if drops.empty:
        st.info('No downward moves were detected on the selected date.')
    else:
        table = drops.copy()
        table['Drop Time'] = table['Drop Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.subheader('Drop Events for Selected Date (up to 50)')
        st.dataframe(
            table.style.format({'Previous Value': '{:.3f}', 'Current Value': '{:.3f}', 'Drop Size': '{:.3f}'}),
            width='stretch', hide_index=True,
        )
        if view['highest_drop'] is None:
            st.info('No downward Flow Z event was detected on the selected date, so no demand/supply zone can be classified from Flow.')
        else:
            render_zone(view['highest_drop'], 'Drop', (
                'Zone Direction Score = 60% Flow Z change + 25% Momentum Z change + 15% directional '
                'volatility regime. The candle range is a candidate zone, not order-book proof.'
            ))
    if view['highest_rise'] is None:
        st.info('No upward Flow Z event was detected on the selected date, so no rise-event demand/supply zone can be classified.')
    else:
        render_zone(view['highest_rise'], 'Rise', (
            'The largest upward Flow Z event is the primary demand candidate. Momentum Z and directional '
            'volatility provide confirmation using the same 60% / 25% / 15% weighting.'
        ))


def render_top_derivatives_tab(scan_assets, fetch_history, color_metrics, format_large_number):
    state = st.session_state
    st.subheader('🔭 Top 100 Binance Derivatives Scan')
    st.write(
        'Scans the 100 most active Binance USD-M crypto perpetuals by 24-hour USDT volume. '
        'Select a row or use the asset list to run the Historical Z-Score analysis below.'
    )
    timeframe = st.selectbox('Select scan timeframe', ['5m', '15m', '1h', '4h'], index=2, key='td_scan_timeframe')
    if st.button('Scan Top 100 Derivatives', type='primary', key='td_scan_run'):
        with st.spinner('Scanning the top 100 Binance perpetuals. This can take a few minutes...'):
            results = scan_assets(timeframe=timeframe, top_n=SCAN_SIZE)
        if results is not None and not results.empty:
            state['td_scan'] = {
                'results': rank_scan(results), 'timeframe': timeframe, 'scanned_at': pd.Timestamp.now(tz='UTC'),
            }
            state.pop('td_last_row', None)
        elif state.get('td_scan'):
            st.warning('The scan returned no data. The table below is the previous scan; see its time.')
        else:
            st.warning('The scan returned no data from Binance. Try again in a moment.')
    scan = state.get('td_scan')
    if not scan:
        st.info('Click Scan Top 100 Derivatives to unlock the Historical Z-Score Component Analysis.')
        return

    ranked = scan['results']
    st.caption(
        f"{len(ranked)} of {SCAN_SIZE} assets returned · {scan['timeframe']} candles · "
        f"scanned {scan['scanned_at']:%d %b %Y %H:%M} Ghana."
    )
    if len(ranked) < SCAN_SIZE:
        st.warning(
            f'{SCAN_SIZE - len(ranked)} asset(s) could not be analyzed, usually because a new listing has '
            'too little history or Binance rate-limited the request.'
        )
    columns = [column for column in SCAN_COLUMNS if column in ranked.columns]
    selection = st.dataframe(
        ranked[columns].style.format({
            'price': format_price, 'momentum': '{:.2%}', 'z_score': '{:+.2f}', 'rsi_15m': '{:.1f}',
            'funding_rate': '{:.4%}', 'open_interest': format_large_number, '24h_volume': format_large_number,
            'money_flow_signal': '{:+.2f}', 'vol_ratio': '{:.2f}x', 'tps': '{:+.2f}', 'entry_score': '{:+.2f}',
        }, na_rep='n/a'),
        width='stretch', hide_index=True, height=420, on_select='rerun', selection_mode='single-row',
        key=f"td_scan_table_{scan['scanned_at'].value}",
    )
    rows = getattr(getattr(selection, 'selection', None), 'rows', [])
    # Apply only a new row click, so the asset list still works while a row stays highlighted.
    if rows and rows[0] != state.get('td_last_row') and 0 <= rows[0] < len(ranked):
        state['td_last_row'] = rows[0]
        state['td_asset'] = str(ranked.iloc[rows[0]]['symbol'])
    render_historical_z_section(
        fetch_history, color_metrics, ranked['symbol'].dropna().astype(str).tolist(),
    )

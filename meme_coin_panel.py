"""US Indices tab logic applied to DEX meme coins in the Meme Coins tab.

Dexscreener chooses the coins and GeckoTerminal supplies their candles. The QQQ/NQ
confirmation and backtest engines are reused unchanged through a symbol adapter: two
chosen coins take the QQQ and NQ roles, and Binance BTCDOM takes the inverse VIX role.

Every section loads on a button because Streamlit runs all tabs on each rerun and
the candle API allows only a few calls per minute.
"""

import math
import re
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from alert_quant_verdict import render_alert_quant_verdict
from binance_futures import STABLE_BASES
from derivative_crypto_panel import (
    add_qs_scores, build_daily_alerts, build_overview_row, color_bracketed_change,
    combine_metric_value, gex_figure, new_threshold_alerts, order_flow_summary,
    qs_factor_row, with_flow_history, with_volume_delta,
)
from dex_candles import asset_id, last_errors
from historical_flow_analysis import analyze_ghana_flow_events, highest_flow_zone
from qqq_nq_backtest import backtest_qqq_nq
from qqq_nq_backtest_panel import backtest_equity_figure
from qqq_nq_daily_alerts import EVIDENCE_COLUMNS, alert_source_fingerprint, prepare_daily_alerts
from qqq_nq_signals import build_qqq_nq_signals, signals_on_ghana_day


GAUGE = 'BTCDOM/USDT'  # Rising BTC dominance is the inverse gauge, as VIX is for QQQ/NQ.
BASKET_SIZES = [3, 5, 7, 10]
TOP_COINS = 10
MIN_CANDLES = 80  # The 30-bar momentum plus its 50-bar Z-score. Newer pools cannot be scored.
FLOW_FREQUENCY = {'5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h'}
# Enough signal-timeframe warmup for the Z-scores without spending extra API pages.
WARMUP_DAYS = {'5m': 2, '15m': 5, '1h': 15, '4h': 45}
BACKTEST_HISTORY_DAYS = 40  # Stored daily alerts span one page: 1000 1h bars, about 41 days.
ALERTS_KEY = 'mc_daily_alerts'
ALERT_TF_KEY = 'mc_daily_alert_timeframe'
ALERT_THRESHOLD_KEY = 'mc_daily_alert_threshold'
PRICE_LIKE_TRADE_COLUMNS = {
    'Entry Price', 'ATR14', 'Risk Points', 'Stop Price', 'Target Price', 'Target Zone Bottom',
    'Target Zone Top', 'Exit Price', 'Gross Points', 'Cost Points', 'Net Points',
}


def format_meme_price(value):
    """Keep four significant figures: many coins trade far below a millionth of a dollar."""
    if value is None or pd.isna(value):
        return 'n/a'
    number = float(value)
    magnitude = abs(number)
    if magnitude >= 1:
        return f'${number:,.2f}' if magnitude >= 1000 else f'${number:,.4f}'
    if magnitude == 0:
        return '$0.00'
    return f'${number:.{min(3 - math.floor(math.log10(magnitude)), 18)}f}'


def format_price_change(value):
    if value is None or pd.isna(value):
        return 'n/a'
    text = format_meme_price(abs(float(value)))
    return f"{'-' if value < 0 else '+'}{text}"


def label_pairs(coins):
    """Name pairs uniquely: one symbol can trade on several chains and pools."""
    labels = coins['Asset'].astype(str).copy()
    repeated = labels.duplicated(keep=False)
    labels[repeated] = labels[repeated] + ' (' + coins.loc[repeated, 'Chain'].astype(str) + ')'
    repeated = labels.duplicated(keep=False)
    labels[repeated] = labels[repeated] + ' ' + coins.loc[repeated, 'Pair Address'].astype(str).str[:6]
    return {
        label: asset_id(row['Chain'], row['Pair Address'], row['Token Address'])
        for label, (_, row) in zip(labels, coins.iterrows())
    }


def eligible_pairs(overview):
    if overview is None or overview.empty:
        return pd.DataFrame()
    bases = overview['Asset'].astype(str).str.split('/').str[0].str.upper()
    has_addresses = overview['Pair Address'].astype(bool) & overview['Token Address'].astype(bool)
    return overview.loc[~bases.isin(STABLE_BASES) & has_addresses]


def build_basket(overview, size):
    """The deepest pools make the steadiest candles, so rank the basket by liquidity."""
    coins = eligible_pairs(overview)
    if coins.empty:
        return {}
    return label_pairs(coins.sort_values('Liquidity USD', ascending=False, kind='stable').head(int(size)))


def top_by_volume(overview, count=TOP_COINS):
    coins = eligible_pairs(overview)
    if coins.empty:
        return {}
    return label_pairs(coins.sort_values('Volume USD', ascending=False, kind='stable').head(int(count)))


def resolve_basket(overview, size, source_key, refreshed, state):
    """Hold the basket steady between refreshes so feed churn cannot swap coins mid-analysis."""
    key = (size, source_key)
    saved = state.get('mc_basket')
    if saved is None or refreshed or state.get('mc_basket_key') != key:
        built = build_basket(overview, size)
        if built or saved is None or state.get('mc_basket_key') != key:
            saved = built
        state['mc_basket'] = saved
        state['mc_basket_key'] = key
    return saved


def choose(container, label, options, key, index=0):
    """A selectbox that survives its saved choice leaving the option list."""
    options = list(options)
    if key in st.session_state and st.session_state[key] not in options:
        st.session_state.pop(key)
    return container.selectbox(label, options, index=min(index, len(options) - 1), key=key)


def load_candles(deps, asset, timeframe, limit=None, start_date=None):
    """Return (candles, None), or (None, reason). Indicators keep their full warmup."""
    options = {'timeframe': timeframe, 'silent': True}
    if start_date:
        options['start_date'] = start_date
    fetch = deps.fetch_gauge if asset == GAUGE else deps.fetch_candles
    frame = fetch(asset, **options)
    if frame is None or frame.empty:
        return None, last_errors.get(asset) or 'no candles returned'
    if not start_date and len(frame) < MIN_CANDLES:
        return None, f'only {len(frame)} {timeframe} candles; {MIN_CANDLES} needed'
    return (frame.tail(int(limit)) if limit else frame), None


def load_basket(deps, coins, timeframe, limit=None):
    frames, failures = {}, []
    for label, asset in coins.items():
        frame, reason = load_candles(deps, asset, timeframe, limit)
        if frame is None:
            failures.append(f'{label} ({reason})')
        else:
            frames[label] = frame
    return frames, failures


def spinner_text(what, count):
    """Say why a load is slow: the public API allows about five calls per minute."""
    return f'{what} for {count} pair(s). Uncached pairs can take about 12 seconds each...'


def show_failures(failures):
    if failures:
        st.warning('Not loaded: ' + '; '.join(failures) + '. These will be retried on refresh.')


def daily_flow_row(symbol, candles, now=None):
    """Today's flow per timeframe, all built from one 5m history to save API calls."""
    moment = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    moment = moment.tz_localize('UTC') if moment.tzinfo is None else moment.tz_convert('UTC')
    today = candles.loc[pd.DatetimeIndex(candles.index).normalize() == moment.tz_localize(None).normalize()]
    today = today.assign(notional=today['volume'] * today['close'])
    row = {'Symbol': symbol}
    for interval, frequency in FLOW_FREQUENCY.items():
        bars = today.resample(frequency).agg({'open': 'first', 'close': 'last', 'notional': 'sum'})
        is_up = bars['close'] >= bars['open']
        inflow = float(bars.loc[is_up, 'notional'].sum())
        outflow = float(bars.loc[~is_up, 'notional'].sum())
        row[f'{interval} Inflow'] = inflow
        row[f'{interval} Outflow'] = outflow
        row[f'{interval} Net Flow'] = inflow - outflow
    return row


def scaled_gex_proxy(build_proxy, price, volatility):
    """Run the proxy at a normal price, then scale back.

    The proxy floors its bell-curve width at 1.0, which would flatten the profile of
    any coin priced far below a dollar.
    """
    reference = 1000.0
    profile = build_proxy(reference, volatility)
    scale = price / reference
    scaled = profile * scale
    scaled.index = profile.index * scale
    scaled.index.name = 'strike'
    return scaled


def relabel_text(value, asset_a, asset_b):
    """Translate engine wording so the UI never reports a meme signal as QQQ/NQ/VIX."""
    labels = {'QQQ': asset_a, 'NQ': asset_b, 'VIX': 'BTCDOM'}
    text = str(value).replace('Refresh Indices Data', 'Refresh Meme Overview')
    return re.sub(r'\b(QQQ|NQ|VIX)\b', lambda match: labels[match.group(1)], text)


def relabel_frame(frame, asset_a, asset_b):
    result = frame.rename(columns=lambda name: relabel_text(name, asset_a, asset_b)).copy()
    for column in result.columns:
        if 'Reason' in column:
            result[column] = result[column].map(
                lambda value: relabel_text(value, asset_a, asset_b), na_action='ignore',
            )
    return result


def pair_alert_inputs(state, asset_a, asset_b):
    """Present the stored alerts to the shared engine under its symbol names."""
    engine_symbols = {asset_a: 'QQQ', asset_b: 'NQ=F', GAUGE: '^VIX'}
    rows = [
        {**row, 'Symbol': engine_symbols[row['Symbol']]}
        for row in state.get(ALERTS_KEY, []) if row.get('Symbol') in engine_symbols
    ]
    return {
        'daily_alerts': rows,
        'alert_timeframe': state.get(ALERT_TF_KEY),
        'alert_threshold': state.get(ALERT_THRESHOLD_KEY, 1.0),
    }


def build_pair_signals(history_a, history_b, asset_a, asset_b, timeframe, threshold, as_of, inputs):
    signals = build_qqq_nq_signals(history_a, history_b, timeframe, threshold, as_of=as_of, **inputs)
    return relabel_frame(signals, asset_a, asset_b)


def median_close(execution, start):
    window = execution.loc[execution.index >= pd.Timestamp(start)]
    return float((window if not window.empty else execution)['close'].median())


def pair_figure(day, threshold, asset_a, asset_b):
    figure = go.Figure()
    x = pd.to_datetime(day['Signal Time (Ghana)'], utc=True).dt.tz_localize(None)
    for column, color, dash in [
        (f'{asset_a} Score', '#4C78A8', 'solid'), (f'{asset_b} Score', '#F28E2B', 'dash'),
        ('Confirmation Score', '#B8BCC6', 'dot'),
    ]:
        figure.add_trace(go.Scatter(
            x=x, y=day[column], name=column,
            mode='lines+markers' if day[column].notna().sum() >= 8 else 'markers',
            line=dict(color=color, dash=dash, width=2), marker=dict(size=5),
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
        title=f'{asset_a} and {asset_b} demand/supply confirmation',
        xaxis_title='Signal time — Ghana (GMT/UTC)', yaxis_title='Direction score',
        yaxis=dict(range=[-105, 105]), height=440,
        legend=dict(orientation='h', y=1.18), margin=dict(t=100),
    )
    return figure


def render_overview(deps, basket):
    st.divider()
    st.subheader('🏛️ Top Meme Coins Overview')
    st.write('Candle basket: ' + ', '.join(basket) + f', plus Binance {GAUGE} as the inverse gauge.')
    timeframe = st.selectbox('Select overview timeframe', ['15m', '1h', '4h'], index=1, key='mc_overview_tf')

    with st.expander('Meme Coin Alert Settings'):
        left, right = st.columns(2)
        enabled = left.checkbox('Enable overview alerts', value=False, key='mc_alert_enabled')
        threshold = float(right.number_input(
            'Absolute Z-Score alert level', min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            key='mc_alert_threshold_input',
        ))
        telegram = st.checkbox(
            'Send qualifying alerts to Telegram', value=True, key='mc_alert_telegram',
            help='Uses the Telegram Bot Token and Chat ID entered in the sidebar.',
        )
        st.caption(
            'Triggers when the table Z-Score reaches the positive level or falls to the negative '
            'level. Alerts are checked when the overview is refreshed.'
        )

    state = st.session_state
    coins = {**basket, GAUGE: GAUGE}
    if st.button('Refresh Meme Overview', key='mc_overview_refresh'):
        as_of = pd.Timestamp.now(tz='UTC')
        rows, daily_alerts = [], []
        with st.spinner(spinner_text('Fetching candles', len(coins))):
            frames, failures = load_basket(deps, coins, timeframe)
        for label, candles in frames.items():
            rows.append(build_overview_row(label, candles, timeframe))
            daily_alerts.extend(build_daily_alerts(
                label, candles, timeframe, threshold, as_of, deps.build_daily_alert_history,
            ))
        state['mc_overview_rows'] = rows
        state['mc_overview_timeframe'] = timeframe
        state['mc_overview_failures'] = failures
        state[ALERTS_KEY] = sorted(daily_alerts, key=lambda alert: alert['Ghana Time'], reverse=True)
        state[ALERT_TF_KEY] = timeframe
        state[ALERT_THRESHOLD_KEY] = threshold
        if enabled and rows:
            seen = state.setdefault('mc_alert_keys', set())
            fresh = new_threshold_alerts(rows, threshold, timeframe, seen)
            for alert in fresh:
                if telegram and deps.tg_token and deps.tg_chat_id:
                    deps.send_telegram_alert(
                        deps.tg_token, deps.tg_chat_id, alert['Symbol'], alert['Alert'],
                        format_meme_price(alert['Price']), f"Z-Score {alert['Z-Score']:+.2f}",
                        [f'Threshold: +/-{threshold:.2f}',
                         f"Z-Score Change: {alert['Z-Score Change']:+.2f}", f'Timeframe: {timeframe}'],
                    )
            if fresh:
                state['mc_recent_alerts'] = (fresh + state.get('mc_recent_alerts', []))[:100]
                st.toast(f'{len(fresh)} new meme coin alert(s) triggered.', icon='🚨')
            else:
                st.info('No meme coins met the selected alert thresholds.')

    rows = state.get('mc_overview_rows')
    show_failures(state.get('mc_overview_failures'))
    if not rows:
        st.info('Click Refresh Meme Overview to load the table and daily alerts.')
        return

    table = pd.DataFrame(rows)
    advancing, declining = int(table['Advancing'].sum()), int(table['Declining'].sum())
    if declining:
        breadth = f'{advancing / declining:.2f} ({advancing}/{declining})'
    else:
        breadth = f"{'All advancing' if advancing else 'No decliners'} ({advancing}/{declining})"
    table['Breadth Ratio'] = breadth
    table['Breadth %'] = advancing / len(table)
    table['Momentum'] = [
        combine_metric_value(value, change, percentage=True)
        for value, change in zip(table['Momentum'], table['Momentum Change'])
    ]
    for metric in ('Signal Score', 'Flow Z-Score', 'Z-Score'):
        table[metric] = [
            combine_metric_value(value, change)
            for value, change in zip(table[metric], table[f'{metric} Change'])
        ]
    columns = [
        'Symbol', 'Breadth Ratio', 'Breadth %', 'Price', 'Momentum', 'RSI', 'Trend',
        'Signal Score', 'Volume Ratio', 'Flow Z-Score', 'Z-Score', 'Est. Daily Volume',
    ]
    results_timeframe = state.get('mc_overview_timeframe', timeframe)
    st.caption(
        f'Bracketed change = current value minus the previous {results_timeframe} candle. '
        'Breadth Ratio = Advancing / Declining. Breadth % = Advancing / Total. '
        'Signal Score = -Z-Score + ln(1 + Volume Ratio) + Flow Z-Score. '
        'Est. Daily Volume = USD volume over the last 24 hours of candles.'
    )
    st.dataframe(
        table[columns].style.format({
            'Price': format_meme_price, 'RSI': '{:.2f}', 'Est. Daily Volume': deps.format_large_number,
            'Volume Ratio': '{:.2f}x', 'Breadth %': '{:.0%}',
        }, na_rep='n/a').map(deps.color_metrics, subset=['Breadth %']).map(
            color_bracketed_change, subset=['Momentum', 'Signal Score', 'Flow Z-Score', 'Z-Score'],
        ),
        width='stretch', hide_index=True,
    )

    if state.get('mc_recent_alerts'):
        st.subheader('Recent Meme Coin Alerts')
        st.dataframe(
            pd.DataFrame(state['mc_recent_alerts']).style.format({
                'Price': format_meme_price, 'Z-Score': '{:+.2f}', 'Z-Score Change': '{:+.2f}',
            }).map(deps.color_metrics, subset=['Z-Score', 'Z-Score Change']),
            width='stretch', hide_index=True,
        )
    render_daily_alert_table(deps)


def render_daily_alert_table(deps):
    st.subheader('Daily Historical Z-Score Alerts — Ghana Time')
    st.caption(
        'For each Ghana calendar day, this table shows the first time each symbol had aligned '
        'Z-Score and Momentum Z values at the positive threshold, and the first aligned negative '
        'event. Alert times are candle closes; unfinished candles are excluded when refreshed.'
    )
    alerts = pd.DataFrame(st.session_state.get(ALERTS_KEY, []))
    if alerts.empty:
        st.info('No historical candles reached the selected threshold. Refresh the overview to rebuild this history.')
        return
    alerts['Ghana Date'] = pd.to_datetime(alerts['Ghana Date']).dt.date
    dates = sorted(alerts['Ghana Date'].dropna().unique())
    if st.session_state.get('mc_alert_date') not in dates:
        st.session_state.pop('mc_alert_date', None)
    selected = st.selectbox(
        'Select Ghana alert date', options=dates, index=len(dates) - 1,
        format_func=lambda day: day.strftime('%A, %B %d, %Y'), key='mc_alert_date',
    )
    day = alerts.loc[alerts['Ghana Date'] == selected].copy()
    day['Ghana Date'] = day['Ghana Date'].map(lambda value: value.strftime('%Y-%m-%d'))
    day['Ghana Time'] = pd.to_datetime(day['Ghana Time'], utc=True).dt.tz_convert(
        'Africa/Accra'
    ).dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(
        day.style.format({'Price': format_meme_price, 'Z-Score': '{:+.2f}', 'Momentum Z': '{:+.2f}'}).map(
            deps.color_metrics, subset=['Z-Score', 'Momentum Z'],
        ),
        width='stretch', hide_index=True,
    )


def render_gex(deps, basket):
    st.divider()
    st.subheader('🛡️ Meme Coin GEX Proxy (Gamma Exposure)')
    st.write(
        'DEX tokens have no listed options, so real gamma positioning does not exist for them. '
        'This is the same volatility-based proxy the US tab falls back to.'
    )
    asset = choose(st, 'Select coin for GEX analysis', basket, 'mc_gex_asset')
    timeframe = st.selectbox(
        'Select timeframe for price/volatility', ['5m', '15m', '1h', '4h', '12h', '1d'], index=2, key='mc_gex_tf',
    )
    if not st.button('Analyze Meme Coin GEX', key='mc_gex_run'):
        return
    with st.spinner(f'Analyzing GEX for {asset}...'):
        candles, reason = load_candles(deps, basket[asset], timeframe, limit=200)
    if candles is None:
        st.warning(f'Could not load {asset}: {reason}.')
        return
    price = float(candles['close'].iloc[-1])
    returns = candles['close'].pct_change().dropna()
    volatility = float(returns.std() * np.sqrt(24)) if len(returns) > 1 else None
    summary = deps.summarize_gex(scaled_gex_proxy(deps.build_gex_proxy, price, volatility), price)
    st.warning('Showing a synthetic proxy derived from volatility, not real options positioning.')
    metrics = st.columns(3)
    metrics[0].metric('Total GEX', f"{summary['total_gex']:.4g}")
    metrics[1].metric('Zero Gamma Level', format_meme_price(summary['zero_gamma_level']))
    metrics[2].metric('Current Price', format_meme_price(summary['current_price']))
    st.plotly_chart(gex_figure(summary, asset, 'Gamma Exposure PROXY'), width='stretch', key='mc_gex_chart')


def render_order_flow(deps, basket):
    st.divider()
    st.subheader('📊 Real-Time Order Flow')
    left, right = st.columns(2)
    timeframe = left.selectbox('Select Order Flow Timeframe', ['5m', '15m', '1h', '4h'], index=2, key='mc_flow_tf')
    candles_wanted = int(right.number_input(
        'Number of Candles to Analyze', min_value=50, max_value=1000, value=500, step=50, key='mc_flow_candles',
    ))
    state = st.session_state
    if st.button('Refresh Order Flow', key='mc_flow_refresh'):
        with st.spinner(spinner_text('Calculating order flow', len(basket))):
            frames, failures = load_basket(deps, basket, timeframe, candles_wanted)
        state['mc_flow_data'] = (
            [{'Symbol': label, **order_flow_summary(frame)} for label, frame in frames.items()],
            candles_wanted, timeframe,
        )
        state['mc_flow_history'] = {label: with_flow_history(frame) for label, frame in frames.items()}
        state['mc_flow_failures'] = failures
    show_failures(state.get('mc_flow_failures'))
    saved = state.get('mc_flow_data')
    if not saved or not saved[0]:
        st.info('Click Refresh Order Flow to load buying versus selling pressure.')
        return
    rows, used_candles, used_timeframe = saved
    st.caption(
        f'Buying vs. selling pressure in USD over the last {used_candles} {used_timeframe} candles: '
        'volume on up candles versus down candles. This is a candle-direction estimate, not taker data.'
    )
    st.dataframe(
        pd.DataFrame(rows).sort_values('Net Flow', ascending=False).style.format({
            'Inflow': deps.format_large_number, 'Outflow': deps.format_large_number,
            'Net Flow': deps.format_large_number, 'Money Flow Signal': '{:.2f}',
        }).background_gradient(subset=['Net Flow'], cmap='RdYlGn').bar(
            subset=['Money Flow Signal'], align='zero', color=['#d65f5f', '#5fba7d'],
        ),
        width='stretch', hide_index=True,
    )
    st.subheader('📜 Historical Order Flow (Last 50 Candles)')
    history = state.get('mc_flow_history', {})
    if history:
        asset = choose(st, 'Select Asset for Historical View', history, 'mc_flow_asset')
        st.subheader(f'Cumulative Net Flow for {asset}')
        st.line_chart(history[asset]['cumulative_net_flow'])
        st.dataframe(history[asset].tail(50), width='stretch', height=300)


def render_daily_flow(deps, basket):
    st.divider()
    st.subheader('💰 Daily Inflow / Outflow')
    st.caption('Inflow and outflow since Ghana midnight across 5m, 15m, 1h and 4h candles, in USD.')
    state = st.session_state
    if st.button('Refresh Daily Flow', key='mc_daily_flow_refresh'):
        with st.spinner(spinner_text('Loading today’s flow', len(basket))):
            frames, failures = load_basket(deps, basket, '5m')
        state['mc_daily_flow'] = pd.DataFrame(
            [daily_flow_row(label, frame) for label, frame in frames.items()]
        )
        state['mc_daily_flow_failures'] = failures
    show_failures(state.get('mc_daily_flow_failures'))
    table = state.get('mc_daily_flow')
    if table is None or table.empty:
        st.info('Click Refresh Daily Flow to load today’s inflow and outflow.')
        return
    columns = [column for column in table.columns if column != 'Symbol']
    st.dataframe(
        table.sort_values('5m Net Flow', ascending=False).style.format(
            {column: deps.format_large_number for column in columns}, na_rep='n/a',
        ).background_gradient(
            subset=['5m Net Flow', '15m Net Flow', '1h Net Flow', '4h Net Flow'], cmap='RdYlGn',
        ),
        width='stretch', hide_index=True,
    )


def render_qs(deps, basket):
    st.divider()
    st.subheader('Meme Coin Quant Strength (QS) Comparison')
    st.caption(
        'Relative strength across the candle basket. A higher score means stronger performance '
        'versus the group. QS = 50% momentum + 30% flow − 10% volatility + 10% trend.'
    )
    state = st.session_state
    timeframe = state.get('mc_overview_tf', '1h')
    if st.button('Refresh Meme QS Comparison', key='mc_qs_refresh'):
        with st.spinner(spinner_text('Fetching comparison data', len(basket))):
            frames, failures = load_basket(deps, basket, timeframe)
        state['mc_qs_failures'] = failures
        if len(frames) >= 3:
            state['mc_qs_data'] = add_qs_scores(pd.DataFrame(
                [qs_factor_row(label, frame) for label, frame in frames.items()]
            ))
        else:
            state.pop('mc_qs_data', None)
            st.warning('The QS comparison needs at least 3 coins with enough candle history.')
    show_failures(state.get('mc_qs_failures'))
    scores = state.get('mc_qs_data')
    if scores is None:
        st.info("Click 'Refresh Meme QS Comparison' to load the comparison basket.")
        return
    columns = ['symbol', 'qs_score', 'z_momentum', 'z_flow', 'z_volatility', 'z_trend']
    st.dataframe(
        scores[columns].style.format({column: '{:.2f}' for column in columns[1:]}).background_gradient(
            subset=['qs_score'], cmap='viridis',
        ),
        width='stretch', hide_index=True,
    )


def render_historical_z(deps, basket):
    st.divider()
    st.subheader('Historical Z-Score Component Analysis')
    st.caption(
        'Historical Z-scores for each QS component of one coin. A value of +2 means the factor '
        'is 2 standard deviations above its own recent history.'
    )
    left, right = st.columns(2)
    asset = choose(left, 'Select Asset for Z-Score Chart', basket, 'mc_z_asset')
    timeframe = right.selectbox(
        'Select timeframe for historical components', ['5m', '15m', '1h', '4h', '12h', '1d'], index=2, key='mc_z_tf',
    )
    state = st.session_state
    config = (asset, timeframe)
    if st.button(f'Generate Historical Z-Chart for {asset}', key='mc_z_run'):
        with st.spinner(f'Calculating historical Z-scores for {asset}...'):
            history, reason = load_candles(deps, basket[asset], timeframe)
        if history is None:
            st.error(f'Could not generate the Z-chart for {asset}: {reason}.')
        else:
            state['mc_z_data'] = (config, history)
    saved = state.get('mc_z_data')
    if not saved or saved[0] != config:
        return
    history = saved[1].copy()
    dates = sorted(np.unique(pd.DatetimeIndex(history.index).date))
    if not dates:
        return
    with st.popover('Select analysis date'):
        selected = st.date_input(
            'Date to analyze', value=dates[-1], min_value=dates[0], max_value=dates[-1],
            key=f'mc_z_date_{asset}_{timeframe}',
        )
    st.caption(f'Showing component analysis for {selected:%B %d, %Y}.')
    components, _, alerts = analyze_ghana_flow_events(history, asset, selected)
    if components.empty:
        st.info('No historical Z-score data is available for the selected date.')
        return
    figure = go.Figure()
    for column in components.columns:
        figure.add_trace(go.Scatter(x=components.index, y=components[column], mode='lines', name=column))
    figure.add_hline(y=0, line_width=1, line_dash='dash', line_color='grey')
    figure.update_layout(
        title=f'Historical Z-Score Components for {asset} — {selected:%Y-%m-%d}',
        yaxis_title='Z-Score (Standard Deviations from Mean)', template='plotly_dark',
    )
    st.plotly_chart(figure, width='stretch', key='mc_z_chart')
    render_flow_alerts(deps, asset, timeframe, selected, alerts)
    render_alert_quant_verdict(history, alerts, selected, 'mc_aq', price_format=format_meme_price, fee_default=60.0)
    render_drop_events(components)


def render_flow_alerts(deps, asset, timeframe, selected, alerts):
    st.subheader('All Historical Demand & Supply Rise/Drop Alerts — Ghana Day')
    st.caption(
        f'All Flow Z rise and drop events for {asset} during Ghana day {selected:%Y-%m-%d}, using '
        f'{timeframe} candles. The first event after Ghana midnight is compared with the preceding candle.'
    )
    if alerts.empty:
        st.info('No valid Flow Z rise or drop alert exists for this coin during the selected date.')
        return
    counts = alerts['Alert'].value_counts()
    metrics = st.columns(3)
    metrics[0].metric('Total Alerts', len(alerts))
    metrics[1].metric('Rise Alerts', int(counts.get('Flow Z Rise Alert', 0)))
    metrics[2].metric('Drop Alerts', int(counts.get('Flow Z Drop Alert', 0)))
    columns = [
        'Asset', 'Date', 'Alert', 'Ghana Time', 'Flow Z', 'Flow Z Change', 'Momentum Z Change',
        'Trend Z Change', 'Volatility Z Change', 'Price Change', 'Zone Bottom', 'Zone Top',
        'Direction Score', 'Demand/Supply Alert',
    ]
    st.dataframe(
        alerts[columns].sort_values('Ghana Time').style.format({
            'Flow Z': '{:+.2f}', 'Flow Z Change': '{:+.2f}', 'Momentum Z Change': '{:+.2f}',
            'Trend Z Change': '{:+.2f}', 'Volatility Z Change': '{:+.2f}',
            'Price Change': format_price_change, 'Zone Bottom': format_meme_price,
            'Zone Top': format_meme_price, 'Direction Score': '{:+.1f}',
        }, na_rep='n/a').map(deps.color_metrics, subset=[
            'Flow Z', 'Flow Z Change', 'Momentum Z Change', 'Trend Z Change',
            'Volatility Z Change', 'Price Change', 'Direction Score',
        ]),
        width='stretch', hide_index=True,
    )
    for direction, label in [('drop', 'Drop'), ('rise', 'Rise')]:
        zone = highest_flow_zone(alerts, direction)
        if zone is None:
            st.info(f'No Flow Z {direction} event was detected on this date.')
            continue
        st.subheader(f'Demand & Supply Analysis — Highest Flow Z {label}')
        st.caption(
            'Zone Direction Score = 60% Flow Z change + 25% Momentum Z change + 15% directional '
            'volatility regime. The candle range is a candidate zone, not order-book proof.'
        )
        metrics = st.columns(5)
        metrics[0].metric('Classification', zone['Classification'])
        metrics[1].metric('Direction Score', f"{zone['Direction Score']:+.1f}")
        metrics[2].metric('Analyzed Component', 'Flow (z)')
        metrics[3].metric(f'{label} Size', f"{zone['Event Size']:.2f}σ")
        metrics[4].metric('Volatility Regime', zone['Volatility Regime'])
        zone_columns = [
            'Asset', 'Ghana Time', 'Zone Bottom', 'Zone Top', 'Close', 'Flow Z Change',
            'Momentum Z Change', 'Volatility Z Change', 'Price Change', 'Flow Contribution',
            'Momentum Contribution', 'Volatility Contribution',
        ]
        st.dataframe(
            pd.DataFrame([zone])[zone_columns].style.format({
                'Zone Bottom': format_meme_price, 'Zone Top': format_meme_price, 'Close': format_meme_price,
                'Flow Z Change': '{:+.3f}', 'Momentum Z Change': '{:+.3f}',
                'Volatility Z Change': '{:+.3f}', 'Price Change': format_price_change,
                'Flow Contribution': '{:+.1f}', 'Momentum Contribution': '{:+.1f}',
                'Volatility Contribution': '{:+.1f}',
            }, na_rep='n/a'),
            width='stretch', hide_index=True,
        )


def render_drop_events(components):
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
        st.info('No downward moves were detected on the selected date.')
        return
    drops = pd.DataFrame(events).sort_values('Drop Time', ascending=False).head(50)
    drops['Drop Time'] = drops['Drop Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    st.subheader('Drop Events for Selected Date (up to 50)')
    st.dataframe(
        drops.style.format({'Previous Value': '{:.3f}', 'Current Value': '{:.3f}', 'Drop Size': '{:.3f}'}),
        width='stretch', hide_index=True,
    )


def pair_controls(basket):
    left, right = st.columns(2)
    asset_a = choose(left, 'First coin (QQQ role)', basket, 'mc_pair_a')
    asset_b = choose(right, 'Second coin (NQ role)', [name for name in basket if name != asset_a], 'mc_pair_b')
    return asset_a, asset_b


def render_pair_signals(deps, basket):
    st.divider()
    st.subheader('Meme Coin Pair Quantitative Confirmation Signals')
    st.caption(
        'The QQQ + NQ model applied to two meme coins: their direction scores must agree, confirmed '
        'by stored daily alerts for both coins and BTCDOM. BTCDOM replaces VIX as the inverse gauge.'
    )
    if len(basket) < 2:
        st.info('Pair signals need at least two coins in the candle basket.')
        return None
    asset_a, asset_b = pair_controls(basket)
    left, right = st.columns(2)
    timeframe = left.selectbox('Pair timeframe', ['5m', '15m', '1h', '4h'], index=2, key='mc_pair_tf')
    threshold = float(right.number_input(
        'Minimum direction score (each coin)', min_value=1.0, max_value=100.0, value=10.0, step=1.0,
        key='mc_pair_threshold',
    ))
    state = st.session_state
    config = (asset_a, asset_b, timeframe)
    if st.button('Generate Pair Signals', type='primary', key='mc_pair_generate'):
        histories, errors = {}, []
        with st.spinner(spinner_text('Loading histories', 2)):
            for label in (asset_a, asset_b):
                frame, reason = load_candles(deps, basket[label], timeframe)
                if frame is None:
                    errors.append(f'{label}: {reason}')
                else:
                    histories[label] = frame.copy()
        state['mc_pair_source'] = {
            'config': config, 'histories': histories, 'errors': errors,
            'loaded_at': pd.Timestamp.now(tz='UTC'),
        }
    with st.expander('Signal rules'):
        st.write(
            'Direction score = 100 × [0.60·tanh(ΔFlow Z) + 0.25·tanh(ΔMomentum Z) + 0.15·directional '
            'volatility]. BUY: both scores ≥ the threshold, plus positive daily alerts for both coins and a '
            'negative BTCDOM daily alert. SELL: the reverse. CONFLICT: scores have opposite signs. '
            'WAIT: incomplete data or insufficient agreement.'
        )
        st.caption(
            'Daily alerts use only events already available that Ghana day; nothing carries overnight. '
            'Scores are model values, not probabilities.'
        )
    inputs = pair_alert_inputs(state, asset_a, asset_b)
    rows, issue = prepare_daily_alerts(**inputs)
    issue = relabel_text(issue, asset_a, asset_b)
    if issue:
        st.warning(issue + ' BUY/SELL signals remain disabled until this source is ready.')
    else:
        st.caption(
            f'Daily source: {len(rows)} stored {asset_a} / {asset_b} / BTCDOM events · 1h · '
            f'threshold ±{float(inputs["alert_threshold"]):g}. Refresh Meme Overview updates this source.'
        )
    source = state.get('mc_pair_source')
    if not source:
        st.info('Choose two coins and a timeframe, then click Generate Pair Signals.')
    elif source['config'] != config:
        st.info('The coins or timeframe changed. Click Generate Pair Signals to reload them.')
    elif source['errors']:
        st.error('Unable to compare both coins. ' + '; '.join(source['errors']))
    else:
        show_pair_signals(source, asset_a, asset_b, timeframe, threshold, inputs)
    return asset_a, asset_b, timeframe, threshold


def show_pair_signals(source, asset_a, asset_b, timeframe, threshold, inputs):
    signals = build_pair_signals(
        source['histories'][asset_a], source['histories'][asset_b], asset_a, asset_b,
        timeframe, threshold, source['loaded_at'], inputs,
    )
    if signals.empty:
        st.info('No completed candles are available for this comparison.')
        return
    dates = sorted(set(pd.to_datetime(signals['Signal Time (Ghana)'], utc=True).dt.date))
    selected = st.date_input(
        'Pair Ghana date', value=dates[-1], min_value=dates[0], max_value=dates[-1],
        key=f'mc_pair_date_{timeframe}_{dates[0]}_{dates[-1]}',
    )
    day = signals_on_ghana_day(signals, selected)
    if day.empty:
        st.info('No completed candles fall on this Ghana day. Choose another date.')
        return
    summary = st.columns(4)
    summary[0].metric('Comparable candles', int((day[f'{asset_a} Ready'].eq(True) & day[f'{asset_b} Ready'].eq(True)).sum()))
    summary[1].metric('BUY candles', int(day['Signal'].eq('BUY').sum()))
    summary[2].metric('SELL candles', int(day['Signal'].eq('SELL').sum()))
    summary[3].metric('Conflicts', int(day['Signal'].eq('CONFLICT').sum()))
    latest = day.iloc[-1]
    st.write(f'Last available state on this date: **{latest["Signal"]}** — {latest["Reason"]}.')
    blocked = day['Base Signal'].isin(['BUY', 'SELL']) & day['Signal'].eq('WAIT')
    st.caption(f'{int(blocked.sum())} base BUY/SELL candle(s) held at WAIT by the daily alert filter.')
    if day[[f'{asset_a} Score', f'{asset_b} Score']].notna().any().any():
        st.plotly_chart(pair_figure(day, threshold, asset_a, asset_b), width='stretch', key='mc_pair_chart')
    st.markdown('##### All pair signals — selected Ghana day')
    prices = [f'{asset_a} Price', f'{asset_b} Price']
    columns = [
        'Signal Time (Ghana)', 'Signal', 'New Alert', 'Base Signal', f'{asset_a} Score', f'{asset_b} Score',
        'Confirmation Score', f'{asset_a} Flow Event', f'{asset_b} Flow Event', 'Daily Alert Confirmation',
        *[relabel_text(column, asset_a, asset_b) for column in EVIDENCE_COLUMNS],
        *prices, 'Reason', 'Daily Alert Reason',
    ]
    table = day.reset_index()[columns]
    numeric = [column for column in table.select_dtypes(include='number').columns if column not in prices]
    st.dataframe(
        table.style.format(
            {**{column: '{:+.2f}' for column in numeric}, **{column: format_meme_price for column in prices}},
            na_rep='Unavailable',
        ),
        width='stretch', hide_index=True,
    )
    st.download_button(
        'Download this day’s pair signals', table.to_csv(index=False).encode('utf-8'),
        file_name=f'meme_pair_signals_{selected}_{timeframe}.csv', mime='text/csv', key='mc_pair_download',
    )


def render_pair_backtest(deps, basket, pair):
    st.markdown('#### Backtest Pair Confirmation Signals')
    if pair is None:
        return
    asset_a, asset_b, timeframe, threshold = pair
    state = st.session_state
    today = pd.Timestamp.now(tz='UTC').date()
    earliest = today - timedelta(days=BACKTEST_HISTORY_DAYS)
    c1, c2, c3, c4 = st.columns(4)
    start = c1.date_input('Backtest start (Ghana)', today - timedelta(days=14),
                          min_value=earliest, max_value=today, key='mc_bt_start')
    end = c2.date_input('Backtest end (Ghana, inclusive)', today,
                        min_value=earliest, max_value=today, key='mc_bt_end')
    traded = c3.selectbox('Traded coin', [asset_a, asset_b], key=f'mc_bt_traded_{asset_a}_{asset_b}')
    target_label = c4.selectbox('Take-profit rule', ['Nearest 1-hour liquidity zone', 'Fixed reward / risk'],
                                key='mc_bt_target')
    target_mode = 'liquidity' if target_label.startswith('Nearest') else 'fixed_rr'
    c5, c6, c7, c8 = st.columns(4)
    atr_multiple = c5.number_input('Stop: 5m ATR(14) multiple', .25, 10.0, 1.0, .25, key='mc_bt_atr')
    reward_risk = c6.number_input('Reward / risk', .25, 10.0, 2.0, .25,
                                  disabled=target_mode == 'liquidity', key='mc_bt_rr')
    slippage_bps = c7.number_input('Slippage per fill (bps)', 0.0, 2000.0, 30.0, 5.0, key='mc_bt_slip')
    fee_bps = c8.number_input('Round-trip fees (bps)', 0.0, 2000.0, 60.0, 5.0, key='mc_bt_fee')
    st.caption(
        f'Signal timeframe: {timeframe} · score threshold: ±{threshold:g}. Costs are editable assumptions: '
        'a 0.3% pool fee each way, plus slippage. Real price impact grows with trade size and thin pools, '
        'and shorting needs a venue that a DEX spot pool does not provide. Costs convert at the median '
        'close of the test window, which is rough for coins that move many multiples.'
    )
    inputs = pair_alert_inputs(state, asset_a, asset_b)
    _, issue = prepare_daily_alerts(**inputs)
    issue = relabel_text(issue, asset_a, asset_b)
    config = (asset_a, asset_b, timeframe, threshold, start, end, traded, target_mode, atr_multiple,
              reward_risk, slippage_bps, fee_bps, alert_source_fingerprint(inputs))
    with st.expander('Backtest execution rules'):
        st.write(
            'Only new BUY/SELL confirmations enter. BUY goes long the traded coin; SELL goes short. Entry '
            'is the 5-minute open at the exact signal time, with adverse slippage. One position stays open '
            'until stop, target or the end of the test. When a candle touches both stop and target, the '
            'stop counts first. Liquidity targets use only 1-hour candles completed by entry.'
        )
    if st.button('Run Pair Backtest', type='primary', key='mc_bt_run'):
        state.pop('mc_bt_result', None)
        if start > end:
            st.error('The start date must be on or before the end date.')
        elif issue:
            st.error(issue)
        else:
            run_pair_backtest(deps, basket, config)
    show_pair_backtest(config, start, end, asset_a, asset_b, traded, timeframe)


def run_pair_backtest(deps, basket, config):
    (asset_a, asset_b, timeframe, threshold, start, end, traded, target_mode, atr_multiple,
     reward_risk, slippage_bps, fee_bps, _) = config
    state = st.session_state
    warmup = str(start - timedelta(days=WARMUP_DAYS[timeframe]))
    requests_needed = [(asset_a, timeframe, warmup), (asset_b, timeframe, warmup),
                       (traded, '5m', str(start - timedelta(days=2)))]
    if target_mode == 'liquidity':
        requests_needed.append((traded, '1h', str(start - timedelta(days=15))))
    try:
        with st.spinner('Loading coin history and simulating trades. Paged 5m history can take minutes...'):
            sources = {}
            for label, bar, since in requests_needed:
                frame, reason = load_candles(deps, basket[label], bar, start_date=since)
                if frame is None:
                    raise ValueError(f'{label} {bar}: {reason}.')
                sources[(label, bar)] = frame
            execution = sources[(traded, '5m')]
            reference = median_close(execution, start)
            result = backtest_qqq_nq(
                sources[(asset_a, timeframe)], sources[(asset_b, timeframe)], execution,
                sources.get((traded, '1h')), timeframe=timeframe, threshold=threshold,
                start=start, end_exclusive=end + timedelta(days=1),
                zone_calculator=deps.calculate_liquidity_zones, atr_multiple=atr_multiple,
                target_mode=target_mode, reward_risk=reward_risk,
                slippage_points=reference * slippage_bps / 10_000,
                commission_points=reference * fee_bps / 10_000,
                **pair_alert_inputs(state, asset_a, asset_b),
            )
        state['mc_bt_result'] = {
            'config': config, 'result': result, 'reference': reference,
            'run_at': pd.Timestamp.now(tz='UTC'),
        }
    except Exception as error:
        # The engine always executes its NQ role, which here means the chosen traded coin.
        message = str(error).replace('NQ 5-minute', f'{traded} 5-minute')
        st.error('The backtest could not run: ' + relabel_text(message, asset_a, asset_b))


def show_pair_backtest(config, start, end, asset_a, asset_b, traded, timeframe):
    stored = st.session_state.get('mc_bt_result')
    if not stored:
        st.info('Choose the Ghana date range and click Run Pair Backtest.')
        return
    if stored['config'] != config:
        st.info('Backtest settings or stored daily alerts changed. Run again to update the results.')
        return
    result = stored['result']
    summary = result['summary']
    trades = relabel_frame(result['trades'], asset_a, asset_b)
    audit = relabel_frame(result['audit'], asset_a, asset_b)
    st.caption(
        f'Tested {start:%d %b %Y} through {end:%d %b %Y} Ghana · run {stored["run_at"]:%d %b %H:%M} Ghana. '
        f'Trading {traded}.'
    )
    metrics = st.columns(6)
    metrics[0].metric('Trades', summary['Trades'])
    metrics[1].metric('Win rate', f'{summary["Win Rate %"]:.1f}%' if len(trades) else '—')
    metrics[2].metric('Net R', f'{summary["Net R"]:+.2f}R')
    metrics[3].metric('Net % of median price', f'{summary["Net Points"] / stored["reference"]:+.2%}')
    factor = summary['Profit Factor (points)']
    metrics[4].metric('Profit factor', '—' if pd.isna(factor) else ('∞' if np.isinf(factor) else f'{factor:.2f}'))
    metrics[5].metric('Closed drawdown', f'{summary["Max Closed Drawdown R"]:.2f}R')
    st.write(f'New confirmations: {summary["New confirmations"]} · skipped: {summary["Skipped"]}.')
    st.caption('This filter can leave very few trades. A small sample does not establish future profitability.')
    if trades.empty:
        st.info('No trades completed under these settings. The decision log explains any skipped confirmations.')
    else:
        figure = backtest_equity_figure(trades, start)
        figure.update_layout(title=f'{asset_a} / {asset_b} confirmation backtest — closed-trade equity')
        st.plotly_chart(figure, width='stretch', key='mc_bt_equity')
        number_formats = {
            name: '{:.6g}' if name in PRICE_LIKE_TRADE_COLUMNS else '{:,.4f}'
            for name in trades.select_dtypes(include='number').columns if name != 'Trade'
        }
        st.dataframe(trades.style.format(number_formats, na_rep='—'), width='stretch', hide_index=True)
        st.download_button('Download backtest trades', trades.to_csv(index=False).encode(),
                           f'meme_pair_backtest_{start}_{end}_{timeframe}.csv', 'text/csv', key='mc_bt_export')
    with st.expander('Signal decision log — traded and skipped'):
        if audit.empty:
            st.write('No new BUY/SELL confirmations fell inside the selected Ghana dates.')
        else:
            st.dataframe(audit, width='stretch', hide_index=True)


def render_zone_battle(deps, basket):
    st.divider()
    st.subheader('⚔️ Zone Battle Score')
    st.caption(
        'Measures who won after price entered a bullish or bearish FVG zone. '
        'Battle Score = 0.40(Rz) + 0.30(Fz) + 0.20(Mz) + 0.10(Az).'
    )
    c1, c2, c3 = st.columns(3)
    asset = choose(c1, 'Zone Battle coin', basket, 'mc_zone_asset')
    timeframe = c2.selectbox('Zone Battle timeframe', ['5m', '15m', '1h', '4h', '1d'], index=2, key='mc_zone_tf')
    bars = int(c3.number_input('Reaction observation candles', min_value=3, max_value=50, value=10, step=1,
                               key='mc_zone_bars'))
    config = (asset, timeframe, bars)
    state = st.session_state
    if st.button(f'Analyze Zone Battles for {asset}', key='mc_zone_run'):
        with st.spinner(f'Detecting {asset} FVG zones on {timeframe} candles...'):
            candles, reason = load_candles(deps, basket[asset], timeframe)
        if candles is None:
            st.error(f'Could not load {asset}: {reason}.')
            state.pop('mc_zone', None)
        else:
            results, _ = deps.calculate_zone_battle_scores(candles, observation_bars=bars, max_zones=50)
            state['mc_zone'] = (config, results)
    saved = state.get('mc_zone')
    if not saved:
        return
    if saved[0] != config:
        st.info('The coin, timeframe, or observation window changed. Run the analysis again to refresh the results.')
        return
    results = saved[1]
    if results.empty:
        st.info('No completed FVG zone reactions were found for this coin and timeframe.')
        return
    latest = results.sort_values('Zone Entry', ascending=False).iloc[0]
    metrics = st.columns(4)
    metrics[0].metric('Latest Battle Score', f"{latest['Battle Score']:+.2f}")
    metrics[1].metric('Result', latest['Result'])
    metrics[2].metric('Zone Type', latest['Zone Type'])
    metrics[3].metric('Zone Range', f"{format_meme_price(latest['Zone Bottom'])} – {format_meme_price(latest['Zone Top'])}")
    history = results.sort_values('Zone Entry')
    figure = go.Figure(go.Scatter(
        x=history['Zone Entry'], y=history['Battle Score'], mode='lines+markers', name='Battle Score',
        marker=dict(size=8, color=np.where(
            history['Battle Score'] > 1, '#22c55e', np.where(history['Battle Score'] < -1, '#ef4444', '#f59e0b'),
        )),
    ))
    figure.add_hline(y=1.0, line_dash='dash', line_color='#22c55e', annotation_text='Strong Buyer Victory')
    figure.add_hline(y=-1.0, line_dash='dash', line_color='#ef4444', annotation_text='Strong Seller Victory')
    figure.update_layout(title=f'{asset} Zone Battle History ({timeframe})', xaxis_title='Zone Entry Time',
                         yaxis_title='Battle Score', template='plotly_dark')
    st.plotly_chart(figure, width='stretch', key='mc_zone_chart')
    columns = [
        'Zone Entry', 'Zone Type', 'Zone Bottom', 'Zone Top', 'Entry Price', 'Reaction Close', 'Bars Observed',
        'Reaction Rz', 'Flow Fz', 'Momentum Mz', 'Acceptance Az', 'Battle Score', 'Result',
    ]
    scores = ['Reaction Rz', 'Flow Fz', 'Momentum Mz', 'Acceptance Az', 'Battle Score']
    st.dataframe(
        results[columns].style.format({
            'Zone Bottom': format_meme_price, 'Zone Top': format_meme_price, 'Entry Price': format_meme_price,
            'Reaction Close': format_meme_price, **{name: '{:+.2f}' for name in scores},
        }).map(deps.color_metrics, subset=scores),
        width='stretch', hide_index=True,
    )
    with st.expander('How the Zone Battle components are calculated'):
        st.markdown(
            """
- **Reaction Rz:** `(reaction close - zone entry price) / ATR(15)`.
- **Flow Fz:** the Historical Flow Z-score at the end of the reaction window.
- **Momentum Mz:** the Historical Momentum Z-score at the end of the reaction window.
- **Acceptance Az:** average candle location: `+1` above the zone, `0` inside it, and `-1` below it.
- **Battle Score > +1:** strong buyer victory.
- **Battle Score < -1:** strong seller victory.
- **Between -1 and +1:** battle unresolved.
            """
        )


def render_liquidity(deps, basket):
    st.divider()
    st.subheader('Liquidity Zones')
    st.caption(
        'Ranks price bands by volume, absolute order-flow imbalance, ATR compression, swing density, '
        'time at price, and proximity to VWAP. The combined Liquidity Score is normalized from 0 to 100.'
    )
    c1, c2, c3 = st.columns(3)
    asset = choose(c1, 'Liquidity Zone coin', basket, 'mc_liquidity_asset')
    timeframe = c2.selectbox('Liquidity Zone timeframe', ['5m', '15m', '1h', '4h', '1d'], index=2, key='mc_liquidity_tf')
    bins = int(c3.number_input('Number of price bands', min_value=10, max_value=60, value=24, step=2,
                               key='mc_liquidity_bins'))
    config = (asset, timeframe, bins)
    state = st.session_state
    if st.button(f'Find Liquidity Zones for {asset}', key='mc_liquidity_run'):
        with st.spinner(f'Mapping {asset} liquidity on {timeframe} candles...'):
            candles, reason = load_candles(deps, basket[asset], timeframe, limit=750)
        if candles is None:
            st.error(f'Could not load {asset}: {reason}.')
            state.pop('mc_liquidity', None)
        else:
            zones, history = deps.calculate_liquidity_zones(candles, price_bins=bins, top_zones=8)
            state['mc_liquidity'] = (config, zones, history)
    saved = state.get('mc_liquidity')
    if not saved:
        return
    if saved[0] != config:
        st.info('The coin, timeframe, or band count changed. Run the analysis again to refresh the results.')
        return
    _, zones, history = saved
    if zones.empty or history.empty:
        st.info('Not enough price and volume history was available to map liquidity zones.')
        return
    strongest, last_price = zones.iloc[0], float(history['close'].iloc[-1])
    metrics = st.columns(4)
    metrics[0].metric('Strongest Zone',
                      f"{format_meme_price(strongest['Zone Bottom'])} - {format_meme_price(strongest['Zone Top'])}")
    metrics[1].metric('Liquidity Score', f"{strongest['Liquidity Score']:.1f}/100")
    metrics[2].metric('Flow Bias', strongest['Flow Bias'])
    metrics[3].metric('Last Price', format_meme_price(last_price))
    figure = go.Figure(go.Bar(
        x=zones['Liquidity Score'], y=zones['Zone Midpoint'], orientation='h',
        width=(zones['Zone Top'] - zones['Zone Bottom']) * 0.82,
        marker=dict(color=zones['Order Flow Imbalance'], cmin=-1, cmax=1, colorbar=dict(title='Flow'),
                    colorscale=[[0, '#ef4444'], [0.5, '#64748b'], [1, '#22c55e']]),
        name='Liquidity Score',
    ))
    figure.add_hline(y=last_price, line_dash='dash', line_color='#f8fafc', annotation_text='Last Price')
    figure.update_layout(title=f'{asset} Top Liquidity Zones ({timeframe})', xaxis_title='Liquidity Score (0-100)',
                         yaxis_title='Price', template='plotly_dark', xaxis=dict(range=[0, 105]))
    st.plotly_chart(figure, width='stretch', key='mc_liquidity_chart')
    columns = ['Zone Bottom', 'Zone Top', 'Liquidity Score', 'Location vs Price', 'Flow Bias',
               'Order Flow Imbalance', 'Volume Z-Score', 'ATR Compression Z', 'Swing Density Z',
               'Time at Price Z', 'VWAP Proximity Z']
    st.dataframe(
        zones[columns].style.format({
            'Zone Bottom': format_meme_price, 'Zone Top': format_meme_price, 'Liquidity Score': '{:.1f}',
            'Order Flow Imbalance': '{:+.2%}', 'Volume Z-Score': '{:+.2f}', 'ATR Compression Z': '{:+.2f}',
            'Swing Density Z': '{:+.2f}', 'Time at Price Z': '{:+.2f}', 'VWAP Proximity Z': '{:+.2f}',
        }).background_gradient(subset=['Liquidity Score'], cmap='viridis'),
        width='stretch', hide_index=True,
    )
    with st.expander('How the Liquidity Score is calculated'):
        st.markdown(
            """
- **25% Volume Z-Score:** unusually concentrated traded volume in the price band.
- **20% Flow Strength:** absolute buy/sell volume imbalance; the sign is retained as Flow Bias.
- **15% ATR Compression:** preference for bands formed during volatility contraction.
- **15% Swing Density:** clustered local highs and lows, where stops may accumulate.
- **15% Time at Price:** number of candles accepted within the band.
- **10% VWAP Proximity:** preference for liquidity near volume-weighted fair value.

Each component is standardized across the displayed price bands before the weighted score is normalized to **0-100**. A high score identifies a potential liquidity pool, not a guaranteed reversal or breakout.
            """
        )


def render_volume_delta(deps, basket):
    st.divider()
    st.subheader('Volume Delta')
    st.caption(
        'Volume Delta = USD volume on up candles minus USD volume on down candles. '
        'Positive values indicate net buying pressure; negative values indicate net selling pressure.'
    )
    timeframe = st.selectbox('Volume Delta timeframe', ['5m', '15m', '1h', '4h'], index=2, key='mc_delta_tf')
    state = st.session_state
    if st.button('Refresh Meme Volume Delta', key='mc_delta_refresh'):
        rows, history = [], {}
        with st.spinner(spinner_text('Calculating volume delta', len(basket))):
            frames, failures = load_basket(deps, basket, timeframe, 500)
        for label, candles in frames.items():
            frame = with_volume_delta(candles)
            history[label] = frame
            rows.append({
                'Asset': label,
                'Latest Volume Delta': float(frame['Volume Delta'].iloc[-1]),
                '20-Candle Volume Delta': float(frame['20-Candle Volume Delta'].iloc[-1]),
                'Cumulative Volume Delta': float(frame['Cumulative Volume Delta'].iloc[-1]),
            })
        state['mc_delta'] = (timeframe, pd.DataFrame(rows), history)
        state['mc_delta_failures'] = failures
    show_failures(state.get('mc_delta_failures'))
    saved = state.get('mc_delta')
    if not saved or saved[0] != timeframe:
        return
    _, summary, history = saved
    if summary.empty:
        st.info('No volume-delta data was returned. Try another timeframe.')
        return
    fields = ['Latest Volume Delta', '20-Candle Volume Delta', 'Cumulative Volume Delta']
    st.dataframe(summary.style.format({field: deps.format_large_number for field in fields}),
                 width='stretch', hide_index=True)
    asset = choose(st, 'Select coin for detailed Volume Delta', summary['Asset'].tolist(), 'mc_delta_asset')
    selected = history[asset]
    metrics = st.columns(3)
    for column, field, source in zip(metrics, fields, ['Volume Delta', '20-Candle Volume Delta', 'Cumulative Volume Delta']):
        column.metric(field, deps.format_large_number(float(selected[source].iloc[-1])))
    st.line_chart(selected['Cumulative Volume Delta'])
    detail = ['Buy Volume', 'Sell Volume', 'Volume Delta', 'Cumulative Volume Delta']
    st.dataframe(selected[detail].tail(50).style.format({name: deps.format_large_number for name in detail}),
                 width='stretch', height=300)


def render_support(deps, basket):
    st.divider()
    st.subheader('Key Support Levels')
    st.caption('Technical levels that may act as support: the 50-candle swing low, lower Bollinger Band and 200-period MA.')
    state = st.session_state
    timeframe = state.get('mc_overview_tf', '1h')
    if st.button('Refresh Key Support Levels', key='mc_support_refresh'):
        with st.spinner(spinner_text('Fetching key support levels', len(basket))):
            frames, failures = load_basket(deps, basket, timeframe)
        rows = []
        for label, candles in frames.items():
            current = candles.iloc[-1]
            rows.append({
                'Symbol': label, 'Current Price': current['close'],
                'Bullish Order Block': current.get('ob_bull', np.nan),
                'Lower Bollinger Band': current.get('bb_lower', np.nan),
                '200-Period MA': current.get('sma_200', np.nan),
            })
        state['mc_support'] = pd.DataFrame(rows)
        state['mc_support_failures'] = failures
    show_failures(state.get('mc_support_failures'))
    table = state.get('mc_support')
    if table is None or table.empty:
        st.info("Click 'Refresh Key Support Levels' to load the support-level basket.")
        return
    prices = ['Current Price', 'Bullish Order Block', 'Lower Bollinger Band', '200-Period MA']
    st.dataframe(table.style.format({name: format_meme_price for name in prices}), width='stretch', hide_index=True)


def render_top_coins(deps, overview):
    st.divider()
    st.subheader('🏢 Top 10 Meme Coins Overview')
    st.write('The 10 scanned pairs with the highest volume in the selected analysis window.')
    state = st.session_state
    timeframe = state.get('mc_overview_tf', '1h')
    if st.button('Refresh Top Meme Coins', key='mc_top_refresh'):
        coins = top_by_volume(overview)
        with st.spinner(spinner_text('Fetching and analyzing the top coins', len(coins))):
            frames, failures = load_basket(deps, coins, timeframe)
        state['mc_top_failures'] = failures
        if frames:
            state['mc_top'] = add_qs_scores(pd.DataFrame(
                [qs_factor_row(label, frame) for label, frame in frames.items()]
            ))
        else:
            state.pop('mc_top', None)
            st.warning('No coin data was returned.')
    show_failures(state.get('mc_top_failures'))
    top = state.get('mc_top')
    if top is None:
        st.info("Click 'Refresh Top Meme Coins' to load the latest metrics.")
        return
    bullish, positive, total = int(top['trend_bullish'].sum()), int((top['momentum'] > 0).sum()), len(top)
    health = bullish / total * 100
    status = '🟢 STRONG BULL' if health >= 70 else '🔴 BEARISH' if health <= 30 else '🟡 NEUTRAL'
    metrics = st.columns(3)
    metrics[0].metric('Coins in Bullish Trend', f'{bullish} / {total}')
    metrics[1].metric('Positive Momentum', f'{positive} / {total}')
    metrics[2].metric('Overall Health', status)
    columns = ['symbol', 'qs_score', 'price', 'momentum', 'rsi']
    st.dataframe(
        top[columns].style.format({
            'price': format_meme_price, 'qs_score': '{:.2f}', 'momentum': '{:.2%}', 'rsi': '{:.1f}',
        }, na_rep='n/a').map(deps.color_metrics, subset=['momentum']),
        width='stretch', hide_index=True,
    )


def render_meme_sections(deps, overview, source_key, refreshed=False):
    """Every US Indices section, in the US tab's order, on the current scan's coins."""
    st.divider()
    st.header('US Indices Analysis on Meme Coins')
    st.write(
        'Dexscreener picks the coins and GeckoTerminal supplies their historical candles, so every '
        'section below uses the same formulas as the US Indices tab.'
    )
    size = st.selectbox(
        'Candle basket size (deepest liquidity first)', BASKET_SIZES, index=1, key='mc_basket_size',
        help='The public candle API allows about five calls per minute, so larger baskets load slowly.',
    )
    basket = resolve_basket(overview, size, source_key, refreshed, st.session_state)
    if not basket:
        st.info('No eligible pairs are in the current scan. Refresh or search for coins above.')
        return
    st.caption(
        'The basket stays fixed until you press Refresh Meme-Coin Tab or change the coin source. '
        'Loaded candles are reused across sections for 5 minutes. Pools younger than '
        f'{MIN_CANDLES} candles cannot be scored and are listed as not loaded.'
    )
    render_overview(deps, basket)
    render_gex(deps, basket)
    render_order_flow(deps, basket)
    render_daily_flow(deps, basket)
    render_qs(deps, basket)
    render_historical_z(deps, basket)
    pair = render_pair_signals(deps, basket)
    render_pair_backtest(deps, basket, pair)
    render_zone_battle(deps, basket)
    render_liquidity(deps, basket)
    render_volume_delta(deps, basket)
    render_support(deps, basket)
    render_top_coins(deps, overview)

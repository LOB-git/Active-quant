"""US Indices tab logic applied to Binance USD-M perpetuals in the Derivative Crypto tab.

The QQQ/NQ confirmation and backtest engines are reused unchanged through a symbol
adapter: BTC takes the QQQ role, ETH the NQ role and BTCDOM the inverse VIX role.
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from binance_futures import fetch_options_gex, intraday_money_flow, top_usdm_symbols
from qqq_nq_backtest import backtest_qqq_nq
from qqq_nq_backtest_panel import backtest_equity_figure
from qqq_nq_daily_alerts import (
    EVIDENCE_COLUMNS, alert_source_fingerprint, prepare_daily_alerts,
)
from qqq_nq_signals import build_qqq_nq_signals, signals_on_ghana_day
from quant_edge import quant_edge_row
import watchlist_history


OVERVIEW_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT', 'BTCDOM/USDT',
]
FLOW_SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT', 'DOGE/USDT']
QS_SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'LINK/USDT', 'AVAX/USDT', 'LTC/USDT',
]
GEX_SYMBOLS = FLOW_SYMBOLS  # Assets with a Binance options chain.
BAR_DURATION = {'15m': pd.Timedelta(minutes=15), '1h': pd.Timedelta(hours=1), '4h': pd.Timedelta(hours=4)}
BARS_PER_DAY = {'5m': 288, '15m': 96, '1h': 24, '4h': 6}

ENGINE_SYMBOLS = {'BTC/USDT': 'QQQ', 'ETH/USDT': 'NQ=F', 'BTCDOM/USDT': '^VIX'}
ENGINE_LABELS = {'QQQ': 'BTC', 'NQ': 'ETH', 'VIX': 'BTCDOM'}
ALERTS_KEY = 'dc_daily_alerts'
ALERT_TF_KEY = 'dc_daily_alert_timeframe'
ALERT_THRESHOLD_KEY = 'dc_daily_alert_threshold'


def format_price(value):
    """Keep low-priced perps readable: DOGE and 1000SHIB are not two-decimal assets."""
    if value is None or pd.isna(value):
        return 'n/a'
    magnitude = abs(float(value))
    decimals = 2 if magnitude >= 1 else 4 if magnitude >= 0.1 else 6 if magnitude >= 0.001 else 8
    return f'${float(value):,.{decimals}f}'


def combine_metric_value(value, change, percentage=False):
    pattern = '{:.2%}' if percentage else '{:.2f}'
    signed = '{:+.2%}' if percentage else '{:+.2f}'
    change_text = signed.format(float(change)) if pd.notna(change) else 'n/a'
    return f'{pattern.format(float(value))} ({change_text})'


def color_bracketed_change(value):
    try:
        change = float(str(value).rsplit('(', 1)[1].rstrip(')').replace('%', ''))
    except (ValueError, IndexError):
        return 'color: gray'
    return f"color: {'green' if change > 0 else 'red' if change < 0 else 'gray'}"


def latest_and_change(series):
    clean = pd.to_numeric(series, errors='coerce').fillna(0.0)
    latest = float(clean.iloc[-1])
    return latest, latest - (float(clean.iloc[-2]) if len(clean) > 1 else latest)


def build_overview_row(symbol, candles, timeframe):
    """Same per-candle metrics as the US overview, with a 24-hour volume day."""
    current = candles.iloc[-1]
    previous = candles.iloc[-2] if len(candles) > 1 else current
    z_scores = pd.to_numeric(candles['z_score'], errors='coerce').fillna(0.0)
    volume_ratio = (
        pd.to_numeric(candles['volume'], errors='coerce')
        / pd.to_numeric(candles['vol_ma'], errors='coerce').replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cmf = pd.to_numeric(candles['cmf'], errors='coerce')
    flow_z = (
        (cmf - cmf.rolling(20).mean()) / (cmf.rolling(20).std() + 1e-9)
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    signal = -z_scores + np.log1p(volume_ratio.clip(lower=0.0)) + flow_z
    momentum, momentum_change = latest_and_change(candles['momentum'])
    z_score, z_change = latest_and_change(z_scores)
    flow_score, flow_change = latest_and_change(flow_z)
    signal_score, signal_change = latest_and_change(signal)
    daily_notional = (candles['volume'] * candles['close']).tail(BARS_PER_DAY[timeframe]).sum()
    return {
        'Symbol': symbol,
        'Candle Time': pd.Timestamp(candles.index[-1]),
        'Price': float(current['close']),
        'Momentum': momentum,
        'Momentum Change': momentum_change,
        'RSI': float(current['rsi']),
        'Trend': 'Bullish 🟢' if current['close'] > current['ema_50'] else 'Bearish 🔴',
        'Advancing': bool(current['close'] > previous['close']),
        'Declining': bool(current['close'] < previous['close']),
        'Est. Daily Volume': float(daily_notional),
        'Z-Score': z_score,
        'Z-Score Change': z_change,
        'Volume Ratio': latest_and_change(volume_ratio)[0],
        'Flow Z-Score': flow_score,
        'Flow Z-Score Change': flow_change,
        'Signal Score': signal_score,
        'Signal Score Change': signal_change,
    }


def build_daily_alerts(symbol, candles, timeframe, threshold, as_of, alert_builder):
    """Stamp bars at their close so unfinished candles cannot raise a daily alert."""
    closes = pd.DatetimeIndex(candles.index) + BAR_DURATION[timeframe]
    z_scores = pd.to_numeric(candles['z_score'], errors='coerce').fillna(0.0)
    momentum = pd.to_numeric(candles['momentum'], errors='coerce')
    momentum_z = (
        (momentum - momentum.rolling(50).mean()) / momentum.rolling(50).std().replace(0, np.nan)
    )
    stamped = candles.copy()
    stamped.index, z_scores.index, momentum_z.index = closes, closes, closes
    return alert_builder(
        stamped, z_scores, symbol, timeframe, threshold=threshold,
        momentum_z_series=momentum_z, as_of=as_of,
    )


def new_threshold_alerts(rows, threshold, timeframe, seen):
    """Return unseen threshold breaches. Adds their keys to `seen`."""
    alerts = []
    for row in rows:
        z_score = float(row['Z-Score'])
        if abs(z_score) < threshold:
            continue
        side = 'POSITIVE Z-SCORE' if z_score >= threshold else 'NEGATIVE Z-SCORE'
        candle_time = pd.Timestamp(row['Candle Time'])
        key = (row['Symbol'], timeframe, candle_time.isoformat(), side)
        if key in seen:
            continue
        seen.add(key)
        alerts.append({
            'Candle Time': candle_time, 'Symbol': row['Symbol'], 'Alert': side,
            'Price': float(row['Price']), 'Z-Score': z_score,
            'Z-Score Change': float(row['Z-Score Change']), 'Timeframe': timeframe,
        })
    return alerts


def order_flow_summary(candles):
    """Inflow and outflow in USDT, so assets with different unit prices are comparable."""
    notional = candles['volume'] * candles['close']
    is_up = candles['close'] >= candles['open']
    inflow, outflow = float(notional[is_up].sum()), float(notional[~is_up].sum())
    total = inflow + outflow
    return {
        'Inflow': inflow, 'Outflow': outflow, 'Net Flow': inflow - outflow,
        'Money Flow Signal': (inflow - outflow) / total if total > 0 else 0.0,
    }


def with_flow_history(candles):
    frame = candles.copy()
    frame['net_flow_per_candle'] = (
        frame['volume'] * frame['close'] * np.where(frame['close'] >= frame['open'], 1, -1)
    )
    frame['cumulative_net_flow'] = frame['net_flow_per_candle'].cumsum()
    return frame


def with_volume_delta(candles):
    frame = candles.copy()
    notional = frame['volume'] * frame['close']
    frame['Buy Volume'] = notional.where(frame['close'] >= frame['open'], 0)
    frame['Sell Volume'] = notional.where(frame['close'] < frame['open'], 0)
    frame['Volume Delta'] = frame['Buy Volume'] - frame['Sell Volume']
    frame['Cumulative Volume Delta'] = frame['Volume Delta'].cumsum()
    frame['20-Candle Volume Delta'] = frame['Volume Delta'].rolling(20).sum()
    return frame


def qs_factor_row(symbol, candles):
    current = candles.iloc[-1]
    is_up = candles['close'] >= candles['open']
    inflow, outflow = float(candles.loc[is_up, 'volume'].sum()), float(candles.loc[~is_up, 'volume'].sum())
    total = inflow + outflow
    return {
        'symbol': symbol,
        'price': float(current['close']),
        'momentum': float(current.get('momentum', 0.0)),
        'money_flow_signal': (inflow - outflow) / total if total > 0 else 0.0,
        # ATR is a price distance; compare it as a percentage across unlike price scales.
        'atr14': float(current.get('atr14', 0.0)) / float(current['close']),
        'z_score_raw': float(current.get('z_score', 0.0)),
        'rsi': float(current.get('rsi', 50.0)),
        'trend_bullish': bool(current['close'] > current.get('ema_50', 0)),
    }


def add_qs_scores(factors):
    """US QS formula: 50% momentum, 30% flow, -10% volatility, 10% trend."""
    data = factors.copy()

    def cross_section_z(column):
        # A factor with no spread is neutral; it must not blank out the other factors.
        spread = data[column].std()
        if not spread or pd.isna(spread):
            return pd.Series(0.0, index=data.index)
        return ((data[column] - data[column].mean()) / spread).fillna(0.0)

    data['z_momentum'] = cross_section_z('momentum')
    data['z_flow'] = cross_section_z('money_flow_signal')
    data['z_volatility'] = cross_section_z('atr14')
    data['z_trend'] = cross_section_z('z_score_raw')
    data['qs_score'] = (
        0.50 * data['z_momentum'] + 0.30 * data['z_flow']
        - 0.10 * data['z_volatility'] + 0.10 * data['z_trend']
    ).fillna(0)
    return data.sort_values('qs_score', ascending=False)


def relabel_text(value):
    """Translate engine wording so the UI never reports a crypto signal as QQQ/NQ/VIX."""
    text = str(value).replace('Refresh Indices Data', 'Refresh Derivative Overview')
    return re.sub(r'\b(QQQ|NQ|VIX)\b', lambda match: ENGINE_LABELS[match.group(1)], text)


def relabel_frame(frame):
    result = frame.rename(columns=relabel_text).copy()
    for column in result.columns:
        if 'Reason' in column:
            result[column] = result[column].map(relabel_text, na_action='ignore')
    return result


def pair_alert_inputs(state):
    """Present the stored crypto alerts to the shared engine under its symbol names."""
    rows = [
        {**row, 'Symbol': ENGINE_SYMBOLS[row['Symbol']]}
        for row in state.get(ALERTS_KEY, []) if row.get('Symbol') in ENGINE_SYMBOLS
    ]
    return {
        'daily_alerts': rows,
        'alert_timeframe': state.get(ALERT_TF_KEY),
        'alert_threshold': state.get(ALERT_THRESHOLD_KEY, 1.0),
    }


def build_pair_signals(btc, eth, timeframe, threshold, as_of, alert_inputs):
    return relabel_frame(build_qqq_nq_signals(btc, eth, timeframe, threshold, as_of=as_of, **alert_inputs))


def cost_in_price_terms(execution, start, slippage_bps, fee_bps):
    """Convert percentage costs to the engine's fixed price units at the median close."""
    window = execution.loc[execution.index >= pd.Timestamp(start)]
    reference = float((window if not window.empty else execution)['close'].median())
    return reference * slippage_bps / 10_000, reference * fee_bps / 10_000


def pair_figure(day, threshold):
    # Same chart contract as the QQQ/NQ view: one Ghana day, fixed -100..100 scale,
    # asset lines distinguished by colour and dash, shape-coded new alerts.
    figure = go.Figure()
    x = pd.to_datetime(day['Signal Time (Ghana)'], utc=True).dt.tz_localize(None)
    for column, color, dash in [
        ('BTC Score', '#4C78A8', 'solid'), ('ETH Score', '#F28E2B', 'dash'),
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
        title='BTC and ETH demand/supply confirmation',
        xaxis_title='Signal time — Ghana (GMT/UTC)', yaxis_title='Direction score',
        yaxis=dict(range=[-105, 105]), height=440,
        legend=dict(orientation='h', y=1.18), margin=dict(t=100),
    )
    return figure


def gex_figure(summary, asset, label):
    frame = summary['gex_profile'].reset_index()
    frame.columns = ['strike', 'gamma_exposure']
    figure = go.Figure(go.Bar(x=frame['strike'], y=frame['gamma_exposure'], name='Gamma Exposure'))
    figure.add_vline(x=summary['zero_gamma_level'], line_width=2, line_dash='dash', line_color='yellow',
                     annotation_text='Zero Gamma', annotation_position='top right')
    figure.add_vline(x=summary['current_price'], line_width=2, line_color='cyan',
                     annotation_text='Current Price', annotation_position='top left')
    figure.update_layout(title=f'{label} for {asset}', xaxis_title='Strike Price',
                         yaxis_title='Gamma Exposure', template='plotly_dark')
    return figure


QUANT_FORMULAS = [
    ('Log return', r'r_t = \ln\left(\frac{C_t}{C_{t-1}}\right)'),
    ('Bad-print wick cap (m = rolling 50-bar median of (H − L)/C)',
     r'H^{*} = \min\left(H,\ \max(O, C) + 5\,m\,C\right),\quad L^{*} = \max\left(L,\ \min(O, C) - 5\,m\,C\right)'),
    ('Yang-Zhang volatility (annualized, N = bars per year)',
     r'\sigma_{YZ}^2 = \sigma_o^2 + k\,\sigma_c^2 + (1-k)\,\sigma_{RS}^2,\quad '
     r'k = \frac{0.34}{1.34 + \frac{n+1}{n-1}},\quad \sigma_{ann} = \sqrt{N\,\sigma_{YZ}^2}'),
    ('Volatility fallback for noisy high/low data, and the bar rate N',
     r'\sigma_{ann} = \sigma_{cc}\sqrt{N}\ \text{ if }\ \sigma_{YZ} > 1.5\,\sigma_{cc},\quad '
     r'N = \frac{\text{bars} - 1}{\text{calendar days}} \times 365'),
    ('Rogers-Satchell term', r'\sigma_{RS}^2 = \overline{u(u-c) + d(d-c)},\quad '
     r'u = \ln\tfrac{H}{O},\ d = \ln\tfrac{L}{O},\ c = \ln\tfrac{C}{O}'),
    ('Hurst exponent (H > 0.5 trend, H < 0.5 mean reversion)',
     r'\operatorname{std}(x_{t+\tau} - x_t) \propto \tau^{H},\quad x_t = \ln C_t'),
    ('Lo-MacKinlay variance ratio (q = 8)',
     r'VR(q) = \frac{\operatorname{Var}(r_t + \dots + r_{t-q+1})}{q\,\operatorname{Var}(r_t)},\quad '
     r'z = \frac{VR(q) - 1}{\sqrt{\frac{2(2q-1)(q-1)}{3qn}}}'),
    ('Ornstein-Uhlenbeck half-life', r'\Delta x_t = \alpha + \beta x_{t-1} + \varepsilon_t,\quad '
     r'HL = -\frac{\ln 2}{\ln(1+\beta)}'),
    ('Probabilistic Sharpe ratio (per-bar SR, skew γ₃, kurtosis γ₄)',
     r'PSR = \Phi\!\left(\frac{\widehat{SR}\sqrt{n-1}}{\sqrt{1 - \gamma_3 \widehat{SR} + '
     r'\frac{\gamma_4 - 1}{4}\widehat{SR}^2}}\right)'),
    ('Cornish-Fisher VaR 95% (S skew, K excess kurtosis, z = −1.645)',
     r'z_{cf} = z + \frac{(z^2-1)S}{6} + \frac{(z^3-3z)K}{24} - \frac{(2z^3-5z)S^2}{36},\quad '
     r'VaR = -(\mu + z_{cf}\,\sigma)'),
    ('Regime weight', r'a = \operatorname{clip}\left(\frac{H - 0.45}{0.10},\,0,\,1\right)'),
    ('Quant Score (t_drift = 48-bar drift t-stat, Z = overview Z-Score)',
     r'Q_t = 0.7\left[a\tanh\tfrac{t_{drift}}{2} - (1-a)\tanh\tfrac{Z_t}{2}\right] + '
     r'0.3\tanh\tfrac{FlowZ_t}{2}'),
    ('Information coefficient (Spearman, next bar)',
     r'IC = \rho_s(Q_t,\,r_{t+1}),\quad t_{IC} = IC\sqrt{\frac{n-2}{1-IC^2}}'),
    ('Triple-barrier trade result in R (fees f in bps)',
     r'R = \begin{cases} +RR & \text{target first} \\ -1 & \text{stop first} \\ '
     r'\frac{s\,(C_{exit} - C_{entry})}{m\cdot ATR} & \text{time exit} \end{cases}'
     r' \;-\; \frac{f}{10^4}\cdot\frac{C_{entry}}{m\cdot ATR}'),
    ('Kelly fraction (p win rate, b = avg win / avg loss)', r'f^* = p - \frac{1-p}{b}'),
    ('Position size (E equity, ρ your max risk, m ATR multiple)',
     r'\text{risk} = \min\left(\rho,\ \tfrac{f^*}{2}\right),\quad '
     r'\text{units} = \min\left(\frac{E\cdot \text{risk}}{m\cdot ATR},\ \frac{3E}{C}\right)'),
]


def render_quant_settings(prefix='dc_q', fee_default=10.0):
    with st.expander('Quant Edge Settings (position sizing and backtest rule)'):
        c1, c2, c3, c4 = st.columns(4)
        settings = {
            'equity': float(c1.number_input('Account equity (USDT)', 10.0, 1e9, 1000.0, 100.0, key=f'{prefix}_equity')),
            'risk_pct': float(c2.number_input('Max risk per trade (%)', 0.1, 5.0, 1.0, 0.1, key=f'{prefix}_risk')),
            'atr_multiple': float(c3.number_input('Stop: ATR(14) multiple', 0.5, 5.0, 1.5, 0.25, key=f'{prefix}_atr')),
            'reward_risk': float(c4.number_input('Reward / risk', 0.5, 5.0, 2.0, 0.25, key=f'{prefix}_rr')),
        }
        c5, c6, c7 = st.columns(3)
        settings['threshold'] = float(c5.number_input(
            'Entry threshold |Q|', 0.05, 0.9, 0.25, 0.05, key=f'{prefix}_threshold'))
        settings['max_hold'] = int(c6.number_input('Max hold (bars)', 4, 500, 48, 4, key=f'{prefix}_hold'))
        settings['fee_bps'] = float(c7.number_input(
            'Round-trip fees + slippage (bps)', 0.0, 100.0, fee_default, 0.5, key=f'{prefix}_fee'))
        st.caption(
            'Changing these recalculates the quant layer from the loaded candles; no refresh is needed. '
            'Check the fee figure against your Binance fee tier. Funding payments are not included.'
        )
    return settings


def build_quant_table(histories, timeframe, settings, round_the_clock=True, conversions=None):
    """round_the_clock=False measures the bar rate from timestamps (US market sessions).

    conversions(histories) -> {symbol: quote-to-account rate}; symbols left out use 1.0.
    """
    bars_per_day = BARS_PER_DAY.get(timeframe) if round_the_clock else None
    rates = conversions(histories) if conversions else {}
    rows = []
    for symbol, candles in histories.items():
        try:
            rows.append(quant_edge_row(symbol, candles, bars_per_day, settings, rates.get(symbol, 1.0)))
        except Exception as exc:  # One bad series must not hide the other symbols.
            rows.append({'Symbol': symbol, 'Verdict': 'ERROR', 'Reason': str(exc)})
    return pd.DataFrame(rows)


def color_verdict(value):
    return {
        'LONG': 'color: #22c55e; font-weight: 600', 'SHORT': 'color: #ef4444; font-weight: 600',
        'NO EDGE': 'color: #f59e0b', 'ERROR': 'color: #ef4444',
    }.get(value, 'color: gray')


def render_quant_edge(quant, timeframe, settings):
    st.markdown('#### 🧮 Quant Edge Layer')
    if quant.empty:
        st.info('Click Refresh Derivative Overview to load candles for the quant layer.')
        return
    quant = quant.reindex(columns=list(dict.fromkeys([*quant.columns, '$ Risk', '$ Expected', 'Regime'])))
    actionable = quant[quant['Verdict'].isin(['LONG', 'SHORT'])]
    metrics = st.columns(4)
    metrics[0].metric('Actionable setups', f'{len(actionable)} / {len(quant)}')
    metrics[1].metric('Total $ at risk', f"${actionable['$ Risk'].sum():,.2f}" if len(actionable) else '$0.00')
    metrics[2].metric('Expected $ (backtest E[R])',
                      f"${actionable['$ Expected'].sum():+,.2f}" if len(actionable) else '$0.00')
    metrics[3].metric('Trending regimes', int(quant['Regime'].eq('Trending').sum()))
    st.caption(
        f'Computed on the loaded {timeframe} candles. LONG/SHORT appears only when |Q| clears the threshold, '
        f'the same rule backtested at least 10 trades with positive expectancy after fees, Kelly > 0, '
        f'and the signal\'s IC is positive. Everything else is FLAT or NO EDGE.'
    )

    st.markdown('##### Trade plan')
    plan_columns = ['Symbol', 'Verdict', 'Quant Score Q', 'Entry', 'Stop', 'Target', 'Risk %',
                    'Size (units)', 'Notional', 'Leverage', '$ Risk', '$ Expected', 'Reason']
    plan = quant.reindex(columns=plan_columns)
    st.dataframe(
        plan.style.format({
            'Quant Score Q': '{:+.2f}', 'Entry': format_price, 'Stop': format_price, 'Target': format_price,
            'Risk %': '{:.2%}', 'Size (units)': '{:,.4f}', 'Notional': '${:,.2f}', 'Leverage': '{:.2f}x',
            '$ Risk': '${:,.2f}', '$ Expected': '${:+,.2f}',
        }, na_rep='—').map(color_verdict, subset=['Verdict']),
        width='stretch', hide_index=True,
    )

    st.markdown('##### Statistical matrix')
    stat_columns = ['Symbol', 'Regime', 'Hurst H', 'Variance Ratio', 'VR z', 'OU Half-Life (bars)',
                    'Volatility (ann.)', 'Vol Method', 'PSR (drift > 0)', 'CF VaR 95% (bar)', 'CVaR 95% (bar)', 'IC', 'IC t',
                    'Trend Weight a', 'BT Trades', 'BT Win Rate', 'BT Expectancy R', 'BT Expectancy t',
                    'BT Profit Factor', 'BT Kelly f*']
    stats = quant.reindex(columns=stat_columns)
    st.dataframe(
        stats.style.format({
            'Hurst H': '{:.3f}', 'Variance Ratio': '{:.3f}', 'VR z': '{:+.2f}', 'OU Half-Life (bars)': '{:,.0f}',
            'Volatility (ann.)': '{:.1%}', 'PSR (drift > 0)': '{:.1%}', 'CF VaR 95% (bar)': '{:.2%}',
            'CVaR 95% (bar)': '{:.2%}', 'IC': '{:+.3f}', 'IC t': '{:+.2f}', 'Trend Weight a': '{:.2f}',
            'BT Trades': '{:.0f}', 'BT Win Rate': '{:.1%}', 'BT Expectancy R': '{:+.3f}',
            'BT Expectancy t': '{:+.2f}', 'BT Profit Factor': '{:.2f}', 'BT Kelly f*': '{:+.3f}',
        }, na_rep='—'),
        width='stretch', hide_index=True,
    )

    with st.expander('Formulas used by the Quant Edge Layer'):
        for label, formula in QUANT_FORMULAS:
            st.markdown(f'**{label}**')
            st.latex(formula)
    st.warning(
        'Backtest figures are in-sample: the regime weight uses the full loaded history, and about 60 days '
        'of candles is a short sample. A positive expectancy here is evidence, not a guarantee. '
        'Forward-test with small size before trusting it.'
    )


SCAN_CANDLES = 1000   # One Binance request per symbol (≈42 days of 1h bars).
SCAN_WORKERS = 6      # Parallel downloads; 200 symbols × weight 5 stays well inside Binance's 2400/min limit.
SCAN_KEY = 'dc_quant_scan'
SCAN_MAX_SYMBOLS = 200
VERDICT_ORDER = {'LONG': 0, 'SHORT': 0, 'FLAT': 1, 'NO EDGE': 2, 'ERROR': 3}


def scan_quant_edges(symbols, fetch, timeframe, settings, workers=SCAN_WORKERS, progress=None,
                     candles=SCAN_CANDLES, min_bars=200, round_the_clock=True, conversions=None):
    """Fetch each symbol in parallel and compute its quant edge row.

    Returns (rows DataFrame ranked by volume order, failed symbols, {symbol: candles}).
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
            # Too little history makes Hurst, IC and the backtest meaningless.
            if frame is None or len(frame) < min_bars:
                failed.append(symbol)
            else:
                histories[symbol] = frame
            if progress:
                progress(done / len(symbols), symbol)
    ordered = {symbol: histories[symbol] for symbol in symbols if symbol in histories}
    table = build_quant_table(ordered, timeframe, settings, round_the_clock, conversions)
    return with_rank(table, symbols), failed, ordered


def with_rank(table, symbols):
    """Number rows by their position in the scanned list (volume rank for crypto)."""
    if not table.empty:
        table.insert(0, 'Volume Rank', [symbols.index(symbol) + 1 for symbol in table['Symbol']])
    return table


def rank_quant_scan(table):
    """Actionable setups first, strongest backtest expectancy first within each verdict."""
    if table.empty:
        return table
    ranked = table.assign(_order=table['Verdict'].map(VERDICT_ORDER).fillna(4))
    return ranked.sort_values(
        ['_order', 'BT Expectancy R'], ascending=[True, False], na_position='last', kind='stable',
    ).drop(columns='_order').reset_index(drop=True)


RISK_TOLERANCE_USD = 0.005  # "Exactly $10" means $10.00 to the cent.


def risk_leverage_funnel(table, risk_usd=10.0, max_leverage=0.4, tolerance=RISK_TOLERANCE_USD):
    """Apply the watch list filters in order and count the survivors of each step.

    Returns (matches ranked by backtest expectancy, [(step label, rows left), ...]).
    Steps: verdict LONG/SHORT → regime Mean-Reverting → $ risk equal to risk_usd
    (within tolerance) → leverage strictly below max_leverage.
    """
    needed = {'Verdict', 'Regime', '$ Risk', 'Leverage'}
    if table.empty or not needed <= set(table.columns):
        return table.iloc[0:0], []
    risk = pd.to_numeric(table['$ Risk'], errors='coerce')
    leverage = pd.to_numeric(table['Leverage'], errors='coerce')
    steps = [
        ('Verdict LONG / SHORT', table['Verdict'].isin(['LONG', 'SHORT'])),
        ('Regime Mean-Reverting', table['Regime'].eq('Mean-Reverting')),
        (f'$ Risk = ${risk_usd:,.2f}', (risk - risk_usd).abs() <= tolerance),
        (f'Leverage < {max_leverage:.2f}x', leverage < max_leverage),
    ]
    mask, funnel = pd.Series(True, index=table.index), []
    for label, condition in steps:
        mask &= condition.fillna(False)
        funnel.append((label, int(mask.sum())))
    matches = table[mask]
    if 'BT Expectancy R' in matches:
        matches = matches.sort_values('BT Expectancy R', ascending=False, na_position='last', kind='stable')
    return matches, funnel


def render_risk_leverage_watchlist(table, key_prefix, rank_label, scan=None):
    st.markdown('##### 🎯 Risk & Leverage Watch List — Mean-Reverting LONG / SHORT')
    c1, c2 = st.columns(2)
    risk_usd = float(c1.number_input(
        'Exact $ risk per trade', min_value=0.01, max_value=1_000_000.0, value=10.0, step=1.0,
        format='%.2f', key=f'{key_prefix}_wl_risk',
        help='$ Risk must equal this amount to the cent. $ Risk = equity × Max risk per trade (%), '
             'or less when half-Kelly or the 3x leverage cap is smaller.',
    ))
    max_leverage = float(c2.number_input(
        'Leverage below', min_value=0.01, max_value=3.0, value=0.40, step=0.05, format='%.2f',
        key=f'{key_prefix}_wl_leverage', help='Keeps assets whose Leverage is strictly below this value.',
    ))
    matches, funnel = risk_leverage_funnel(table, risk_usd, max_leverage)
    st.caption('Filters applied in order: ' + ' → '.join(f'{label}: {count}' for label, count in funnel))
    if scan is not None and not matches.empty:
        try:
            saved = watchlist_history.record_matches(
                matches.rename(columns={rank_label: 'Volume Rank'}), scan['scanned_at'], scan['timeframe'],
                risk_usd, max_leverage,
            )
            if saved:
                st.caption(f'💾 Saved {saved} match(es) to the daily watch list history.')
        except OSError as exc:
            st.warning(f'Could not save the watch list history: {exc}')
    if matches.empty:
        st.info('No asset passes all four filters on this scan.')
        return
    columns = [rank_label, 'Symbol', 'Verdict', 'Regime', 'Quant Score Q', '$ Risk', 'Leverage',
               'Entry', 'Stop', 'Target', 'Size (units)', 'Notional', 'BT Expectancy R', 'BT Win Rate',
               'Hurst H', 'Reason']
    st.dataframe(
        matches.reindex(columns=columns).style.format({
            rank_label: '{:.0f}', 'Quant Score Q': '{:+.2f}', '$ Risk': '${:,.2f}', 'Leverage': '{:.2f}x',
            'Entry': format_price, 'Stop': format_price, 'Target': format_price, 'Size (units)': '{:,.4f}',
            'Notional': '${:,.2f}', 'BT Expectancy R': '{:+.3f}', 'BT Win Rate': '{:.1%}', 'Hurst H': '{:.3f}',
        }, na_rep='—').map(color_verdict, subset=['Verdict']),
        width='stretch', hide_index=True,
    )
    render_watchlist_flow(matches, key_prefix, scan['timeframe'] if scan else None)


def watchlist_flow_options(matches):
    """Drop-down choices for the watch list: one per symbol, in table order, labelled with the verdict."""
    if matches is None or matches.empty or 'Symbol' not in matches:
        return {}
    options = {}
    for _, row in matches.iterrows():
        symbol = row['Symbol']
        if symbol not in options.values():
            options[f"{symbol} — {row.get('Verdict', '')}".strip(' —')] = symbol
    return options


def render_watchlist_flow(matches, key_prefix, timeframe=None):
    """Real Money Flow (taker buy vs sell, CVD, price–CVD cross watch) for any asset on the watch list."""
    from real_flow import INTERVAL_MS
    from real_flow_panel import render_flow_view

    options = watchlist_flow_options(matches)
    if not options:
        return
    st.markdown('###### 💵 Real money flow for a watch-list asset')
    c1, c2 = st.columns([1, 2])
    label = c1.selectbox('Watch-list asset', list(options), key=f'{key_prefix}_wl_flow_symbol')
    intervals = list(INTERVAL_MS)
    interval = c2.radio('Candles', intervals, horizontal=True, key=f'{key_prefix}_wl_flow_interval',
                        index=intervals.index(timeframe) if timeframe in intervals else intervals.index('1h'))
    symbol = options[label]
    row = matches[matches['Symbol'] == symbol].iloc[0]
    st.caption(
        f"Watch-list plan for {symbol}: **{row.get('Verdict', '—')}** · entry {format_price(row.get('Entry'))} · "
        f"stop {format_price(row.get('Stop'))} · target {format_price(row.get('Target'))} · "
        f"risk \\${row.get('$ Risk', float('nan')):,.2f} · leverage {row.get('Leverage', float('nan')):.2f}x. "
        'The flow below is the same Binance taker buy/sell logic as the 💵 Real Money Flow panel.'
    )
    render_flow_view(symbol, interval, key=f'{key_prefix}_wl_flow')


def render_watchlist_history(key_prefix):
    st.markdown('##### 📅 Watch List History — by day')
    history = watchlist_history.load_history()
    dates = watchlist_history.available_dates(history)
    if not dates:
        st.info('Nothing stored yet. Matches are saved automatically each time a scan shows them.')
        return
    today = pd.Timestamp.now(tz=watchlist_history.GHANA_TZ).date()
    c1, c2 = st.columns([1, 2])
    day = c1.date_input('Day (Ghana time)', value=today, key=f'{key_prefix}_wl_day')
    c2.caption('Days with stored matches: ' + ', '.join(dates[:14]) + (' …' if len(dates) > 14 else ''))
    summary = watchlist_history.day_summary(history, day)
    if summary.empty:
        st.info(f'No watch list matches were stored on {day:%d %b %Y}.')
        return
    rows = watchlist_history.rows_for_day(history, day)
    m1, m2, m3 = st.columns(3)
    m1.metric('Assets on the watch list', len(summary))
    m2.metric('LONG / SHORT', f"{int(summary['Verdict'].eq('LONG').sum())} / {int(summary['Verdict'].eq('SHORT').sum())}")
    m3.metric('Scans that day', rows['Scan Time (UTC)'].nunique())
    st.caption('One row per asset. Trade plan columns are from the latest scan it matched that day.')
    st.dataframe(
        summary.style.format({
            'Quant Score Q': '{:+.2f}', '$ Risk': '${:,.2f}', 'Leverage': '{:.2f}x', 'Entry': format_price,
            'Stop': format_price, 'Target': format_price, 'Size (units)': '{:,.4f}', 'Notional': '${:,.2f}',
            'BT Expectancy R': '{:+.3f}', 'BT Win Rate': '{:.1%}',
        }, na_rep='—').map(color_verdict, subset=['Verdict']),
        width='stretch', hide_index=True,
    )
    if st.checkbox('Show every stored scan row for this day', key=f'{key_prefix}_wl_rows'):
        st.dataframe(rows, width='stretch', hide_index=True)
    st.download_button(
        f'Download {day:%Y-%m-%d} watch list (CSV)', rows.to_csv(index=False).encode('utf-8'),
        file_name=f'watchlist_{day:%Y-%m-%d}.csv', mime='text/csv', key=f'{key_prefix}_wl_download',
    )


def near_trigger(table, threshold, closeness=0.8):
    """FLAT rows whose backtest already passes and whose |Q| is within reach of the threshold."""
    if table.empty or 'Backtest Gate' not in table:
        return table.iloc[0:0]
    q = table['Quant Score Q'].abs()
    mask = table['Verdict'].eq('FLAT') & table['Backtest Gate'].eq('PASS') & (q >= closeness * threshold)
    return table[mask].sort_values('Quant Score Q', key=abs, ascending=False)


def render_quant_scan(deps, settings):
    st.markdown(f'#### 🔎 Top {SCAN_MAX_SYMBOLS} Derivatives — Quant Edge Scan')
    st.caption(
        'Runs the Quant Edge Layer on the most active Binance USD-M perpetuals by 24-hour USDT volume, '
        f'using {SCAN_CANDLES} candles each and the Quant Edge Settings above. '
        f'Takes about 1–4 minutes for {SCAN_MAX_SYMBOLS} perpetuals.'
    )
    c1, c2, c3 = st.columns(3)
    timeframe = c1.selectbox('Scan timeframe', ['15m', '1h', '4h'], index=1, key='dc_qscan_tf')
    count = int(c2.number_input(
        'Number of perpetuals', 10, SCAN_MAX_SYMBOLS, SCAN_MAX_SYMBOLS, 10, key='dc_qscan_count',
    ))
    only_actionable = c3.checkbox('Show only LONG / SHORT', value=False, key='dc_qscan_only')

    if st.button(f'Scan Top {count} for Quant Edge', type='primary', key='dc_qscan_run'):
        try:
            symbols = top_usdm_symbols(count)
        except Exception as exc:
            st.error(f'Could not rank Binance perpetuals: {exc}')
            symbols = []
        if symbols:
            run_quant_scan(SCAN_KEY, symbols, deps.fetch_history, timeframe, settings)

    render_quant_scan_results(
        SCAN_KEY, 'dc_qscan', settings, only_actionable,
        empty_hint='Click Scan to rank the top perpetuals by quant edge.', min_bars=200,
        correlation_warning=(
            'Most altcoins move with BTC, so several LONGs (or SHORTs) at once behave like one larger bet. '
            'Consider taking only the top few, or splitting your risk between them.'
        ),
        risk_watchlist=True,
    )
    render_watchlist_history('dc_qscan')


def run_quant_scan(state_key, symbols, fetch, timeframe, settings, **scan_options):
    """Scan with a progress bar and store the result under state_key."""
    bar = st.progress(0.0, text='Starting scan...')
    table, failed, histories = scan_quant_edges(
        symbols, fetch, timeframe, settings,
        progress=lambda share, symbol: bar.progress(share, text=f'Loaded {symbol}'), **scan_options,
    )
    bar.empty()
    st.session_state[state_key] = {
        'table': table, 'failed': failed, 'histories': histories, 'symbols': list(symbols),
        'timeframe': timeframe, 'settings': tuple(sorted(settings.items())),
        'round_the_clock': scan_options.get('round_the_clock', True),
        'conversions': scan_options.get('conversions'),
        'scanned_at': pd.Timestamp.now(tz='UTC'),
    }


def render_quant_scan_results(state_key, key_prefix, settings, only_actionable, empty_hint, min_bars,
                              correlation_warning, rank_label='Volume Rank', decorate=None, extra_columns=(),
                              extra_formats=None, risk_watchlist=False):
    """Shared results view: metrics, ranked table, watchlist and CSV export.

    decorate(table) may add market-specific columns, listed in extra_columns.
    """
    state = st.session_state
    scan = state.get(state_key)
    if not scan:
        st.info(empty_hint)
        return
    fingerprint = tuple(sorted(settings.items()))
    if scan['settings'] != fingerprint:
        st.warning('Quant Edge Settings changed since this scan.')
        if st.button('Recalculate with new settings (no download)', key=f'{key_prefix}_recalc'):
            table = build_quant_table(
                scan['histories'], scan['timeframe'], settings, scan.get('round_the_clock', True),
                scan.get('conversions'),
            )
            scan.update(table=with_rank(table, scan['symbols']), settings=fingerprint)
            st.rerun()

    table = rank_quant_scan(scan['table'])
    if table.empty:
        st.warning('The scan returned no usable data. Try again in a moment.')
        return
    if decorate:
        table = decorate(table)
    table = table.rename(columns={'Volume Rank': rank_label})
    verdicts = table['Verdict'].value_counts()
    actionable = table[table['Verdict'].isin(['LONG', 'SHORT'])]
    metrics = st.columns(5)
    metrics[0].metric('Scanned', f"{len(table)} / {len(scan['symbols'])}")
    metrics[1].metric('LONG', int(verdicts.get('LONG', 0)))
    metrics[2].metric('SHORT', int(verdicts.get('SHORT', 0)))
    metrics[3].metric('FLAT / NO EDGE', f"{int(verdicts.get('FLAT', 0))} / {int(verdicts.get('NO EDGE', 0))}")
    metrics[4].metric('Total $ risk if all taken', f"${actionable['$ Risk'].sum():,.2f}")
    st.caption(
        f"{scan['timeframe']} candles · scanned {scan['scanned_at'].tz_convert('Africa/Accra'):%d %b %Y %H:%M} Ghana. "
        'LONG/SHORT first, ranked by backtest expectancy (R per trade).'
    )
    if scan['failed']:
        st.caption(f"Skipped (no data or under {min_bars} candles): {', '.join(scan['failed'])}")
    if len(actionable) > 3:
        st.warning(correlation_warning)

    shown = actionable if only_actionable else table
    columns = [rank_label, 'Symbol', 'Verdict', 'Quant Score Q', 'Regime', 'Hurst H', 'IC', 'IC t',
               'BT Trades', 'BT Win Rate', 'BT Expectancy R', 'BT Profit Factor', 'BT Kelly f*',
               'Entry', 'Stop', 'Target', 'Size (units)', *extra_columns, 'Leverage', '$ Risk', 'Reason']
    st.dataframe(
        shown.reindex(columns=columns).style.format({
            rank_label: '{:.0f}', 'Quant Score Q': '{:+.2f}', 'Hurst H': '{:.3f}', 'IC': '{:+.3f}',
            'IC t': '{:+.2f}', 'BT Trades': '{:.0f}', 'BT Win Rate': '{:.1%}', 'BT Expectancy R': '{:+.3f}',
            'BT Profit Factor': '{:.2f}', 'BT Kelly f*': '{:+.3f}', 'Entry': format_price, 'Stop': format_price,
            'Target': format_price, 'Size (units)': '{:,.4f}', 'Leverage': '{:.2f}x', '$ Risk': '${:,.2f}',
            **{column: '{:,.2f}' for column in extra_columns}, **(extra_formats or {}),
        }, na_rep='—').map(color_verdict, subset=['Verdict']),
        width='stretch', hide_index=True, height=min(460, 38 + 35 * max(len(shown), 1)),
    )

    if risk_watchlist:
        render_risk_leverage_watchlist(table, key_prefix, rank_label, scan)

    watch = near_trigger(table, settings['threshold'])
    st.markdown('##### 👀 Watchlist — FLAT but close to triggering')
    if watch.empty:
        st.caption('Nothing FLAT is within 80% of the entry threshold with a passing backtest.')
    else:
        st.caption(
            f"These pass the backtest gate and have |Q| ≥ {0.8 * settings['threshold']:.2f}. "
            'Refresh at the next candle close; they may turn LONG or SHORT.'
        )
        st.dataframe(
            watch.reindex(columns=['Symbol', 'Quant Score Q', 'Regime', 'BT Expectancy R', 'BT Win Rate', 'Entry'])
            .style.format({'Quant Score Q': '{:+.2f}', 'BT Expectancy R': '{:+.3f}', 'BT Win Rate': '{:.1%}',
                           'Entry': format_price}, na_rep='—'),
            width='stretch', hide_index=True,
        )
    st.download_button(
        'Download scan (CSV)', table.to_csv(index=False).encode('utf-8'),
        file_name=f"{key_prefix}_quant_edge_{scan['timeframe']}_{scan['scanned_at']:%Y%m%d_%H%M}.csv",
        mime='text/csv', key=f'{key_prefix}_download',
    )


def render_overview(deps):
    st.divider()
    st.subheader('🏛️ Top Derivatives Overview')
    st.write('Binance USD-M perpetuals: ' + ', '.join(s.split('/')[0] for s in OVERVIEW_SYMBOLS) + '.')
    timeframe = st.selectbox('Select overview timeframe', ['15m', '1h', '4h'], index=1, key='dc_overview_tf')

    with st.expander('Derivative Alert Settings'):
        left, right = st.columns(2)
        enabled = left.checkbox('Enable overview alerts', value=False, key='dc_alert_enabled')
        threshold = float(right.number_input(
            'Absolute Z-Score alert level', min_value=0.1, max_value=5.0, value=1.0, step=0.1,
            key='dc_alert_threshold_input',
        ))
        telegram = st.checkbox(
            'Send qualifying alerts to Telegram', value=True, key='dc_alert_telegram',
            help='Uses the Telegram Bot Token and Chat ID entered in the sidebar.',
        )
        st.caption(
            'Triggers when the table Z-Score reaches the positive level or falls to the negative '
            'level. Alerts are checked when the overview is refreshed.'
        )

    quant_settings = render_quant_settings()

    state = st.session_state
    if st.button('Refresh Derivative Overview', key='dc_overview_refresh'):
        as_of = pd.Timestamp.now(tz='UTC')
        rows, missing, daily_alerts, histories = [], [], [], {}
        with st.spinner('Fetching Binance perpetual data...'):
            for symbol in OVERVIEW_SYMBOLS:
                candles = deps.fetch_history(symbol, timeframe=timeframe, silent=True)
                if candles is None or candles.empty:
                    missing.append(symbol)
                    continue
                rows.append(build_overview_row(symbol, candles, timeframe))
                histories[symbol] = candles
                daily_alerts.extend(build_daily_alerts(
                    symbol, candles, timeframe, threshold, as_of, deps.build_daily_alert_history,
                ))
        state['dc_overview_rows'] = rows
        state['dc_overview_histories'] = histories
        state['dc_overview_timeframe'] = timeframe
        state['dc_overview_missing'] = missing
        state[ALERTS_KEY] = sorted(daily_alerts, key=lambda alert: alert['Ghana Time'], reverse=True)
        state[ALERT_TF_KEY] = timeframe
        state[ALERT_THRESHOLD_KEY] = threshold
        state['dc_alerts_built_at'] = as_of
        if enabled and rows:
            seen = state.setdefault('dc_alert_keys', set())
            fresh = new_threshold_alerts(rows, threshold, timeframe, seen)
            for alert in fresh:
                if telegram and deps.tg_token and deps.tg_chat_id:
                    deps.send_telegram_alert(
                        deps.tg_token, deps.tg_chat_id, alert['Symbol'], alert['Alert'], alert['Price'],
                        f"Z-Score {alert['Z-Score']:+.2f}",
                        [f'Threshold: +/-{threshold:.2f}',
                         f"Z-Score Change: {alert['Z-Score Change']:+.2f}", f'Timeframe: {timeframe}'],
                    )
            if fresh:
                state['dc_recent_alerts'] = (fresh + state.get('dc_recent_alerts', []))[:100]
                st.toast(f'{len(fresh)} new derivative alert(s) triggered.', icon='🚨')
            else:
                st.info('No derivatives met the selected alert thresholds.')

    rows = state.get('dc_overview_rows')
    if not rows:
        if state.get('dc_overview_missing'):
            st.error('Binance returned no data for: ' + ', '.join(state['dc_overview_missing']))
        else:
            st.info('Click Refresh Derivative Overview to load the table and daily alerts.')
        render_quant_scan(deps, quant_settings)
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
    results_timeframe = state.get('dc_overview_timeframe', timeframe)
    quant = build_quant_table(state.get('dc_overview_histories', {}), results_timeframe, quant_settings)
    if not quant.empty:
        table = table.merge(
            quant.reindex(columns=['Symbol', 'Quant Score Q', 'Regime', 'Verdict']), on='Symbol', how='left',
        )
    columns = [
        'Symbol', 'Verdict', 'Quant Score Q', 'Regime', 'Breadth Ratio', 'Breadth %', 'Price', 'Momentum',
        'RSI', 'Trend', 'Signal Score', 'Volume Ratio', 'Flow Z-Score', 'Z-Score', 'Est. Daily Volume',
    ]
    columns = [column for column in columns if column in table.columns]
    missing = state.get('dc_overview_missing', [])
    if missing:
        st.warning('Binance returned no data for: ' + ', '.join(missing) + '. They will be retried on refresh.')
    st.caption(
        f'Bracketed change = current value minus the previous {results_timeframe} candle. '
        'Breadth Ratio = Advancing / Declining. Breadth % = Advancing / Total. '
        'Signal Score = -Z-Score + ln(1 + Volume Ratio) + Flow Z-Score. '
        'Est. Daily Volume = USDT notional over the last 24 hours of candles. '
        'Verdict, Quant Score Q and Regime come from the Quant Edge Layer below.'
    )
    styled = table[columns].style.format({
        'Price': format_price, 'RSI': '{:.2f}', 'Est. Daily Volume': deps.format_large_number,
        'Volume Ratio': '{:.2f}x', 'Breadth %': '{:.0%}', 'Quant Score Q': '{:+.2f}',
    }, na_rep='n/a').map(deps.color_metrics, subset=['Breadth %']).map(
        color_bracketed_change, subset=['Momentum', 'Signal Score', 'Flow Z-Score', 'Z-Score'],
    )
    if 'Verdict' in columns:
        styled = styled.map(color_verdict, subset=['Verdict'])
    st.dataframe(styled, width='stretch', hide_index=True)
    render_quant_edge(quant, results_timeframe, quant_settings)
    render_quant_scan(deps, quant_settings)

    if state.get('dc_recent_alerts'):
        st.subheader('Recent Derivative Alerts')
        st.dataframe(
            pd.DataFrame(state['dc_recent_alerts']).style.format({
                'Price': format_price, 'Z-Score': '{:+.2f}', 'Z-Score Change': '{:+.2f}',
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
    if st.session_state.get('dc_alert_date') not in dates:
        st.session_state.pop('dc_alert_date', None)
    selected = st.selectbox(
        'Select Ghana alert date', options=dates, index=len(dates) - 1,
        format_func=lambda day: day.strftime('%A, %B %d, %Y'), key='dc_alert_date',
    )
    day = alerts.loc[alerts['Ghana Date'] == selected].copy()
    day['Ghana Date'] = day['Ghana Date'].map(lambda value: value.strftime('%Y-%m-%d'))
    day['Ghana Time'] = pd.to_datetime(day['Ghana Time'], utc=True).dt.tz_convert(
        'Africa/Accra'
    ).dt.strftime('%Y-%m-%d %H:%M')
    st.dataframe(
        day.style.format({'Price': format_price, 'Z-Score': '{:+.2f}', 'Momentum Z': '{:+.2f}'}).map(
            deps.color_metrics, subset=['Z-Score', 'Momentum Z'],
        ),
        width='stretch', hide_index=True,
    )


def render_gex(deps):
    st.divider()
    st.subheader('🛡️ Crypto GEX (Gamma Exposure)')
    st.write(
        'Uses live gamma and open interest from Binance options, expiring within 60 days. '
        'If the chain is unavailable, a clearly labelled volatility-based proxy is shown.'
    )
    asset = st.selectbox('Select asset for GEX analysis', GEX_SYMBOLS, key='dc_gex_asset')
    timeframe = st.selectbox(
        'Select timeframe for price/volatility', ['5m', '15m', '1h', '4h', '12h', '1d'], index=2, key='dc_gex_tf',
    )
    if not st.button('Analyze Crypto GEX', key='dc_gex_run'):
        return
    with st.spinner(f'Analyzing GEX for {asset}...'):
        candles = deps.fetch_history(asset, timeframe=timeframe, silent=True, limit=200)
        if candles is None or candles.empty:
            st.warning('Could not determine the current price from Binance. Try another timeframe.')
            return
        price = float(candles['close'].iloc[-1])
        try:
            profile, label, reason = fetch_options_gex(asset.split('/')[0]), 'Gamma Exposure Profile', None
        except Exception as exc:
            returns = candles['close'].pct_change().dropna()
            volatility = float(returns.std() * np.sqrt(24)) if len(returns) > 1 else None
            profile, label, reason = deps.build_gex_proxy(price, volatility), 'Gamma Exposure PROXY', str(exc)
    summary = deps.summarize_gex(profile, price)
    if reason:
        st.warning(
            f'Showing a synthetic proxy, not real options positioning. Options data unavailable: {reason}'
        )
    else:
        st.caption(
            'GEX = gamma × open interest × contract size, with puts counted as negative. '
            'Gamma scales with the asset price, so compare Total GEX only within one asset.'
        )
    metrics = st.columns(3)
    metrics[0].metric('Total GEX', f"{summary['total_gex']:,.2f}")
    metrics[1].metric('Zero Gamma Level', format_price(summary['zero_gamma_level']))
    metrics[2].metric('Current Price', format_price(summary['current_price']))
    st.plotly_chart(gex_figure(summary, asset, label), width='stretch', key='dc_gex_chart')


def render_order_flow(deps):
    st.divider()
    st.subheader('📊 Real-Time Order Flow')
    left, right = st.columns(2)
    timeframe = left.selectbox('Select Order Flow Timeframe', ['5m', '15m', '1h', '4h'], index=2, key='dc_flow_tf')
    candles_wanted = int(right.number_input(
        'Number of Candles to Analyze', min_value=50, max_value=1000, value=500, step=50, key='dc_flow_candles',
    ))
    state = st.session_state
    if st.button('Refresh Order Flow', key='dc_flow_refresh'):
        rows, history = [], {}
        with st.spinner(f'Calculating order flow for {timeframe} timeframe...'):
            for symbol in OVERVIEW_SYMBOLS:
                candles = deps.fetch_history(symbol, timeframe=timeframe, silent=True, limit=candles_wanted)
                if candles is not None and not candles.empty:
                    rows.append({'Symbol': symbol, **order_flow_summary(candles)})
                    history[symbol] = with_flow_history(candles)
        state['dc_flow_data'] = (rows, candles_wanted, timeframe)
        state['dc_flow_history'] = history
    saved = state.get('dc_flow_data')
    if not saved or not saved[0]:
        st.info('Click Refresh Order Flow to load buying versus selling pressure.')
        return
    rows, used_candles, used_timeframe = saved
    st.caption(
        f'Buying vs. selling pressure in USDT over the last {used_candles} {used_timeframe} candles: '
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
    history = state.get('dc_flow_history', {})
    if history:
        asset = st.selectbox('Select Asset for Historical View', list(history), key='dc_flow_asset')
        st.subheader(f'Cumulative Net Flow for {asset}')
        st.line_chart(history[asset]['cumulative_net_flow'])
        st.dataframe(history[asset].tail(50), width='stretch', height=300)


def render_daily_flow(deps):
    st.divider()
    st.subheader('💰 Daily Inflow / Outflow (BTC, ETH, BNB, SOL, XRP, DOGE)')
    st.caption('Inflow and outflow since Ghana midnight across 5m, 15m, 1h and 4h candles, in USDT notional.')
    state = st.session_state
    # Loaded on demand: Streamlit runs every tab on each rerun, and this needs 24 API calls.
    if st.button('Refresh Daily Flow', key='dc_daily_flow_refresh'):
        rows, failed = [], []
        with st.spinner('Loading today’s flow from Binance...'):
            for symbol in FLOW_SYMBOLS:
                row = {'Symbol': symbol}
                for interval in ('5m', '15m', '1h', '4h'):
                    try:
                        flow = intraday_money_flow(symbol, interval=interval)
                    except Exception:
                        flow = None
                    if flow is None:
                        failed.append(f'{symbol} {interval}')
                    flow = flow or {}
                    row[f'{interval} Inflow'] = flow.get('inflow', np.nan)
                    row[f'{interval} Outflow'] = flow.get('outflow', np.nan)
                    row[f'{interval} Net Flow'] = flow.get('net_flow', np.nan)
                rows.append(row)
        state['dc_daily_flow'] = pd.DataFrame(rows)
        state['dc_daily_flow_failed'] = failed
    table = state.get('dc_daily_flow')
    if table is None:
        st.info('Click Refresh Daily Flow to load today’s inflow and outflow.')
        return
    if state.get('dc_daily_flow_failed'):
        st.warning('No data for: ' + ', '.join(state['dc_daily_flow_failed']))
    columns = [column for column in table.columns if column != 'Symbol']
    st.dataframe(
        table.sort_values('5m Net Flow', ascending=False).style.format(
            {column: deps.format_large_number for column in columns}, na_rep='n/a',
        ).background_gradient(
            subset=['5m Net Flow', '15m Net Flow', '1h Net Flow', '4h Net Flow'], cmap='RdYlGn',
        ),
        width='stretch', hide_index=True,
    )


def render_qs(deps):
    st.divider()
    st.subheader('Crypto Quant Strength (QS) Comparison')
    st.caption(
        'Relative strength across major perpetuals. A higher score means stronger performance '
        'versus the group. QS = 50% momentum + 30% flow − 10% volatility + 10% trend.'
    )
    timeframe = st.session_state.get('dc_overview_tf', '1h')
    state = st.session_state
    if st.button('Refresh Crypto QS Comparison', key='dc_qs_refresh'):
        rows = []
        with st.spinner('Fetching comparison data...'):
            for symbol in QS_SYMBOLS:
                candles = deps.fetch_history(symbol, timeframe=timeframe, silent=True)
                if candles is not None and not candles.empty:
                    rows.append(qs_factor_row(symbol, candles))
        if len(rows) >= 3:
            state['dc_qs_data'] = add_qs_scores(pd.DataFrame(rows))
        else:
            state.pop('dc_qs_data', None)
            st.warning('Could not fetch enough data for the QS comparison.')
    scores = state.get('dc_qs_data')
    if scores is None:
        st.info("Click 'Refresh Crypto QS Comparison' to load the comparison basket.")
        return
    columns = ['symbol', 'qs_score', 'z_momentum', 'z_flow', 'z_volatility', 'z_trend']
    st.dataframe(
        scores[columns].style.format({column: '{:.2f}' for column in columns[1:]}).background_gradient(
            subset=['qs_score'], cmap='viridis',
        ),
        width='stretch', hide_index=True,
    )


def render_pair_signals(deps):
    st.divider()
    st.subheader('BTC + ETH Quantitative Confirmation Signals')
    st.caption(
        'The QQQ + NQ model applied to perpetuals: BTC and ETH direction scores must agree, confirmed '
        'by stored daily alerts for BTC, ETH and BTCDOM. BTCDOM replaces VIX as the inverse gauge, '
        'because Binance lists no VIX.'
    )
    left, right = st.columns(2)
    timeframe = left.selectbox('BTC / ETH timeframe', ['5m', '15m', '1h', '4h'], index=2, key='dc_pair_tf')
    threshold = float(right.number_input(
        'Minimum direction score (each asset)', min_value=1.0, max_value=100.0, value=10.0, step=1.0,
        key='dc_pair_threshold',
    ))
    state = st.session_state
    if st.button('Generate BTC + ETH Signals', type='primary', key='dc_pair_generate'):
        histories, errors = {}, []
        with st.spinner('Loading BTC and ETH perpetual histories...'):
            for symbol in ('BTC/USDT', 'ETH/USDT'):
                frame = deps.fetch_history(symbol, timeframe=timeframe, silent=True)
                if frame is None or frame.empty:
                    errors.append(f'{symbol}: no candle history returned')
                else:
                    histories[symbol] = frame.copy()
        state['dc_pair_source'] = {
            'timeframe': timeframe, 'histories': histories, 'errors': errors,
            'loaded_at': pd.Timestamp.now(tz='UTC'),
        }
    with st.expander('Signal rules'):
        st.write(
            'Direction score = 100 × [0.60·tanh(ΔFlow Z) + 0.25·tanh(ΔMomentum Z) + 0.15·directional '
            'volatility]. BUY: both scores ≥ the threshold, plus positive BTC and ETH daily alerts and a '
            'negative BTCDOM daily alert. SELL: the reverse. CONFLICT: scores have opposite signs. '
            'WAIT: incomplete data or insufficient agreement.'
        )
        st.caption(
            'Daily alerts use only events already available that Ghana day; nothing carries overnight. '
            'Scores are model values, not probabilities.'
        )
    inputs = pair_alert_inputs(state)
    rows, issue = prepare_daily_alerts(**inputs)
    issue = relabel_text(issue)
    if issue:
        st.warning(issue + ' BUY/SELL signals remain disabled until this source is ready.')
    else:
        st.caption(
            f'Daily source: {len(rows)} stored BTC/ETH/BTCDOM events · 1h · '
            f'threshold ±{float(inputs["alert_threshold"]):g}. Refresh Derivative Overview updates this source.'
        )
    source = state.get('dc_pair_source')
    if not source:
        st.info('Choose a timeframe and click Generate BTC + ETH Signals.')
        return
    if source['timeframe'] != timeframe:
        st.info(f'Click Generate BTC + ETH Signals to load the {timeframe} timeframe.')
        return
    if source['errors']:
        st.error('Unable to compare both assets. ' + '; '.join(source['errors']))
        return
    signals = build_pair_signals(
        source['histories']['BTC/USDT'], source['histories']['ETH/USDT'],
        timeframe, threshold, source['loaded_at'], inputs,
    )
    if signals.empty:
        st.info('No completed candles are available for this comparison.')
        return
    dates = sorted(set(pd.to_datetime(signals['Signal Time (Ghana)'], utc=True).dt.date))
    selected = st.date_input(
        'BTC / ETH Ghana date', value=dates[-1], min_value=dates[0], max_value=dates[-1],
        key=f'dc_pair_date_{timeframe}_{dates[0]}_{dates[-1]}',
    )
    day = signals_on_ghana_day(signals, selected)
    if day.empty:
        st.info('No completed candles fall on this Ghana day. Choose another date.')
        return
    summary = st.columns(4)
    summary[0].metric('Comparable candles', int((day['BTC Ready'].eq(True) & day['ETH Ready'].eq(True)).sum()))
    summary[1].metric('BUY candles', int(day['Signal'].eq('BUY').sum()))
    summary[2].metric('SELL candles', int(day['Signal'].eq('SELL').sum()))
    summary[3].metric('Conflicts', int(day['Signal'].eq('CONFLICT').sum()))
    latest = day.iloc[-1]
    st.write(f'Last available state on this date: **{latest["Signal"]}** — {latest["Reason"]}.')
    blocked = day['Base Signal'].isin(['BUY', 'SELL']) & day['Signal'].eq('WAIT')
    st.caption(f'{int(blocked.sum())} base BUY/SELL candle(s) held at WAIT by the daily alert filter.')
    if day[['BTC Score', 'ETH Score']].notna().any().any():
        st.plotly_chart(pair_figure(day, threshold), width='stretch', key='dc_pair_chart')
    st.markdown('##### All BTC / ETH signals — selected Ghana day')
    columns = [
        'Signal Time (Ghana)', 'Signal', 'New Alert', 'Base Signal', 'BTC Score', 'ETH Score',
        'Confirmation Score', 'BTC Flow Event', 'ETH Flow Event', 'Daily Alert Confirmation',
        *[relabel_text(column) for column in EVIDENCE_COLUMNS],
        'BTC Price', 'ETH Price', 'Reason', 'Daily Alert Reason',
    ]
    table = day.reset_index()[columns]
    numeric = table.select_dtypes(include='number').columns
    st.dataframe(
        table.style.format({column: '{:+.2f}' for column in numeric}, na_rep='Unavailable'),
        width='stretch', hide_index=True,
    )
    st.download_button(
        'Download this day’s BTC / ETH signals', table.to_csv(index=False).encode('utf-8'),
        file_name=f'BTC_ETH_signals_{selected}_{timeframe}.csv', mime='text/csv', key='dc_pair_download',
    )


def render_pair_backtest(deps):
    st.markdown('#### Backtest BTC + ETH Confirmation Signals')
    state = st.session_state
    timeframe = state.get('dc_pair_tf', '1h')
    threshold = float(state.get('dc_pair_threshold', 10.0))
    today = pd.Timestamp.now(tz='UTC').date()
    earliest = today - timedelta(days=59)  # Stored daily alerts span about 60 days of 1h bars.
    c1, c2, c3, c4 = st.columns(4)
    start = c1.date_input('Backtest start (Ghana)', today - timedelta(days=30),
                          min_value=earliest, max_value=today, key='dc_bt_start')
    end = c2.date_input('Backtest end (Ghana, inclusive)', today,
                        min_value=earliest, max_value=today, key='dc_bt_end')
    traded = c3.selectbox('Traded perpetual', ['BTC/USDT', 'ETH/USDT'], key='dc_bt_traded')
    target_label = c4.selectbox('Take-profit rule', ['Nearest 1-hour liquidity zone', 'Fixed reward / risk'],
                                key='dc_bt_target')
    target_mode = 'liquidity' if target_label.startswith('Nearest') else 'fixed_rr'
    c5, c6, c7, c8 = st.columns(4)
    atr_multiple = c5.number_input('Stop: 5m ATR(14) multiple', .25, 10.0, 1.0, .25, key='dc_bt_atr')
    reward_risk = c6.number_input('Reward / risk', .25, 10.0, 2.0, .25,
                                  disabled=target_mode == 'liquidity', key='dc_bt_rr')
    slippage_bps = c7.number_input('Slippage per fill (bps)', 0.0, 100.0, 1.0, .5, key='dc_bt_slip')
    fee_bps = c8.number_input('Round-trip fees (bps)', 0.0, 100.0, 10.0, .5, key='dc_bt_fee')
    st.caption(
        f'Signal timeframe: {timeframe} · score threshold: ±{threshold:g}. Costs are editable assumptions; '
        'check them against your Binance fee tier. They are converted to USD at the median close of the '
        'test window. Funding payments are not included.'
    )
    inputs = pair_alert_inputs(state)
    _, issue = prepare_daily_alerts(**inputs)
    issue = relabel_text(issue)
    config = (timeframe, threshold, start, end, traded, target_mode, atr_multiple, reward_risk,
              slippage_bps, fee_bps, alert_source_fingerprint(inputs))
    with st.expander('Backtest execution rules'):
        st.write(
            'Only new BUY/SELL confirmations enter. BUY goes long the traded perpetual; SELL goes short. '
            'Entry is the 5-minute open at the exact signal time, with adverse slippage. One position '
            'stays open until stop, target or the end of the test. When a candle touches both stop and '
            'target, the stop counts first. Liquidity targets use only 1-hour candles completed by entry.'
        )
    if st.button('Run BTC + ETH Backtest', type='primary', key='dc_bt_run'):
        state.pop('dc_bt_result', None)
        if start > end:
            st.error('The start date must be on or before the end date.')
        elif issue:
            st.error(issue)
        else:
            warmup_start = str(start - timedelta(days=30))
            try:
                with st.spinner('Loading perpetual history and simulating trades...'):
                    sources = {}
                    requests = [('BTC/USDT', timeframe, warmup_start), ('ETH/USDT', timeframe, warmup_start),
                                (traded, '5m', str(start - timedelta(days=2)))]
                    if target_mode == 'liquidity':
                        requests.append((traded, '1h', warmup_start))
                    for symbol, bar, since in requests:
                        frame = deps.fetch_history(symbol, timeframe=bar, silent=True, start_date=since)
                        if frame is None or frame.empty:
                            raise ValueError(f'No {bar} candle history returned for {symbol}.')
                        sources[(symbol, bar)] = frame
                    execution = sources[(traded, '5m')]
                    slippage, fees = cost_in_price_terms(execution, start, slippage_bps, fee_bps)
                    result = backtest_qqq_nq(
                        sources[('BTC/USDT', timeframe)], sources[('ETH/USDT', timeframe)], execution,
                        sources.get((traded, '1h')), timeframe=timeframe, threshold=threshold,
                        start=start, end_exclusive=end + timedelta(days=1),
                        zone_calculator=deps.calculate_liquidity_zones, atr_multiple=atr_multiple,
                        target_mode=target_mode, reward_risk=reward_risk,
                        slippage_points=slippage, commission_points=fees, **inputs,
                    )
                state['dc_bt_result'] = {
                    'config': config, 'result': result, 'run_at': pd.Timestamp.now(tz='UTC'),
                }
            except Exception as exc:
                st.error('The backtest could not run: ' + relabel_text(exc))
    stored = state.get('dc_bt_result')
    if not stored:
        st.info('Choose the Ghana date range and click Run BTC + ETH Backtest.')
        return
    if stored['config'] != config:
        st.info('Backtest settings or stored daily alerts changed. Run again to update the results.')
        return
    result = stored['result']
    summary = result['summary']
    trades, audit = relabel_frame(result['trades']), relabel_frame(result['audit'])
    st.caption(
        f'Tested {start:%d %b %Y} through {end:%d %b %Y} Ghana · run {stored["run_at"]:%d %b %H:%M} Ghana. '
        f'Trading {traded}.'
    )
    metrics = st.columns(6)
    metrics[0].metric('Trades', summary['Trades'])
    metrics[1].metric('Win rate', f'{summary["Win Rate %"]:.1f}%' if len(trades) else '—')
    metrics[2].metric('Net R', f'{summary["Net R"]:+.2f}R')
    metrics[3].metric('Net USD per 1 unit', f'{summary["Net Points"]:+,.2f}')
    factor = summary['Profit Factor (points)']
    metrics[4].metric('Profit factor', '—' if pd.isna(factor) else ('∞' if np.isinf(factor) else f'{factor:.2f}'))
    metrics[5].metric('Closed drawdown', f'{summary["Max Closed Drawdown R"]:.2f}R')
    st.write(f'New confirmations: {summary["New confirmations"]} · skipped: {summary["Skipped"]}.')
    st.caption('This filter can leave very few trades. A small sample does not establish future profitability.')
    if trades.empty:
        st.info('No trades completed under these settings. The decision log explains any skipped confirmations.')
    else:
        figure = backtest_equity_figure(trades, start)
        figure.update_layout(title='BTC / ETH confirmation backtest — closed-trade equity')
        st.plotly_chart(figure, width='stretch', key='dc_bt_equity')
        st.dataframe(trades.style.format({
            name: '{:,.4f}' for name in trades.select_dtypes(include='number').columns if name != 'Trade'
        }, na_rep='—'), width='stretch', hide_index=True)
        st.download_button('Download backtest trades', trades.to_csv(index=False).encode(),
                           f'BTC_ETH_backtest_{start}_{end}_{timeframe}.csv', 'text/csv', key='dc_bt_export')
    with st.expander('Signal decision log — traded and skipped'):
        if audit.empty:
            st.write('No new BUY/SELL confirmations fell inside the selected Ghana dates.')
        else:
            st.dataframe(audit, width='stretch', hide_index=True)


def render_liquidity(deps):
    st.divider()
    st.subheader('Liquidity Zones')
    st.caption(
        'Ranks price bands by volume, absolute order-flow imbalance, ATR compression, swing density, '
        'time at price, and proximity to VWAP. The combined Liquidity Score is normalized from 0 to 100.'
    )
    c1, c2, c3 = st.columns(3)
    asset = c1.selectbox('Liquidity Zone asset', OVERVIEW_SYMBOLS, key='dc_liquidity_asset')
    timeframe = c2.selectbox('Liquidity Zone timeframe', ['5m', '15m', '1h', '4h', '1d'], index=2, key='dc_liquidity_tf')
    bins = int(c3.number_input('Number of price bands', min_value=10, max_value=60, value=24, step=2,
                               key='dc_liquidity_bins'))
    config = (asset, timeframe, bins)
    state = st.session_state
    if st.button(f'Find Liquidity Zones for {asset}', key='dc_liquidity_run'):
        with st.spinner(f'Mapping {asset} liquidity on {timeframe} candles...'):
            candles = deps.fetch_history(asset, timeframe=timeframe, silent=True, limit=750)
            zones, history = deps.calculate_liquidity_zones(candles, price_bins=bins, top_zones=8)
        state['dc_liquidity'] = (config, zones, history)
    saved = state.get('dc_liquidity')
    if not saved:
        return
    if saved[0] != config:
        st.info('The asset, timeframe, or band count changed. Run the analysis again to refresh the results.')
        return
    _, zones, history = saved
    if zones.empty or history.empty:
        st.info('Not enough price and volume history was available to map liquidity zones.')
        return
    strongest, last_price = zones.iloc[0], float(history['close'].iloc[-1])
    metrics = st.columns(4)
    metrics[0].metric('Strongest Zone',
                      f"{format_price(strongest['Zone Bottom'])} - {format_price(strongest['Zone Top'])}")
    metrics[1].metric('Liquidity Score', f"{strongest['Liquidity Score']:.1f}/100")
    metrics[2].metric('Flow Bias', strongest['Flow Bias'])
    metrics[3].metric('Last Price', format_price(last_price))
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
    st.plotly_chart(figure, width='stretch', key='dc_liquidity_chart')
    columns = ['Zone Bottom', 'Zone Top', 'Liquidity Score', 'Location vs Price', 'Flow Bias',
               'Order Flow Imbalance', 'Volume Z-Score', 'ATR Compression Z', 'Swing Density Z',
               'Time at Price Z', 'VWAP Proximity Z']
    st.dataframe(
        zones[columns].style.format({
            'Zone Bottom': format_price, 'Zone Top': format_price, 'Liquidity Score': '{:.1f}',
            'Order Flow Imbalance': '{:+.2%}', 'Volume Z-Score': '{:+.2f}', 'ATR Compression Z': '{:+.2f}',
            'Swing Density Z': '{:+.2f}', 'Time at Price Z': '{:+.2f}', 'VWAP Proximity Z': '{:+.2f}',
        }).background_gradient(subset=['Liquidity Score'], cmap='viridis'),
        width='stretch', hide_index=True,
    )


def render_volume_delta(deps):
    st.divider()
    st.subheader('Volume Delta: BTC, ETH, BNB, SOL, XRP, DOGE, BTCDOM')
    st.caption(
        'Volume Delta = USDT volume on up candles minus USDT volume on down candles. '
        'Positive values indicate net buying pressure; negative values indicate net selling pressure.'
    )
    timeframe = st.selectbox('Volume Delta timeframe', ['5m', '15m', '1h', '4h'], index=2, key='dc_delta_tf')
    state = st.session_state
    if st.button('Refresh Crypto Volume Delta', key='dc_delta_refresh'):
        rows, history = [], {}
        with st.spinner('Calculating volume delta...'):
            for symbol in OVERVIEW_SYMBOLS:
                candles = deps.fetch_history(symbol, timeframe=timeframe, silent=True, limit=500)
                if candles is None or candles.empty:
                    continue
                frame = with_volume_delta(candles)
                history[symbol] = frame
                rows.append({
                    'Asset': symbol,
                    'Latest Volume Delta': float(frame['Volume Delta'].iloc[-1]),
                    '20-Candle Volume Delta': float(frame['20-Candle Volume Delta'].iloc[-1]),
                    'Cumulative Volume Delta': float(frame['Cumulative Volume Delta'].iloc[-1]),
                })
        state['dc_delta'] = (timeframe, pd.DataFrame(rows), history)
    saved = state.get('dc_delta')
    if not saved or saved[0] != timeframe:
        return
    _, summary, history = saved
    if summary.empty:
        st.info('No volume-delta data was returned. Try another timeframe.')
        return
    fields = ['Latest Volume Delta', '20-Candle Volume Delta', 'Cumulative Volume Delta']
    st.dataframe(summary.style.format({field: deps.format_large_number for field in fields}),
                 width='stretch', hide_index=True)
    asset = st.selectbox('Select asset for detailed Volume Delta', summary['Asset'].tolist(), key='dc_delta_asset')
    selected = history[asset]
    metrics = st.columns(3)
    for column, field, source in zip(metrics, fields, ['Volume Delta', '20-Candle Volume Delta', 'Cumulative Volume Delta']):
        column.metric(field, deps.format_large_number(float(selected[source].iloc[-1])))
    st.line_chart(selected['Cumulative Volume Delta'])
    detail = ['Buy Volume', 'Sell Volume', 'Volume Delta', 'Cumulative Volume Delta']
    st.dataframe(selected[detail].tail(50).style.format({name: deps.format_large_number for name in detail}),
                 width='stretch', height=300)


def render_support(deps):
    st.divider()
    st.subheader('Key Support Levels')
    st.caption('Technical levels that may act as support: the 50-candle swing low, lower Bollinger Band and 200-period MA.')
    timeframe = st.session_state.get('dc_overview_tf', '1h')
    state = st.session_state
    if st.button('Refresh Key Support Levels', key='dc_support_refresh'):
        rows = []
        with st.spinner('Fetching key support levels...'):
            for symbol in QS_SYMBOLS:
                candles = deps.fetch_history(symbol, timeframe=timeframe, silent=True)
                if candles is not None and not candles.empty:
                    current = candles.iloc[-1]
                    rows.append({
                        'Symbol': symbol, 'Current Price': current['close'],
                        'Bullish Order Block': current.get('ob_bull', np.nan),
                        'Lower Bollinger Band': current.get('bb_lower', np.nan),
                        '200-Period MA': current.get('sma_200', np.nan),
                    })
        state['dc_support'] = pd.DataFrame(rows)
    table = state.get('dc_support')
    if table is None or table.empty:
        st.info("Click 'Refresh Key Support Levels' to load the support-level basket.")
        return
    prices = ['Current Price', 'Bullish Order Block', 'Lower Bollinger Band', '200-Period MA']
    st.dataframe(table.style.format({name: format_price for name in prices}), width='stretch', hide_index=True)


def render_top_perpetuals(deps):
    st.divider()
    st.subheader('🏢 Top 10 Perpetuals Overview')
    st.write('The 10 crypto perpetuals with the highest 24-hour USDT volume on Binance.')
    timeframe = st.session_state.get('dc_overview_tf', '1h')
    state = st.session_state
    if st.button('Refresh Top Perpetuals', key='dc_top_refresh'):
        rows = []
        with st.spinner('Fetching and analyzing the top perpetuals...'):
            try:
                symbols = top_usdm_symbols(10)
            except Exception as exc:
                symbols = []
                st.error(f'Could not rank Binance perpetuals: {exc}')
            for symbol in symbols:
                candles = deps.fetch_history(symbol, timeframe=timeframe, silent=True)
                if candles is not None and not candles.empty:
                    rows.append(qs_factor_row(symbol, candles))
        if rows:
            state['dc_top'] = add_qs_scores(pd.DataFrame(rows))
        else:
            state.pop('dc_top', None)
            st.warning('No perpetual data was returned.')
    top = state.get('dc_top')
    if top is None:
        st.info("Click 'Refresh Top Perpetuals' to load the latest metrics.")
        return
    bullish, positive, total = int(top['trend_bullish'].sum()), int((top['momentum'] > 0).sum()), len(top)
    health = bullish / total * 100
    status = '🟢 STRONG BULL' if health >= 70 else '🔴 BEARISH' if health <= 30 else '🟡 NEUTRAL'
    metrics = st.columns(3)
    metrics[0].metric('Perpetuals in Bullish Trend', f'{bullish} / {total}')
    metrics[1].metric('Positive Momentum', f'{positive} / {total}')
    metrics[2].metric('Overall Health', status)
    columns = ['symbol', 'qs_score', 'price', 'momentum', 'rsi']
    st.dataframe(
        top[columns].style.format({
            'price': format_price, 'qs_score': '{:.2f}', 'momentum': '{:.2%}', 'rsi': '{:.1f}',
        }).map(deps.color_metrics, subset=['momentum']),
        width='stretch', hide_index=True,
    )


def render_market_sections(deps):
    """Sections that follow the US tab's overview through to its pair signals."""
    render_overview(deps)
    render_gex(deps)
    render_order_flow(deps)
    render_daily_flow(deps)
    render_qs(deps)
    render_pair_signals(deps)
    render_pair_backtest(deps)


def render_structure_sections(deps):
    """Sections that follow the US tab's Zone Battle."""
    render_liquidity(deps)
    render_volume_delta(deps)
    render_support(deps)
    render_top_perpetuals(deps)

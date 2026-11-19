"""Quant verdicts for the Ghana-day Demand & Supply Rise/Drop alerts.

Every alert gets a BUY / SELL verdict only when:
  1. its Direction Score is strong enough (|score| >= min_score),
  2. the Quant Score Q agrees with it (when confirmation is on), and
  3. the same rule had a positive, cost-adjusted edge on this asset in the history
     BEFORE the selected Ghana day (so the day itself is never used to judge itself).
For past days the table also shows what actually happened after each alert.
"""

import math

import numpy as np
import pandas as pd
import streamlit as st

import quant_edge as qe
from qqq_nq_signals import demand_supply_score, historical_z_components


MIN_TRAIN_BARS = 300
VERDICT_COLORS = {
    'BUY': 'color: #22c55e; font-weight: 600', 'SELL': 'color: #ef4444; font-weight: 600',
    'NO EDGE': 'color: #f59e0b', 'CONFLICT': 'color: #f59e0b',
}
ALERT_FORMULAS = [
    ('Direction Score (existing alert formula)',
     r'S_t = 100\left[0.60\tanh\Delta F_t + 0.25\tanh\Delta M_t + 0.15\,\operatorname{sign}(\Delta P_t)'
     r'\tanh|\Delta V_t|\cdot\mathbb{1}_{\Delta V_t > 0}\right]'),
    ('Alert direction', r'd_t = \operatorname{sign}(S_t)\ \text{ if } |S_t| \ge S_{min},\ \text{else } 0'),
    ('Quant confirmation (Q from the Quant Edge Layer, regime weight a from pre-day Hurst)',
     r'\text{confirmed}_t \iff \operatorname{sign}(Q_t) = d_t \ \wedge\ |Q_t| \ge Q_{min}'),
    ('Edge gate (backtest on bars before the selected day only)',
     r'n \ge 10,\quad \bar R > 0,\quad f^* = p - \frac{1-p}{b} > 0'),
    ('Information coefficient of the Direction Score', r'IC = \rho_s(S_t,\ r_{t+1})'),
    ('Position size', r'\text{risk} = \min\left(\rho,\ \tfrac{f^*}{2}\right),\quad '
     r'\text{units} = \min\left(\frac{E\cdot\text{risk}}{m\cdot ATR},\ \frac{3E}{C}\right)'),
]


def scored_history(history, flow='proxy'):
    """UTC-indexed history with the alert Direction Score for every bar (NaN when Flow Z is flat)."""
    data = history.copy()
    data.index = pd.to_datetime(data.index, utc=True)
    data = data[~data.index.duplicated(keep='last')].sort_index()
    if 'cmf' not in data:
        data['cmf'] = 0.0
    changes = historical_z_components(data, flow).diff()
    price_change = data['close'].diff()
    price_change.iloc[0] = data['close'].iloc[0] - data['open'].iloc[0]
    score = pd.Series(demand_supply_score(
        changes['Flow (z)'].to_numpy(), changes['Momentum (z)'].to_numpy(),
        changes['Volatility (z)'].to_numpy(), price_change.to_numpy(),
    ), index=data.index)
    score[changes['Flow (z)'].eq(0)] = np.nan
    return data, score


def alert_direction(score, q, min_score, min_q, confirm):
    """+1 / -1 where the rule would trade, NaN elsewhere."""
    direction = np.sign(score).where(score.abs() >= min_score)
    if confirm:
        direction = direction.where((np.sign(q) == direction) & (q.abs() >= min_q))
    return direction


def trade_outcome(candles, position, side, atr_multiple, reward_risk, max_hold, fee_bps):
    """Result in R of one trade entered at bar `position`'s close, and how it ended."""
    entry = float(candles['close'].iloc[position])
    risk = atr_multiple * float(candles['atr14'].iloc[position])
    if not risk > 0:
        return np.nan, 'No ATR'
    stop, target = entry - side * risk, entry + side * reward_risk * risk
    last = min(position + max_hold, len(candles) - 1)
    fee = fee_bps / 10_000 * entry / risk
    for j in range(position + 1, last + 1):
        high, low = float(candles['high'].iloc[j]), float(candles['low'].iloc[j])
        if (low <= stop) if side > 0 else (high >= stop):
            return -1.0 - fee, 'Stop hit'
        if (high >= target) if side > 0 else (low <= target):
            return reward_risk - fee, 'Target hit'
    if last == position:
        return np.nan, 'Still open'
    marked = side * (float(candles['close'].iloc[last]) - entry) / risk - fee
    return marked, ('Time exit' if position + max_hold <= len(candles) - 1 else 'Still open')


def evaluate_alerts(history, alerts, selected_day, settings, flow='proxy'):
    """Verdict table for the day's alerts plus the pre-day statistics behind it."""
    data, score = scored_history(history, flow)
    candles = qe.clean_wicks(data)
    on_day = np.asarray(data.index.tz_convert('Africa/Accra').date) == selected_day
    if not on_day.any():
        return pd.DataFrame(), {'error': 'No candles fall on the selected Ghana day.'}
    day_start = int(np.argmax(on_day))
    if day_start < MIN_TRAIN_BARS:
        return pd.DataFrame(), {'error': f'Only {day_start} candles exist before this day; '
                                         f'at least {MIN_TRAIN_BARS} are needed to judge the alerts.'}

    train = candles.iloc[:day_start]
    hurst = qe.hurst_exponent(train['close'])
    q = qe.signal_series(candles, qe.regime_weight(hurst))
    args = (0.5, settings['atr_multiple'], settings['reward_risk'], settings['max_hold'], settings['fee_bps'])
    raw = alert_direction(score, q, settings['min_score'], settings['min_q'], confirm=False)
    confirmed = alert_direction(score, q, settings['min_score'], settings['min_q'], confirm=True)
    stats_raw = qe.trade_statistics(qe.barrier_backtest(train, raw.iloc[:day_start], *args))
    stats_confirmed = qe.trade_statistics(qe.barrier_backtest(train, confirmed.iloc[:day_start], *args))
    stats = stats_confirmed if settings['confirm'] else stats_raw
    ic, ic_t = qe.information_coefficient(score.iloc[:day_start], train['close'])

    reasons = []
    if stats['Trades'] < qe.MIN_TRADES:
        reasons.append(f"only {stats['Trades']} past trades")
    if not stats['Expectancy R'] > 0:
        reasons.append('past expectancy ≤ 0 after fees')
    if not stats['Kelly f*'] > 0:
        reasons.append('Kelly ≤ 0')
    risk_fraction = 0.0 if reasons else min(settings['risk_pct'] / 100, stats['Kelly f*'] / 2)

    rows = []
    times = pd.to_datetime(alerts['Ghana Time'], utc=True)
    for alert, moment in zip(alerts.to_dict('records'), times):
        position = data.index.get_indexer([moment])[0]
        if position < 0:
            continue
        s, qv = float(score.iloc[position]), float(q.iloc[position])
        side = int(np.sign(s)) if np.isfinite(s) and abs(s) >= settings['min_score'] else 0
        if side == 0:
            verdict, reason = 'WEAK', f"|Direction Score| below {settings['min_score']:g}"
        elif settings['confirm'] and np.isfinite(qv) and np.sign(qv) == -side and abs(qv) >= settings['min_q']:
            verdict, reason = 'CONFLICT', f'Quant Score Q {qv:+.2f} points the other way'
        elif settings['confirm'] and not (np.isfinite(qv) and np.sign(qv) == side and abs(qv) >= settings['min_q']):
            verdict, reason = 'UNCONFIRMED', f"Quant Score Q {qv:+.2f} too weak (needs {settings['min_q']:g})"
        elif reasons:
            verdict, reason = 'NO EDGE', '; '.join(reasons)
        else:
            verdict = 'BUY' if side > 0 else 'SELL'
            reason = (f"past E[R] {stats['Expectancy R']:+.2f} over {stats['Trades']} trades, "
                      f"win {stats['Win Rate']:.0%}")
        entry = float(candles['close'].iloc[position])
        risk_distance = settings['atr_multiple'] * float(candles['atr14'].iloc[position])
        plan = side != 0 and risk_distance > 0
        units = 0.0
        if verdict in ('BUY', 'SELL') and risk_distance > 0:
            units = min(settings['equity'] * risk_fraction / risk_distance,
                        qe.MAX_LEVERAGE * settings['equity'] / entry)
        outcome_r, outcome = (trade_outcome(candles, position, side, settings['atr_multiple'],
                                            settings['reward_risk'], settings['max_hold'], settings['fee_bps'])
                              if plan else (np.nan, '—'))
        rows.append({
            'Ghana Time': moment.tz_convert('Africa/Accra').strftime('%Y-%m-%d %H:%M'),
            'Alert': alert.get('Alert'), 'Direction Score': s, 'Quant Score Q': qv,
            'Verdict': verdict, 'Reason': reason, 'Entry': entry,
            'Stop': entry - side * risk_distance if plan else np.nan,
            'Target': entry + side * settings['reward_risk'] * risk_distance if plan else np.nan,
            'Size (units)': units, '$ Risk': units * risk_distance,
            'Outcome': outcome, 'Outcome R': outcome_r,
        })
    summary = {
        'hurst': hurst, 'regime': qe.classify_regime(hurst, 0.0), 'ic': ic, 'ic_t': ic_t,
        'raw': stats_raw, 'confirmed': stats_confirmed, 'gate': not reasons, 'train_bars': day_start,
    }
    return pd.DataFrame(rows), summary


def render_verdict_settings(prefix, fee_default):
    with st.expander('Quant Verdict Settings'):
        c1, c2, c3, c4 = st.columns(4)
        settings = {
            'min_score': float(c1.number_input('Min |Direction Score|', 5.0, 100.0, 30.0, 5.0, key=f'{prefix}_score')),
            'confirm': c2.checkbox('Require Quant Score Q to agree', value=True, key=f'{prefix}_confirm'),
            'min_q': float(c3.number_input('Min |Q| to confirm', 0.0, 0.9, 0.10, 0.05, key=f'{prefix}_minq')),
            'max_hold': int(c4.number_input('Max hold (bars)', 4, 500, 48, 4, key=f'{prefix}_hold')),
        }
        c5, c6, c7, c8, c9 = st.columns(5)
        settings['atr_multiple'] = float(c5.number_input('Stop: ATR multiple', 0.5, 5.0, 1.5, 0.25, key=f'{prefix}_atr'))
        settings['reward_risk'] = float(c6.number_input('Reward / risk', 0.5, 5.0, 2.0, 0.25, key=f'{prefix}_rr'))
        settings['fee_bps'] = float(c7.number_input('Round-trip costs (bps)', 0.0, 500.0, fee_default, 0.5,
                                                    key=f'{prefix}_fee'))
        settings['equity'] = float(c8.number_input('Account equity', 10.0, 1e9, 1000.0, 100.0, key=f'{prefix}_equity'))
        settings['risk_pct'] = float(c9.number_input('Max risk per trade (%)', 0.1, 5.0, 1.0, 0.1, key=f'{prefix}_risk'))
    return settings


def stats_frame(summary):
    rows = []
    for label, key in [('Raw alerts (|score| only)', 'raw'), ('Quant-confirmed alerts', 'confirmed')]:
        stats = summary[key]
        rows.append({'Rule': label, 'Past Trades': stats['Trades'], 'Win Rate': stats['Win Rate'],
                     'Expectancy R': stats['Expectancy R'], 't-stat': stats['Expectancy t'],
                     'Profit Factor': stats['Profit Factor'], 'Kelly f*': stats['Kelly f*']})
    return pd.DataFrame(rows)


def render_alert_quant_verdict(history, alerts, selected_day, key_prefix, price_format='{:,.4f}', fee_default=10.0,
                               flow='proxy'):
    """Quant verdicts under a Ghana-day Rise/Drop alert table. Safe to call with empty inputs."""
    st.markdown('##### 🧮 Quant Verdict for these alerts')
    if history is None or alerts is None or history.empty or alerts.empty:
        st.caption('No alerts to judge on this day.')
        return
    settings = render_verdict_settings(key_prefix, fee_default)
    try:
        table, summary = evaluate_alerts(history, alerts, selected_day, settings, flow)
    except Exception as exc:  # A data problem must not break the alert table above.
        st.warning(f'Quant verdict unavailable: {exc}')
        return
    if 'error' in summary:
        st.info(summary['error'])
        return

    verdicts = table['Verdict'].value_counts() if not table.empty else pd.Series(dtype=int)
    taken = table[table['Verdict'].isin(['BUY', 'SELL'])] if not table.empty else table
    resolved = taken[taken['Outcome'].isin(['Target hit', 'Stop hit', 'Time exit'])] if not taken.empty else taken
    metrics = st.columns(5)
    metrics[0].metric('BUY / SELL verdicts', f"{int(verdicts.get('BUY', 0))} / {int(verdicts.get('SELL', 0))}")
    metrics[1].metric('Edge gate', 'PASS' if summary['gate'] else 'FAIL')
    metrics[2].metric('Regime (pre-day Hurst)', f"{summary['regime']} ({summary['hurst']:.2f})"
                      if np.isfinite(summary['hurst']) else 'Unknown')
    metrics[3].metric('Score IC (t)', f"{summary['ic']:+.3f} ({summary['ic_t']:+.1f})"
                      if np.isfinite(summary['ic']) else 'n/a')
    metrics[4].metric('This day\'s verdict result',
                      f"{resolved['Outcome R'].sum():+.2f}R ({len(resolved)} closed)" if len(resolved) else '—')
    st.caption(
        f"Judged only on the {summary['train_bars']} candles before this Ghana day. Outcome columns show what "
        'happened after each alert (stop counts first if one candle hit both). Alerts on the same day overlap, '
        'so taking all of them stacks risk; the first BUY/SELL of a direction is usually enough.'
    )
    st.dataframe(
        table.style.format({
            'Direction Score': '{:+.1f}', 'Quant Score Q': '{:+.2f}', 'Entry': price_format, 'Stop': price_format,
            'Target': price_format, 'Size (units)': '{:,.4f}', '$ Risk': '${:,.2f}', 'Outcome R': '{:+.2f}',
        }, na_rep='—').map(lambda v: VERDICT_COLORS.get(v, 'color: gray'), subset=['Verdict']),
        width='stretch', hide_index=True,
    )
    st.markdown('###### Does the quant confirmation help on this asset? (history before this day)')
    st.dataframe(
        stats_frame(summary).style.format({
            'Win Rate': '{:.1%}', 'Expectancy R': '{:+.3f}', 't-stat': '{:+.2f}', 'Profit Factor': '{:.2f}',
            'Kelly f*': '{:+.3f}',
        }, na_rep='—'),
        width='stretch', hide_index=True,
    )
    with st.expander('Verdict formulas'):
        for label, formula in ALERT_FORMULAS:
            st.markdown(f'**{label}**')
            st.latex(formula)
    st.caption('A past edge is evidence, not a guarantee. Forward-test with small size first.')

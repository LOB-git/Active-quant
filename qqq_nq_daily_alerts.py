"""Causal confirmation from the US Indices stored daily alert table.

Rows are first-per-direction Ghana-day events, NOT continuous Z-score readings.
Their Ghana Time already represents candle close; never shift it a second time.
"""

import hashlib
import json

import numpy as np
import pandas as pd


ASSETS = {'QQQ': 'QQQ', 'NQ=F': 'NQ', '^VIX': 'VIX'}
REQUIRED = ['Ghana Time', 'Symbol', 'Alert', 'Z-Score', 'Momentum Z', 'Timeframe']
EVIDENCE_COLUMNS = [
    f'{label} {field}' for label in ASSETS.values()
    for field in ('Daily Z', 'Daily Momentum Z', 'Alert Time (Ghana)', 'Daily Alert')
]


def stored_alert_inputs(state):
    """Read the actual overview table, without recalculating or changing it."""
    return {
        'daily_alerts': state.get('daily_ghana_zscore_alerts', []),
        'alert_timeframe': state.get('daily_ghana_zscore_alert_timeframe'),
        'alert_threshold': state.get('daily_ghana_zscore_alert_threshold', 1.0),
    }


def alert_source_fingerprint(inputs):
    """Invalidate backtests if the source rows or their definitions change."""
    rows = pd.DataFrame(inputs['daily_alerts'])
    if 'Symbol' in rows:
        rows = rows.loc[rows['Symbol'].isin(ASSETS)]
    serialized = json.dumps({
        'rows': sorted(json.dumps(row, sort_keys=True, default=str) for row in rows.to_dict('records')),
        'timeframe': inputs['alert_timeframe'], 'threshold': inputs['alert_threshold'],
        'rule_version': 1,
    }, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()


def prepare_daily_alerts(daily_alerts, alert_timeframe='1h', alert_threshold=1.0):
    """Validate source semantics; invalid input fails closed with a visible reason."""
    rows = pd.DataFrame(daily_alerts).copy()
    empty = pd.DataFrame(columns=REQUIRED)
    if rows.empty:
        return empty, 'No stored daily alerts. Refresh Indices Data with the overview timeframe set to 1h.'
    if alert_timeframe != '1h':
        return empty, 'Stored daily alerts must use 1h. Set the overview timeframe to 1h and Refresh Indices Data.'
    if not set(REQUIRED).issubset(rows.columns):
        return empty, 'Stored daily alerts are missing required fields. Refresh Indices Data to rebuild the table.'
    try:
        threshold = float(alert_threshold)
    except (ValueError, TypeError):
        threshold = np.nan
    if not np.isfinite(threshold) or threshold <= 0:
        return empty, 'The stored daily Z-score threshold is invalid. Refresh Indices Data.'
    rows = rows.loc[rows['Symbol'].isin(ASSETS)].copy()
    rows['Ghana Time'] = pd.to_datetime(rows['Ghana Time'], utc=True, errors='coerce', format='mixed')
    for column in ('Z-Score', 'Momentum Z'):
        rows[column] = pd.to_numeric(rows[column], errors='coerce')
    positive = rows['Z-Score'].ge(threshold) & rows['Momentum Z'].ge(threshold)
    negative = rows['Z-Score'].le(-threshold) & rows['Momentum Z'].le(-threshold)
    valid = (
        rows['Ghana Time'].notna() & rows['Timeframe'].eq('1h')
        & np.isfinite(rows['Z-Score']) & np.isfinite(rows['Momentum Z'])
        & ((positive & rows['Alert'].eq('POSITIVE Z-SCORE'))
           | (negative & rows['Alert'].eq('NEGATIVE Z-SCORE')))
    )
    if 'Ghana Date' in rows:
        source_dates = pd.to_datetime(rows['Ghana Date'], errors='coerce', format='mixed').dt.date
        valid &= source_dates.eq(rows['Ghana Time'].dt.date)
    if not valid.all():
        return empty, f'{int((~valid).sum())} stored QQQ/NQ/VIX alert row(s) have invalid time, timeframe or aligned Z values. Refresh Indices Data.'
    rows = rows.drop_duplicates(subset=REQUIRED)
    if rows.duplicated(['Symbol', 'Ghana Time']).any():
        return empty, 'Conflicting stored alerts share a symbol and timestamp. Refresh Indices Data.'
    return rows.sort_values('Ghana Time').reset_index(drop=True), ''


def confirm_with_daily_alerts(signals, daily_alerts, timeframe, *,
                              alert_timeframe='1h', alert_threshold=1.0):
    """Gate base scores using each asset's latest already-known same-day event.

    BUY: base BUY + positive QQQ/NQ + negative VIX. SELL: the reverse.
    No next-day carry, future alerts, re-created crossings, or missing-data pass.
    Recompute New Alert AFTER gating so a newly available alert can enable entry.
    """
    result = signals.copy()
    result['Base Signal'] = result['Signal']
    rows, issue = prepare_daily_alerts(daily_alerts, alert_timeframe, alert_threshold)
    times = pd.to_datetime(result['Signal Time (Ghana)'], utc=True)
    left = pd.DataFrame({'Signal Time': times.to_numpy(), '_row': np.arange(len(result))})
    left['Signal Time'] = pd.to_datetime(left['Signal Time'], utc=True)
    left['_day'] = left['Signal Time'].dt.normalize()
    for symbol, label in ASSETS.items():
        events = rows.loc[rows['Symbol'].eq(symbol)].copy()
        if events.empty or left.empty:
            result[f'{label} Daily Z'] = np.nan
            result[f'{label} Daily Momentum Z'] = np.nan
            result[f'{label} Alert Time (Ghana)'] = pd.Series(pd.NaT, index=result.index, dtype='datetime64[ns, UTC]')
            result[f'{label} Daily Alert'] = None
            continue
        events['_day'] = events['Ghana Time'].dt.normalize()
        matched = pd.merge_asof(
            left.sort_values('Signal Time'), events.sort_values('Ghana Time'),
            left_on='Signal Time', right_on='Ghana Time', by='_day', direction='backward',
            allow_exact_matches=True,
        ).sort_values('_row')
        for field, source in [('Daily Z', 'Z-Score'), ('Daily Momentum Z', 'Momentum Z'),
                              ('Alert Time (Ghana)', 'Ghana Time'), ('Daily Alert', 'Alert')]:
            result[f'{label} {field}'] = matched[source].array

    q, n, v = (result[f'{label} Daily Z'] for label in ('QQQ', 'NQ', 'VIX'))
    ready = q.notna() & n.notna() & v.notna()
    buy = ready & q.gt(0) & n.gt(0) & v.lt(0)
    sell = ready & q.lt(0) & n.lt(0) & v.gt(0)
    result['Daily Alert Confirmation'] = np.select(
        [buy, sell, ready], ['BUY', 'SELL', 'CONFLICT'], default='WAIT'
    )
    missing = pd.Series('', index=result.index)
    for label in ASSETS.values():
        missing += np.where(result[f'{label} Daily Z'].isna(), label + ' ', '')
    result['Daily Alert Reason'] = np.select(
        [buy, sell, ready],
        ['Positive QQQ/NQ and negative VIX daily alerts',
         'Negative QQQ/NQ and positive VIX daily alerts',
         'Stored daily alert directions do not confirm (VIX must be opposite QQQ/NQ)'],
        default='No same-day alert available yet: ' + missing.str.strip(),
    )
    if issue:
        result['Daily Alert Reason'] = issue
    candidate = result['Base Signal'].isin(['BUY', 'SELL'])
    confirmed = candidate & result['Base Signal'].eq(result['Daily Alert Confirmation'])
    blocked = candidate & ~confirmed
    result.loc[blocked, 'Signal'] = 'WAIT'
    result.loc[blocked, 'Reason'] = 'Base ' + result.loc[blocked, 'Base Signal'] + ' blocked: ' + result.loc[blocked, 'Daily Alert Reason']
    opposite = blocked & result['Daily Alert Confirmation'].isin(['BUY', 'SELL'])
    result.loc[opposite, 'Reason'] = 'Base direction opposes the stored daily alert confirmation'
    result.loc[confirmed, 'Reason'] = 'Scores confirmed: ' + result.loc[confirmed, 'Daily Alert Reason']
    durations = {'5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h'}
    contiguous = times.diff().eq(pd.Timedelta(durations[timeframe]))
    same_day = times.dt.normalize().eq(times.shift().dt.normalize())
    result['New Alert'] = confirmed & (
        result['Signal'].ne(result['Signal'].shift()) | ~contiguous | ~same_day
    )
    return result

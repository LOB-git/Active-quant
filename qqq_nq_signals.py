"""Shared historical Flow-Z factors and causal QQQ/NQ confirmation signals."""

import numpy as np
import pandas as pd

from qqq_nq_daily_alerts import confirm_with_daily_alerts


TIMEFRAMES = {'5m': '5min', '15m': '15min', '1h': '1h', '4h': '4h'}


def historical_z_components(history):
    """Use the same 20-bar flow / 50-bar Z formulas as the US historical chart."""
    is_up = history['close'] >= history['open']
    volume = pd.to_numeric(history['volume'], errors='coerce')
    inflow = volume.where(is_up, 0).rolling(20).sum()
    outflow = volume.where(~is_up, 0).rolling(20).sum()
    flow = (inflow - outflow) / (inflow + outflow).replace(0, np.nan)

    def z(series):
        values = pd.to_numeric(series, errors='coerce')
        return (values - values.rolling(50).mean()) / values.rolling(50).std().replace(0, np.nan)

    return pd.DataFrame({
        'Momentum (z)': z(history['momentum']),
        'Flow (z)': z(flow),
        'Volatility (z)': z(history['atr14']),
        'Trend (z)': z(history['z_score']),
    }).replace([np.inf, -np.inf], np.nan)


def demand_supply_score(flow_change, momentum_change, volatility_change, price_change):
    """Signed -100..100 score, retaining unavailable factors as NaN."""
    volatility_direction = np.where(
        np.asarray(volatility_change) > 0,
        np.sign(price_change) * np.tanh(np.abs(volatility_change)),
        0.0,
    )
    score = 100 * (
        0.60 * np.tanh(flow_change)
        + 0.25 * np.tanh(momentum_change)
        + 0.15 * volatility_direction
    )
    valid = (
        np.isfinite(flow_change) & np.isfinite(momentum_change)
        & np.isfinite(volatility_change) & np.isfinite(price_change)
    )
    return np.where(valid, score, np.nan)


def completed_direction_history(history, timeframe, as_of):
    """Keep the provider's bar grid and only use bars past their nominal close.

    fetch_and_analyze returns UTC-naive index values. Aware inputs are converted
    to UTC. No time rounding, filling, or matching across different bar grids.
    """
    columns = ['Score', 'Flow Z Change', 'Momentum Z Change', 'Volatility Z Change',
               'Price', 'Flow Event', 'Ready']
    if history is None or history.empty:
        return pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz='UTC'))
    data = history.copy()
    data.index = pd.to_datetime(data.index, utc=True)
    data = data[~data.index.duplicated(keep='last')].sort_index()
    duration = pd.Timedelta(TIMEFRAMES[timeframe])
    cutoff = pd.Timestamp(as_of)
    cutoff = cutoff.tz_localize('UTC') if cutoff.tzinfo is None else cutoff.tz_convert('UTC')
    data = data.loc[(data.index + duration) <= cutoff]
    components = historical_z_components(data)
    changes = components.diff()
    flow = changes['Flow (z)']
    result = pd.DataFrame(index=data.index)
    result['Score'] = demand_supply_score(
        flow, changes['Momentum (z)'], changes['Volatility (z)'], data['close'].diff()
    )
    result['Flow Z Change'] = flow
    result['Momentum Z Change'] = changes['Momentum (z)']
    result['Volatility Z Change'] = changes['Volatility (z)']
    result['Price'] = data['close']
    result['Flow Event'] = np.select(
        [flow.gt(0), flow.lt(0), flow.eq(0)], ['Rise', 'Drop', 'Unchanged'], default='Unavailable'
    )
    result['Ready'] = np.isfinite(result['Score']) & flow.notna() & flow.ne(0)
    return result


def combine_direction_histories(qqq, nq, timeframe, threshold=10.0):
    """Exact-match confirmation; minimum signed strength prevents cancellation.

    BUY if both scores >= threshold, SELL if both <= -threshold. Conflicting
    directions, unresolved factors and unmatched bars cannot produce trades.
    New Alert marks entry into a state; no state carries across missing bars.
    """
    if not 0 < threshold <= 100:
        raise ValueError('The confirmation threshold must be greater than 0 and at most 100.')
    result = qqq.add_prefix('QQQ ').join(nq.add_prefix('NQ '), how='outer').sort_index()
    result.index.name = 'Candle Time (Ghana)'
    duration = pd.Timedelta(TIMEFRAMES[timeframe])
    result['Signal Time (Ghana)'] = result.index + duration
    both = result['QQQ Ready'].eq(True) & result['NQ Ready'].eq(True)
    qscore = pd.to_numeric(result['QQQ Score'], errors='coerce')
    nscore = pd.to_numeric(result['NQ Score'], errors='coerce')
    aligned = both & (qscore * nscore > 0)
    result['Confirmation Score'] = np.where(
        aligned, np.sign(qscore) * np.minimum(qscore.abs(), nscore.abs()),
        np.where(both, 0.0, np.nan),
    )
    buy = both & qscore.ge(threshold) & nscore.ge(threshold)
    sell = both & qscore.le(-threshold) & nscore.le(-threshold)
    conflict = both & (qscore * nscore < 0)
    result['Signal'] = np.select(
        [buy, sell, conflict], ['BUY', 'SELL', 'CONFLICT'], default='WAIT'
    )
    result['Reason'] = np.select(
        [~result.index.isin(qqq.index), ~result.index.isin(nq.index), ~both,
         buy, sell, conflict],
        ['No matching QQQ candle', 'No matching NQ candle',
         'Factors unavailable or no Flow Z event', 'Both demand scores meet threshold',
         'Both supply scores meet threshold', 'Opposing demand/supply scores'],
        default='Scores below confirmation threshold',
    )
    contiguous = result.index.to_series().diff().eq(duration)
    result['New Alert'] = (
        result['Signal'].isin(['BUY', 'SELL'])
        & (result['Signal'].ne(result['Signal'].shift()) | ~contiguous)
    )
    return result


def build_qqq_nq_signals(qqq_history, nq_history, timeframe='1h', threshold=10.0, as_of=None,
                         *, daily_alerts=None, alert_timeframe='1h', alert_threshold=1.0):
    """Calculate on full histories before any Ghana-day filter (preserves warmup)."""
    cutoff = pd.Timestamp.now(tz='UTC') if as_of is None else as_of
    qqq = completed_direction_history(qqq_history, timeframe, cutoff)
    nq = completed_direction_history(nq_history, timeframe, cutoff)
    base = combine_direction_histories(qqq, nq, timeframe, threshold)
    return confirm_with_daily_alerts(
        base, daily_alerts, timeframe,
        alert_timeframe=alert_timeframe, alert_threshold=alert_threshold,
    )


def signals_on_ghana_day(signals, selected_date):
    """Attribute a signal to when the completed candle can be evaluated."""
    times = pd.to_datetime(signals['Signal Time (Ghana)'], utc=True).dt.tz_convert('Africa/Accra')
    return signals.loc[times.dt.date == selected_date].copy()

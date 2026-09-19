"""Shared Ghana-day Flow-Z alerts and extreme-event demand/supply analysis."""

import numpy as np
import pandas as pd

from qqq_nq_signals import demand_supply_score, historical_z_components


COMPONENT_COLUMNS = ['Momentum (z)', 'Flow (z)', 'Volatility (z)', 'Trend (z)']


def classify_direction(score):
    if not np.isfinite(score):
        return 'Insufficient factor data'
    if score >= 10:
        return 'Candidate Demand Zone'
    if score <= -10:
        return 'Candidate Supply Zone'
    return 'Unresolved / Balanced Zone'


def analyze_ghana_flow_events(history, asset, selected_date):
    """Calculate the exact US Indices historical Flow-Z event logic for one day.

    Component changes are made against the full prior history before Ghana-day
    filtering, so the first event of a day uses its real preceding candle.
    """
    required = {'open', 'high', 'low', 'close', 'volume', 'momentum', 'atr14', 'z_score'}
    if history is None or history.empty or not required.issubset(history.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    data = history.copy()
    data.index = pd.to_datetime(data.index, utc=True)
    data = data[~data.index.duplicated(keep='last')].sort_index()
    components = historical_z_components(data)
    changes = components.diff()
    ghana_dates = data.index.tz_convert('Africa/Accra').date
    mask = np.asarray(ghana_dates) == selected_date
    selected_components = components.loc[mask].copy()
    selected_changes = changes.loc[mask].copy()
    events = []
    for event_time, flow_change in selected_changes['Flow (z)'].dropna().items():
        if flow_change == 0:
            continue
        position = data.index.get_indexer([event_time])[0]
        if position < 0:
            continue
        candle = data.iloc[position]
        previous_close = float(data['close'].iloc[position - 1]) if position else float(candle['open'])
        price_change = float(candle['close']) - previous_close
        momentum_change = float(changes.at[event_time, 'Momentum (z)'])
        volatility_change = float(changes.at[event_time, 'Volatility (z)'])
        trend_change = float(changes.at[event_time, 'Trend (z)'])
        score = float(demand_supply_score(
            float(flow_change), momentum_change, volatility_change, price_change,
        ))
        events.append({
            'Asset': asset,
            'Date': selected_date,
            'Alert': 'Flow Z Rise Alert' if flow_change > 0 else 'Flow Z Drop Alert',
            'Ghana Time': event_time,
            'Flow Z': float(components.at[event_time, 'Flow (z)']),
            'Flow Z Change': float(flow_change),
            'Momentum Z Change': momentum_change,
            'Trend Z Change': trend_change,
            'Volatility Z Change': volatility_change,
            'Price Change': price_change,
            'Zone Bottom': float(candle['low']),
            'Zone Top': float(candle['high']),
            'Close': float(candle['close']),
            'Direction Score': score,
            'Demand/Supply Alert': classify_direction(score),
        })
    return selected_components, selected_changes, pd.DataFrame(events)


def highest_flow_zone(events, direction):
    """Summarize the largest signed Flow-Z change with the shared 60/25/15 score."""
    if events is None or events.empty:
        return None
    is_rise = direction == 'rise'
    candidates = events.loc[
        events['Flow Z Change'].gt(0) if is_rise else events['Flow Z Change'].lt(0)
    ]
    if candidates.empty:
        return None
    event = candidates.loc[
        candidates['Flow Z Change'].idxmax() if is_rise else candidates['Flow Z Change'].idxmin()
    ].copy()
    flow = float(event['Flow Z Change'])
    momentum = float(event['Momentum Z Change'])
    volatility = float(event['Volatility Z Change'])
    price = float(event['Price Change'])
    volatility_direction = np.sign(price) * np.tanh(abs(volatility)) if volatility > 0 else 0.0
    flow_contribution = .60 * np.tanh(flow)
    momentum_contribution = .25 * np.tanh(momentum)
    volatility_contribution = .15 * volatility_direction
    score = 100 * (flow_contribution + momentum_contribution + volatility_contribution)
    event['Event Type'] = 'Highest Flow Z Rise' if is_rise else 'Highest Flow Z Drop'
    event['Event Size'] = abs(flow)
    event['Direction Score'] = score
    event['Classification'] = classify_direction(score)
    event['Volatility Regime'] = 'Expansion' if volatility > 0 else 'Compression' if volatility < 0 else 'Unchanged'
    event['Flow Contribution'] = flow_contribution * 100
    event['Momentum Contribution'] = momentum_contribution * 100
    event['Volatility Contribution'] = volatility_contribution * 100
    return event

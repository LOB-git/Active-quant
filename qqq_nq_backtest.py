"""NQ execution of the shared QQQ/NQ confirmation function, in Ghana/UTC time."""

import numpy as np
import pandas as pd

from qqq_nq_signals import build_qqq_nq_signals
from qqq_nq_daily_alerts import EVIDENCE_COLUMNS


FIVE_MINUTES = pd.Timedelta(minutes=5)


def utc_time(value):
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize('UTC') if timestamp.tzinfo is None else timestamp.tz_convert('UTC')


def clean_bars(frame):
    if frame is None or frame.empty:
        return pd.DataFrame(index=pd.DatetimeIndex([], tz='UTC'))
    required = ['open', 'high', 'low', 'close', 'volume']
    if not set(required).issubset(frame.columns):
        raise ValueError('Execution and liquidity history require open, high, low, close and volume.')
    data = frame[required].copy()
    data.index = pd.to_datetime(data.index, utc=True)
    data = data[~data.index.duplicated(keep='last')].sort_index()
    data[required] = data[required].apply(pd.to_numeric, errors='coerce')
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    return data.loc[
        (data[['open', 'high', 'low', 'close']] > 0).all(axis=1)
        & data['high'].ge(data[['open', 'close', 'low']].max(axis=1))
        & data['low'].le(data[['open', 'close', 'high']].min(axis=1))
        & data['volume'].ge(0)
    ]


def summarize_trades(trades, audit):
    summary = {'Trades': len(trades), 'New confirmations': len(audit),
               'Skipped': int(audit['Status'].ne('Traded').sum()) if len(audit) else 0,
               'Win Rate %': np.nan, 'Net R': 0.0, 'Average R': np.nan,
               'Net Points': 0.0, 'Profit Factor (points)': np.nan,
               'Max Closed Drawdown R': 0.0, 'Trades with bar gaps': 0}
    if trades.empty:
        return summary
    net = trades['Net Points']
    profit, loss = net.clip(lower=0).sum(), -net.clip(upper=0).sum()
    equity = trades['Net R'].cumsum()
    peak = equity.cummax().clip(lower=0)
    summary.update({
        'Win Rate %': float(net.gt(0).mean() * 100),
        'Net R': float(trades['Net R'].sum()), 'Average R': float(trades['Net R'].mean()),
        'Net Points': float(net.sum()),
        'Profit Factor (points)': float(profit / loss) if loss > 0 else (np.inf if profit > 0 else np.nan),
        'Max Closed Drawdown R': float((peak - equity).max()),
        'Trades with bar gaps': int(trades['Bar Gap'].sum()),
    })
    return summary


def execute_nq_confirmations(signals, execution_history, liquidity_history, *,
                             start, end_exclusive, as_of, zone_calculator=None,
                             atr_multiple=1.0, target_mode='liquidity', reward_risk=2.0,
                             slippage_points=0.25, commission_points=0.5):
    """Simulate new confirmations at the exact 5m bar open after signal availability.

    Indicators use only closed bars before entry; entries never use stale prices.
    Stop-first for ambiguous intrabar touches, opening gaps filled at the open
    for stops (targets at the limit), adverse slippage at both ends, one position.
    Return a full signal decision audit including missing-data/zone skips.
    """
    params = [atr_multiple, reward_risk, slippage_points, commission_points]
    if not all(np.isfinite(value) for value in params):
        raise ValueError('Risk and cost inputs must be finite numbers.')
    if atr_multiple <= 0 or reward_risk <= 0 or slippage_points < 0 or commission_points < 0:
        raise ValueError('Risk/target multiples must be positive and costs cannot be negative.')
    if target_mode not in ('liquidity', 'fixed_rr'):
        raise ValueError('Unknown take-profit mode.')
    if target_mode == 'liquidity' and zone_calculator is None:
        raise ValueError('The liquidity target requires the dashboard zone calculator.')
    beginning = utc_time(start)
    cutoff = min(utc_time(end_exclusive), utc_time(as_of))
    if beginning >= cutoff:
        raise ValueError('Choose a start date earlier than the available end time.')
    bars = clean_bars(execution_history)
    bars = bars.loc[(bars.index + FIVE_MINUTES) <= cutoff]
    liquidity = clean_bars(liquidity_history)
    liquidity = liquidity.loc[(liquidity.index + pd.Timedelta(hours=1)) <= cutoff]
    if bars.empty:
        raise ValueError('No completed NQ 5-minute execution candles are available.')
    tr = pd.concat([
        bars['high'] - bars['low'], (bars['high'] - bars['close'].shift()).abs(),
        (bars['low'] - bars['close'].shift()).abs(),
    ], axis=1).max(axis=1)
    atr_at_close = tr.rolling(14).mean()
    bar_closes = bars.index + FIVE_MINUTES
    candidates = signals.loc[
        signals['Signal'].isin(['BUY', 'SELL']) & signals['New Alert'].eq(True)
    ].copy()
    candidates['Signal Time (Ghana)'] = pd.to_datetime(candidates['Signal Time (Ghana)'], utc=True)
    candidates = candidates.loc[
        candidates['Signal Time (Ghana)'].ge(beginning)
        & candidates['Signal Time (Ghana)'].lt(cutoff)
    ].sort_values('Signal Time (Ghana)').drop_duplicates('Signal Time (Ghana)')
    trades, decisions = [], []
    next_free_time = beginning
    for _, signal in candidates.iterrows():
        timestamp = signal['Signal Time (Ghana)']
        decision = {'Signal Time (Ghana)': timestamp, 'Signal': signal['Signal'],
                    'QQQ Score': signal.get('QQQ Score', np.nan),
                    'NQ Score': signal.get('NQ Score', np.nan), 'Status': 'Skipped', 'Reason': ''}
        evidence = {name: signal[name] for name in ['Daily Alert Confirmation', *EVIDENCE_COLUMNS] if name in signal}
        decision.update(evidence)
        decisions.append(decision)
        if timestamp < next_free_time:
            decision['Reason'] = 'Earlier position still open'
            continue
        if timestamp not in bars.index:
            decision['Reason'] = 'No completed 5-minute entry candle at the exact signal time'
            continue
        entry_pos = bars.index.get_loc(timestamp)
        prior_pos = int(bar_closes.searchsorted(timestamp, side='right') - 1)
        if prior_pos < 0 or bar_closes[prior_pos] != timestamp:
            decision['Reason'] = 'Missing immediately preceding 5-minute candle for ATR'
            continue
        atr = float(atr_at_close.iloc[prior_pos])
        if not np.isfinite(atr) or atr <= 0:
            decision['Reason'] = 'Insufficient 14-candle ATR history'
            continue
        direction = 1 if signal['Signal'] == 'BUY' else -1
        entry_open = float(bars['open'].iloc[entry_pos])
        entry = entry_open + direction * slippage_points
        risk = atr * atr_multiple
        stop = entry - direction * risk
        zone_bottom = zone_top = np.nan
        if target_mode == 'fixed_rr':
            target = entry + direction * risk * reward_risk
        else:
            known = liquidity.loc[(liquidity.index + pd.Timedelta(hours=1)) <= timestamp]
            if len(known) < 50:
                decision['Reason'] = 'Fewer than 50 completed 1-hour candles for liquidity zones'
                continue
            zones, _ = zone_calculator(known, price_bins=24, top_zones=24)
            if zones is None or zones.empty:
                decision['Reason'] = 'No historical liquidity zones available'
                continue
            if direction == 1:
                targets = zones.loc[zones['Zone Bottom'] > entry].sort_values('Zone Bottom')
                edge = 'Zone Bottom'
            else:
                targets = zones.loc[zones['Zone Top'] < entry].sort_values('Zone Top', ascending=False)
                edge = 'Zone Top'
            if targets.empty:
                decision['Reason'] = 'No 1-hour liquidity zone ahead of entry'
                continue
            zone = targets.iloc[0]
            target, zone_bottom, zone_top = float(zone[edge]), float(zone['Zone Bottom']), float(zone['Zone Top'])
        exit_pos = len(bars) - 1
        exit_reference = float(bars['close'].iloc[-1])
        exit_reason = 'End of period' if bar_closes[-1] == utc_time(end_exclusive) else 'End of available data'
        exit_time = bar_closes[-1]
        gap, last_bar_time = False, timestamp
        for pos in range(entry_pos, len(bars)):
            candle = bars.iloc[pos]
            candle_time = bars.index[pos]
            if pos > entry_pos and candle_time - last_bar_time > FIVE_MINUTES:
                gap = True
            last_bar_time = candle_time
            opened, high, low = float(candle['open']), float(candle['high']), float(candle['low'])
            opening_stop = opened <= stop if direction == 1 else opened >= stop
            opening_target = opened >= target if direction == 1 else opened <= target
            stop_hit = low <= stop if direction == 1 else high >= stop
            target_hit = high >= target if direction == 1 else low <= target
            if opening_stop:
                exit_reference, exit_reason, exit_time = opened, 'Stop gap', candle_time
            elif opening_target:
                exit_reference, exit_reason, exit_time = target, 'Target at open', candle_time
            elif stop_hit:
                exit_reference = stop
                exit_reason = 'Stop (both touched)' if target_hit else 'Stop loss'
                exit_time = candle_time + FIVE_MINUTES
            elif target_hit:
                exit_reference, exit_reason, exit_time = target, 'Take profit', candle_time + FIVE_MINUTES
            else:
                continue
            exit_pos = pos
            break
        exit_fill = exit_reference - direction * slippage_points
        gross = direction * (exit_reference - entry_open)
        net = direction * (exit_fill - entry) - commission_points
        # Intrabar order is unknown: reserve the complete exit candle before re-entry.
        next_free_time = bars.index[exit_pos] + FIVE_MINUTES
        decision.update(Status='Traded', Reason=exit_reason)
        trades.append({
            'Trade': len(trades) + 1, 'Ghana Date': timestamp.date(),
            'Signal Time (Ghana)': timestamp, 'Entry Time (Ghana)': timestamp,
            'Direction': 'LONG' if direction == 1 else 'SHORT',
            'QQQ Score': signal.get('QQQ Score', np.nan), 'NQ Score': signal.get('NQ Score', np.nan),
            'Confirmation Score': signal.get('Confirmation Score', np.nan),
            'Entry Price': entry, 'ATR14': atr, 'Risk Points': risk,
            'Stop Price': stop, 'Target Price': target, 'Target Mode': target_mode,
            'Target Zone Bottom': zone_bottom, 'Target Zone Top': zone_top,
            'Planned Reward/Risk': abs(target - entry) / risk,
            'Exit Time (Ghana)': exit_time, 'Exit Price': exit_fill, 'Exit Reason': exit_reason,
            'Gross Points': gross, 'Cost Points': 2 * slippage_points + commission_points,
            'Net Points': net, 'Net R': net / risk, 'Bar Gap': gap,
            **evidence,
        })
    trades = pd.DataFrame(trades)
    audit = pd.DataFrame(decisions)
    summary = summarize_trades(trades, audit)
    if not trades.empty:
        trades['Cumulative R'] = trades['Net R'].cumsum()
        trades['Closed Drawdown R'] = trades['Cumulative R'].cummax().clip(lower=0) - trades['Cumulative R']
    return {'summary': summary, 'trades': trades, 'audit': audit,
            'execution_start': bars.index[0], 'execution_end': bar_closes[-1],
            'effective_end': cutoff}


def backtest_qqq_nq(qqq_history, nq_history, execution_history, liquidity_history, *,
                    timeframe='1h', threshold=10.0, start, end_exclusive, as_of=None,
                    daily_alerts=None, alert_timeframe='1h', alert_threshold=1.0,
                    zone_calculator=None, **execution_options):
    now = pd.Timestamp.now(tz='UTC') if as_of is None else utc_time(as_of)
    cutoff = min(now, utc_time(end_exclusive))
    signals = build_qqq_nq_signals(
        qqq_history, nq_history, timeframe, threshold, as_of=cutoff,
        daily_alerts=daily_alerts, alert_timeframe=alert_timeframe, alert_threshold=alert_threshold,
    )
    result = execute_nq_confirmations(
        signals, execution_history, liquidity_history, start=start,
        end_exclusive=end_exclusive, as_of=cutoff, zone_calculator=zone_calculator,
        **execution_options,
    )
    result['signals'] = signals.loc[
        signals['Signal Time (Ghana)'].ge(utc_time(start))
        & signals['Signal Time (Ghana)'].lt(cutoff)
    ]
    return result

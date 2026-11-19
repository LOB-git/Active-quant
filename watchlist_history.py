"""Daily storage for the Risk & Leverage Watch List.

Every scan's matches are appended to a CSV file, one row per (scan, criteria,
symbol). Days are Ghana days (Africa/Accra, which is UTC). The functions are
pure apart from the file read/write so they can be tested with a temp path.
"""
import os

import pandas as pd

HISTORY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'watchlist_history.csv')
GHANA_TZ = 'Africa/Accra'

KEY_COLUMNS = ['Scan Time (UTC)', 'Timeframe', 'Risk Target', 'Leverage Limit', 'Symbol']
VALUE_COLUMNS = ['Volume Rank', 'Verdict', 'Regime', 'Quant Score Q', '$ Risk', 'Leverage', 'Entry', 'Stop',
                 'Target', 'Size (units)', 'Notional', 'BT Expectancy R', 'BT Win Rate']
COLUMNS = ['Ghana Date', 'Ghana Time', *KEY_COLUMNS, *VALUE_COLUMNS]


def _scan_stamp(scanned_at):
    stamp = pd.Timestamp(scanned_at)
    return stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')


def load_history(path=None):
    """All stored watch list rows, or an empty frame with the stored columns."""
    path = path or HISTORY_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=COLUMNS)
    try:
        history = pd.read_csv(path, dtype={'Ghana Date': str, 'Scan Time (UTC)': str})
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame(columns=COLUMNS)
    return history.reindex(columns=COLUMNS)


def _key(frame):
    return set(zip(
        frame['Scan Time (UTC)'].astype(str), frame['Timeframe'].astype(str),
        pd.to_numeric(frame['Risk Target'], errors='coerce').round(2),
        pd.to_numeric(frame['Leverage Limit'], errors='coerce').round(4),
        frame['Symbol'].astype(str),
    ))


def record_matches(matches, scanned_at, timeframe, risk_target, leverage_limit, path=None):
    """Append this scan's watch list matches. Already-stored rows are skipped.

    Returns the number of rows written.
    """
    path = path or HISTORY_PATH
    if matches is None or matches.empty:
        return 0
    stamp = _scan_stamp(scanned_at)
    local = stamp.tz_convert(GHANA_TZ)
    rows = matches.rename(columns={'Rank': 'Volume Rank'}).reindex(columns=VALUE_COLUMNS + ['Symbol']).copy()
    rows['Ghana Date'] = local.strftime('%Y-%m-%d')
    rows['Ghana Time'] = local.strftime('%H:%M')
    rows['Scan Time (UTC)'] = stamp.isoformat()
    rows['Timeframe'] = timeframe
    rows['Risk Target'] = round(float(risk_target), 2)
    rows['Leverage Limit'] = round(float(leverage_limit), 4)
    rows = rows.reindex(columns=COLUMNS)

    stored = load_history(path)
    if not stored.empty:
        seen = _key(stored)
        rows = rows[[key not in seen for key in zip(
            rows['Scan Time (UTC)'], rows['Timeframe'], rows['Risk Target'].round(2),
            rows['Leverage Limit'].round(4), rows['Symbol'].astype(str),
        )]]
    if rows.empty:
        return 0
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    rows.to_csv(path, mode='a', header=write_header, index=False)
    return len(rows)


def available_dates(history):
    """Stored Ghana dates, newest first."""
    if history.empty:
        return []
    return sorted(history['Ghana Date'].dropna().astype(str).unique(), reverse=True)


def rows_for_day(history, day):
    """Every stored row for one Ghana date (a date, Timestamp or 'YYYY-MM-DD')."""
    if history.empty:
        return history
    wanted = pd.Timestamp(day).strftime('%Y-%m-%d')
    rows = history[history['Ghana Date'].astype(str) == wanted]
    return rows.sort_values(['Scan Time (UTC)', 'Symbol']).reset_index(drop=True)


def day_summary(history, day):
    """One row per asset that matched on that day, with its latest scan's trade plan."""
    rows = rows_for_day(history, day)
    if rows.empty:
        return pd.DataFrame(columns=['Symbol', 'Verdict', 'First Seen', 'Last Seen', 'Times Seen',
                                     'Leverage Limits', *VALUE_COLUMNS[3:]])
    groups = rows.groupby('Symbol', sort=False)
    latest = groups.tail(1).set_index('Symbol')
    summary = pd.DataFrame({
        'First Seen': groups['Ghana Time'].min(),
        'Last Seen': groups['Ghana Time'].max(),
        'Times Seen': groups['Scan Time (UTC)'].nunique(),
        'Verdicts': groups['Verdict'].agg(lambda v: ' / '.join(dict.fromkeys(v.astype(str)))),
        'Leverage Limits': groups['Leverage Limit'].agg(
            lambda v: ', '.join(f'<{x:.2f}x' for x in sorted(pd.to_numeric(v).unique()))),
    })
    summary = summary.join(latest[['Verdict', 'Regime', *VALUE_COLUMNS[3:]]])
    summary = summary.reset_index().sort_values(['First Seen', 'Symbol']).reset_index(drop=True)
    return summary[['Symbol', 'Verdict', 'Verdicts', 'First Seen', 'Last Seen', 'Times Seen', 'Leverage Limits',
                    'Regime', *VALUE_COLUMNS[3:]]]

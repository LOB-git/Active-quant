"""Implied volatility from Deribit: the DVOL index for BTC and ETH (30-day forward-looking
implied volatility built from Deribit options), shown next to realised volatility.

What the history says (daily, walk-forward, Jul 2021 - Oct 2026; see the tab caption):
  * DVOL is a better forecast of the next 20 days' realised volatility than trailing
    realised volatility (correlation 0.50 vs 0.30), but runs ~6-9 vol points too high
    (the volatility risk premium).
  * When DVOL is above realised volatility, realised volatility rose over the next 20 days
    65% (BTC) / 68% (ETH) of the time.
  * DVOL below its own median only weakly predicted DVOL rising (~56%), so the volatility
    outlook stays on realised volatility (volatility_quantum_scan.volatility_outlook).
"""
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from alt_iv import ALT_COINS, live_iv, load_snapshots, record_snapshot, snapshot_series, with_live
from live_state import live_cached

DERIBIT = 'https://www.deribit.com/api/v2/public/'
DVOL_CURRENCIES = ('BTC', 'ETH')
DAYS_PER_YEAR = 365   # crypto trades every day; DVOL is annualised on 365 days
RV_WINDOW = 20
HISTORY_DAYS = 400    # enough for a one-year median plus the realised-vol warm-up
MEDIAN_DAYS = 365
DAY_MS = 86_400_000


def perpetual_name(currency):
    """Deribit perpetual for a coin: BTC-PERPETUAL / ETH-PERPETUAL, SOL_USDC-PERPETUAL for the USDC-settled coins."""
    return f'{currency}-PERPETUAL' if currency in DVOL_CURRENCIES else f'{currency}_USDC-PERPETUAL'


def deribit_currency(symbol):
    """'BTC/USD', 'BTC-USD', 'BTCUSDT', 'eth/usdt:usdt' -> 'BTC' / 'ETH'; None for anything else."""
    if not symbol:
        return None
    base = str(symbol).upper().replace('-', '/').split('/')[0].split(':')[0]
    for currency in DVOL_CURRENCIES:
        if base == currency or (base.startswith(currency) and base[len(currency):] in ('USD', 'USDT', 'USDC')):
            return currency
    return None


def _call(method, params, get=requests.get):
    response = get(DERIBIT + method, params=params, timeout=20)
    response.raise_for_status()
    payload = response.json()
    if 'result' not in payload:
        raise RuntimeError(payload.get('error', {}).get('message', 'Deribit returned no result'))
    return payload['result']


def fetch_dvol(currency, days=HISTORY_DAYS, get=requests.get, now_ms=None):
    """Daily DVOL closes as fractions (0.40 = 40%), oldest first."""
    end = now_ms or int(time.time() * 1000)
    start, rows = end - days * DAY_MS, []
    while True:
        result = _call('get_volatility_index_data', {
            'currency': currency, 'start_timestamp': start, 'end_timestamp': end, 'resolution': 86400,
        }, get)
        rows += result.get('data') or []
        if not result.get('continuation') or result['continuation'] <= start:
            break
        end = result['continuation']
    if not rows:
        return pd.Series(dtype=float, name='dvol')
    frame = pd.DataFrame(rows).drop_duplicates(0).sort_values(0)
    index = pd.to_datetime(frame[0], unit='ms', utc=True).dt.normalize()
    series = pd.Series(frame[4].to_numpy() / 100, index=index, name='dvol')
    series.attrs['open'] = dict(zip(index, frame[1].to_numpy() / 100))   # DVOL at each 00:00 UTC
    return series


def fetch_daily_closes(currency, days=HISTORY_DAYS, get=requests.get, now_ms=None):
    """Daily closes of the Deribit perpetual, oldest first.

    Note: Deribit's daily perpetual candles run 08:00-08:00 UTC (the index labels them by date).
    They are only used for realised volatility; today's range uses fetch_day_open instead.
    """
    end = now_ms or int(time.time() * 1000)
    result = _call('get_tradingview_chart_data', {
        'instrument_name': perpetual_name(currency), 'start_timestamp': end - days * DAY_MS,
        'end_timestamp': end, 'resolution': '1D',
    }, get)
    if not result.get('ticks'):
        return pd.Series(dtype=float, name='close')
    index = pd.to_datetime(result['ticks'], unit='ms', utc=True).normalize()
    closes = pd.Series(result['close'], index=index, name='close')
    return closes[~closes.index.duplicated(keep='last')]


def fetch_day_open(currency, get=requests.get, now_ms=None):
    """(today's 00:00 UTC, perpetual price at 00:00 UTC) from the hourly candle that opens at midnight.

    00:00 UTC is 00:00 Ghana time, so this is the open of the Ghana day. None if Deribit has no candle.
    """
    now_ms = now_ms or int(time.time() * 1000)
    day_ms = now_ms // DAY_MS * DAY_MS
    result = _call('get_tradingview_chart_data', {
        'instrument_name': perpetual_name(currency), 'start_timestamp': day_ms,
        'end_timestamp': day_ms + 3_600_000, 'resolution': '60',
    }, get)
    for tick, price in zip(result.get('ticks') or [], result.get('open') or []):
        if tick == day_ms and price:
            return pd.Timestamp(day_ms, unit='ms', tz='UTC'), float(price)
    return None


def realised_volatility(closes, window=RV_WINDOW):
    """Annualised (365-day) standard deviation of the last `window` daily log returns."""
    return np.log(closes).diff().rolling(window).std() * np.sqrt(DAYS_PER_YEAR)


def dvol_at(dvol, moment):
    """DVOL as it stood at `moment` (a UTC midnight): the open of that day's DVOL candle (daily DVOL
    candles start at 00:00 UTC), else the last close before it. None if unknown."""
    if moment in dvol.index and np.isfinite(dvol.attrs.get('open', {}).get(moment, np.nan)):
        return float(dvol.attrs['open'][moment])
    before = dvol[dvol.index < moment].dropna()
    return float(before.iloc[-1]) if len(before) else None


def iv_summary(dvol, closes, day=None):
    """Numbers for the panel, from daily DVOL and daily closes. None without enough data.

    With `day` = (today's 00:00 UTC, price at 00:00 UTC) from fetch_day_open, the 1-day range is anchored
    to that open and to DVOL at 00:00 UTC, so it stays fixed for the whole Ghana (UTC) day like a daily
    pivot. Without it the range falls back to the live price and live DVOL.
    """
    anchor_iv = dvol_at(dvol, day[0]) if day else None
    dvol = dvol.dropna()
    rv = realised_volatility(closes).dropna()
    if dvol.empty or rv.empty or closes.dropna().empty:
        return None
    iv_now = float(dvol.iloc[-1])
    rv_now = float(rv.iloc[-1])
    recent = dvol.tail(MEDIAN_DAYS)
    price = float(closes.dropna().iloc[-1])
    weekly = iv_now * np.sqrt(7 / DAYS_PER_YEAR)
    anchor = None
    day_iv_live = False
    if day and day[1]:
        if anchor_iv:
            anchor = (float(day[1]), anchor_iv, day[0])
        elif iv_now:
            # No implied reading from 00:00 (coins tracked by snapshots): centre on the 00:00 open, size with live IV.
            anchor, day_iv_live = (float(day[1]), iv_now, day[0]), True
    if anchor:
        day_open, day_iv, day_start = anchor
    else:
        day_open, day_iv, day_start = price, iv_now, None
    daily = day_iv / np.sqrt(DAYS_PER_YEAR)
    sigma_now = (price / day_open - 1) / daily if daily > 0 else np.nan
    return {
        'iv': iv_now,
        'iv_date': dvol.index[-1],
        'iv_median': float(recent.median()),
        'iv_percentile': float((recent <= iv_now).mean() * 100),
        'rv': rv_now,
        'spread': iv_now - rv_now,
        'rv_outlook': 'INCREASE' if iv_now > rv_now else 'DECREASE' if iv_now < rv_now else 'NEUTRAL',
        'price': price,
        'daily_move_pct': daily,
        'weekly_move_pct': weekly,
        'daily_range': (day_open * (1 - daily), day_open * (1 + daily)),
        'day_anchored': anchor is not None,
        'day_iv_live': day_iv_live,
        'day_open': day_open,
        'day_iv': day_iv,
        'day_start': day_start,
        'day_sigma': sigma_now,             # where price is now, in daily moves from the open
        'day_position': ('ABOVE range' if sigma_now > 1 else 'BELOW range' if sigma_now < -1 else 'inside range')
        if np.isfinite(sigma_now) else '—',
        'weekly_range': (price * (1 - weekly), price * (1 + weekly)),
        'history': pd.DataFrame({'Implied (DVOL)': dvol, f'Realised {RV_WINDOW}d': rv}).dropna(how='all'),
    }




# ---------- loading: DVOL for BTC/ETH, Deribit USDC options + snapshots for the other coins ----------


IV_COINS = DVOL_CURRENCIES + ALT_COINS
MIN_HISTORY_DAYS = 30   # fewer days of implied history than this: no median / percentile yet


def iv_label(currency):
    return 'DVOL' if currency in DVOL_CURRENCIES else '30-day ATM IV'


def price_text(value):
    """Prices from BTC (64,890) down to TRX (0.3128) with 4 significant figures, trailing zeros kept
    so both ends of a range line up (XRP 0.9300 – 1.000, not 0.93 – 1)."""
    if value is None or not np.isfinite(value):
        return '—'
    if value == 0:
        return '0'
    decimals = max(0, 3 - int(np.floor(np.log10(abs(value)))))
    return f'{value:,.{decimals}f}'


def alt_summary(currency, reading, snapshots, closes, day):
    """iv_summary for a snapshot-tracked coin: stored readings plus the live one as today's value."""
    series = with_live(snapshot_series(snapshots, currency), reading)
    summary = iv_summary(series, closes, day)
    if summary:
        summary['history_days'] = int(series.dropna().index.normalize().nunique())
        summary['reading'] = reading
    return summary


def load_alt_summary(currency):
    reading = live_iv(currency)
    try:
        record_snapshot(currency, reading)
    except Exception:
        pass   # a read-only folder must not break the panel
    try:
        day = fetch_day_open(currency)
    except Exception:
        day = None
    return alt_summary(currency, reading, load_snapshots(), fetch_daily_closes(currency), day)


@st.cache_data(ttl=300, show_spinner=False)
def _load_iv_summary(currency, fresh_slot=0):
    if currency not in DVOL_CURRENCIES:
        return load_alt_summary(currency)
    try:
        day = fetch_day_open(currency)
    except Exception:
        day = None   # fall back to a range around the live price
    summary = iv_summary(fetch_dvol(currency), fetch_daily_closes(currency), day)
    if summary:
        summary['history_days'] = int(summary['history']['Implied (DVOL)'].notna().sum())
    return summary


load_iv_summary = live_cached(_load_iv_summary)  # refreshes with the dashboard's live updates


def iv_chart(summary, currency):
    history = summary['history'].tail(MEDIAN_DAYS)
    dvol = currency in DVOL_CURRENCIES
    name = 'Implied (DVOL)' if dvol else 'Implied (30-day ATM)'
    fig = go.Figure()
    fig.add_scatter(x=history.index, y=history['Implied (DVOL)'] * 100, name=name,
                    mode='lines' if dvol else 'lines+markers', connectgaps=True,
                    line={'color': '#7c3aed', 'width': 2})
    fig.add_scatter(x=history.index, y=history[f'Realised {RV_WINDOW}d'] * 100, name=f'Realised {RV_WINDOW}d',
                    line={'color': '#0ea5e9', 'width': 2})
    if summary.get('history_days', MIN_HISTORY_DAYS) >= MIN_HISTORY_DAYS:
        fig.add_hline(y=summary['iv_median'] * 100, line_dash='dash', line_color='#7c3aed', opacity=0.6,
                      annotation_text=f'{iv_label(currency)} 1-year median', annotation_position='top left')
    fig.update_layout(title=f'{currency} implied vs realised volatility (annualised %)', height=380,
                      margin={'l': 10, 'r': 10, 't': 50, 'b': 10}, yaxis_title='%', hovermode='x unified',
                      legend={'orientation': 'h', 'y': -0.15})
    return fig


def render_implied_volatility_panel(default_symbol=None):
    st.subheader('📈 Implied Volatility (Deribit options)')
    st.caption(
        "Implied volatility is what the options market expects volatility to be over the next month. **BTC, ETH:** "
        "Deribit's DVOL index. **SOL, XRP, AVAX, HYPE, TRX:** 30-day at-the-money implied volatility from Deribit's "
        'USDC options (no DVOL exists for them). Realised volatility is what price actually did over the last 20 days.'
    )
    default = deribit_currency(default_symbol) or 'BTC'
    currency = st.radio('Asset', IV_COINS, index=IV_COINS.index(default), horizontal=True, key='iv_currency')
    label = iv_label(currency)
    try:
        summary = load_iv_summary(currency)
    except Exception as exc:
        st.warning(f'Could not load implied volatility from Deribit: {exc}')
        return None
    if summary is None:
        st.warning(f'Deribit returned too little data for {currency}.')
        return None

    enough = summary.get('history_days', 0) >= MIN_HISTORY_DAYS
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f'Implied vol ({label})', f"{summary['iv']:.1%}",
              f"{(summary['iv'] - summary['iv_median']) * 100:+.1f} pts vs 1-yr median" if enough
              else f"history building: {summary.get('history_days', 0)} day(s)", delta_color='off', delta_arrow='off')
    c2.metric(f'{label} percentile (1 yr)', f"{summary['iv_percentile']:.0f}%" if enough else '—',
              help=None if enough else f'Needs {MIN_HISTORY_DAYS} days of saved readings.')
    c3.metric(f'Realised vol ({RV_WINDOW}d)', f"{summary['rv']:.1%}")
    c4.metric('Implied − realised', f"{summary['spread'] * 100:+.1f} pts")

    m1, m2, m3 = st.columns(3)
    low, high = summary['daily_range']
    day_label = "Today's range (from 00:00 UTC open)" if summary['day_anchored'] else 'Expected 1-day move'
    m1.metric(day_label, f'{price_text(low)} – {price_text(high)}', f"±{summary['daily_move_pct']:.2%}",
              delta_color='off', delta_arrow='off')
    m2.metric("Price now vs today's range", price_text(summary['price']),
              f"{summary['day_sigma']:+.2f}× the daily move · {summary['day_position']}",
              delta_color='off', delta_arrow='off')
    low, high = summary['weekly_range']
    m3.metric('Expected 7-day move (from now)', f"±{summary['weekly_move_pct']:.2%}",
              f'{price_text(low)} – {price_text(high)}', delta_color='off', delta_arrow='off')
    if summary['day_anchored']:
        size = (f"live {label} ({summary['day_iv']:.1%}): no reading was saved near 00:00 UTC today, so the size "
                'can still change as implied volatility moves' if summary.get('day_iv_live')
                else f"{label} at 00:00 ({summary['day_iv']:.1%})")
        st.caption(
            f"Today's range for the Ghana day ({summary['day_start']:%Y-%m-%d} 00:00–24:00 UTC): centre = the "
            f"00:00 UTC open {price_text(summary['day_open'])}, size = {size} / √365 = ±{summary['daily_move_pct']:.2%}. "
            'It is one standard deviation, so price typically closes outside it about 1 day in 3.'
        )

    tested = currency in DVOL_CURRENCIES
    hit_note = ('Reading implied vs realised this way called the direction of realised volatility over the '
                'next 20 days right about 65–68% of the time in the backtest'
                + ('.' if tested else f' (tested on BTC and ETH DVOL, not on {currency}).'))
    if summary['rv_outlook'] == 'INCREASE':
        st.info('**Implied is above realised → realised volatility likely to RISE.** The options market is pricing '
                f'more movement than price has shown lately. {hit_note}')
    elif summary['rv_outlook'] == 'DECREASE':
        st.info('**Implied is below realised → realised volatility likely to FALL.** The options market expects '
                f'things to calm down from the recent pace. {hit_note}')

    st.plotly_chart(iv_chart(summary, currency), width='stretch')
    if tested:
        st.caption(
            f"DVOL as of {summary['iv_date']:%Y-%m-%d} UTC (cached 5 min), price {price_text(summary['price'])} from "
            f'the {perpetual_name(currency)}. Moves are one standard deviation: DVOL / √365 for a day, DVOL × √(7/365) '
            'for a week — a size, not a direction. Backtest Jul 2021 – Oct 2026: DVOL tracked the next 20 days of '
            'realised volatility better than trailing realised volatility (correlation 0.50 vs 0.30) but usually ran '
            '6–9 points too high, because options carry a premium. The 65–68% hit rates are from that same test.'
        )
    else:
        reading = summary.get('reading') or {}
        st.caption(
            f"{currency} implied volatility (cached 5 min): at-the-money options on Deribit's {currency}_USDC "
            f"expiries {reading.get('near_days', float('nan')):.0f} and {reading.get('far_days', float('nan')):.0f} days "
            'out, blended to exactly 30 days. It is at-the-money only, so it reads a little below a DVOL-style index '
            '(on BTC the same method gave 43.6% when DVOL was 45.0%). Deribit keeps no history of it: each reading is '
            f'saved to iv_snapshots.csv (the alert runner adds one every hour), so the 1-year median, percentile and '
            f'the date picker below fill in from the day saving started. Price from the {perpetual_name(currency)}. '
            'BNB and DOGE are not offered: their options only run 8 days out.'
        )
    st.divider()
    from iv_history import render_iv_history
    render_iv_history(currency)
    return summary

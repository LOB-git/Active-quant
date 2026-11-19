"""Historical implied volatility for BTC and ETH: pick a date, see that day's DVOL, range and outcome.

Each day is rebuilt exactly the way the 📈 Implied Volatility panel builds today:
  * DVOL at 00:00 UTC (the open of that day's DVOL candle), Deribit's 30-day implied volatility.
  * One-day range = 00:00 UTC open of the perpetual × (1 ± DVOL / √365), fixed for the Ghana (UTC) day.
  * 7-day range = the same open × (1 ± DVOL × √(7/365)).
Then it adds what actually happened: the day's high, low and close (from Deribit hourly candles,
grouped into UTC days), whether the close stayed inside the range, and the realised volatility
known at 00:00 (20 daily UTC closes up to the day before).
"""
import numpy as np
import pandas as pd
import requests
import streamlit as st

from alt_iv import load_snapshots, snapshot_series
from implied_volatility_panel import (DAYS_PER_YEAR, DVOL_CURRENCIES, RV_WINDOW, _call, dvol_at, fetch_dvol, iv_label,
                                      perpetual_name, price_text)

DVOL_START = pd.Timestamp('2021-03-24', tz='UTC')   # first DVOL day for BTC and ETH
FIRST_DAY = DVOL_START + pd.Timedelta(days=RV_WINDOW + 1)
TABLE_DAYS = 30
WEEK = 7


def utc_day(value):
    """A date / Timestamp / 'YYYY-MM-DD' -> that day's 00:00 UTC."""
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
    return stamp.normalize()


def fetch_hourly(currency, start, end, get=requests.get):
    """Hourly OHLC of the Deribit perpetual from `start` to `end` (UTC Timestamps), oldest first.

    Deribit returns at most ~1440 candles per call, so longer spans are fetched in 50-day pages.
    """
    frames, page = [], pd.Timedelta(days=50)
    cursor = start
    while cursor < end:
        stop = min(cursor + page, end)
        result = _call('get_tradingview_chart_data', {
            'instrument_name': perpetual_name(currency), 'start_timestamp': int(cursor.value // 1_000_000),
            'end_timestamp': int(stop.value // 1_000_000) - 1, 'resolution': '60',
        }, get)
        if result.get('ticks'):
            frames.append(pd.DataFrame(
                {key: result[key] for key in ('open', 'high', 'low', 'close')},
                index=pd.to_datetime(result['ticks'], unit='ms', utc=True)))
        cursor = stop
    if not frames:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close'])
    hourly = pd.concat(frames).astype(float)
    return hourly[~hourly.index.duplicated(keep='last')].sort_index()


def utc_daily(hourly):
    """Hourly candles -> UTC-day open/high/low/close with the number of hours seen (24 = complete day)."""
    if hourly.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'hours'])
    grouped = hourly.groupby(hourly.index.floor('D'))
    daily = pd.DataFrame({'open': grouped['open'].first(), 'high': grouped['high'].max(),
                          'low': grouped['low'].min(), 'close': grouped['close'].last(),
                          'hours': grouped['close'].size()})
    # The open must be the 00:00 candle; a day that starts later has no 00:00 open to anchor the range to.
    first_hour = grouped.apply(lambda part: part.index[0].hour)
    daily.loc[first_hour != 0, 'open'] = np.nan
    return daily


def history_table(dvol, daily):
    """One row per UTC day: DVOL at 00:00, the ranges it gave, and what price did."""
    if daily.empty:
        return pd.DataFrame()
    complete = daily[daily['hours'] == 24]
    rv = np.log(complete['close']).diff().rolling(RV_WINDOW).std() * np.sqrt(DAYS_PER_YEAR)
    rv.index = rv.index + pd.Timedelta(days=1)   # known at 00:00 of the next day, finished or not
    rows = []
    for day, bar in daily.iterrows():
        iv = dvol_at(dvol, day)
        if iv is None or not np.isfinite(bar['open']):
            continue
        move = iv / np.sqrt(DAYS_PER_YEAR)
        week_move = iv * np.sqrt(WEEK / DAYS_PER_YEAR)
        low, high = bar['open'] * (1 - move), bar['open'] * (1 + move)
        done = bar['hours'] == 24
        sigma = (bar['close'] / bar['open'] - 1) / move
        result = ('inside' if abs(sigma) <= 1 else 'closed above' if sigma > 0 else 'closed below') if done \
            else 'in progress'
        week_end = day + pd.Timedelta(days=WEEK - 1)
        week_close = daily['close'].get(week_end) if week_end in daily.index and daily.at[week_end, 'hours'] == 24 \
            else np.nan
        week_sigma = (week_close / bar['open'] - 1) / week_move if np.isfinite(week_close) else np.nan
        rows.append({
            'Date': day, 'DVOL @00:00': iv, 'DVOL close': float(dvol.get(day, np.nan)),
            'Realised 20d': float(rv.get(day, np.nan)), 'Daily move': move, 'Open 00:00': bar['open'],
            'Range low': low, 'Range high': high, 'High': bar['high'], 'Low': bar['low'], 'Close': bar['close'],
            'Close (× move)': sigma, 'Result': result,
            'Wicks inside': bool(bar['high'] <= high and bar['low'] >= low) if done else None,
            '7-day move': week_move, '7-day low': bar['open'] * (1 - week_move),
            '7-day high': bar['open'] * (1 + week_move), '7-day close': week_close, '7-day (× move)': week_sigma,
        })
    table = pd.DataFrame(rows)
    if not table.empty:
        table['IV − RV'] = table['DVOL @00:00'] - table['Realised 20d']
    return table


def dvol_percentile(dvol, day, days=365):
    """Where DVOL at that day's close sat within the year before it (0-100)."""
    year = dvol[(dvol.index > day - pd.Timedelta(days=days)) & (dvol.index <= day)].dropna()
    return float((year <= year.iloc[-1]).mean() * 100) if len(year) else np.nan


def day_window(day, now=None):
    """(start, end) of the hourly candles needed for `day`: 30-day table + 20-day RV warm-up, + 7 days ahead."""
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    start = day - pd.Timedelta(days=TABLE_DAYS + RV_WINDOW + 2)
    end = min(day + pd.Timedelta(days=WEEK + 1), now.floor('h'))
    return max(start, DVOL_START), end


def load_day(currency, day, dvol, get=requests.get, now=None):
    """(table of the TABLE_DAYS ending on `day`, row for `day` or None) using hourly candles from Deribit."""
    day = utc_day(day)
    start, end = day_window(day, now)
    table = history_table(dvol, utc_daily(fetch_hourly(currency, start, end, get)))
    if table.empty:
        return table, None
    shown = table[(table['Date'] <= day) & (table['Date'] > day - pd.Timedelta(days=TABLE_DAYS))]
    selected = table[table['Date'] == day]
    return shown.reset_index(drop=True), (selected.iloc[0] if len(selected) else None)


# ---------- Streamlit section ----------

@st.cache_data(ttl=300, show_spinner=False)
def load_dvol_history(currency):
    """BTC/ETH: DVOL since 2021. Other coins: their saved 30-day implied snapshots (alt_iv)."""
    if currency not in DVOL_CURRENCIES:
        return snapshot_series(load_snapshots(), currency)
    days = (pd.Timestamp.now(tz='UTC') - DVOL_START).days + 2
    return fetch_dvol(currency, days=days)


@st.cache_data(ttl=300, show_spinner=False)
def load_day_cached(currency, day):
    return load_day(currency, day, load_dvol_history(currency))


def _range_text(low, high):
    return f'{price_text(low)} – {price_text(high)}'


def first_day(currency, dvol):
    """Earliest date the picker allows: DVOL start + RV warm-up, or the first saved snapshot day."""
    if currency in DVOL_CURRENCIES:
        return FIRST_DAY
    days = list(dvol.index) + list(dvol.attrs.get('open', {}))
    return min(days) if days else None


def render_iv_history(currency):
    label = iv_label(currency)
    st.markdown(f'#### 📅 {currency} implied volatility on a past date')
    today = pd.Timestamp.now(tz='UTC').normalize()
    try:
        dvol = load_dvol_history(currency)
    except Exception as exc:
        st.warning(f'Could not load implied volatility history from Deribit: {exc}')
        return
    earliest = first_day(currency, dvol)
    if earliest is None:
        saved = load_snapshots()
        saved = saved[saved['Coin'] == currency]
        started = (f"Saving started {saved['Time (UTC)'].min():%Y-%m-%d %H:%M} UTC ({len(saved)} reading(s) so far). "
                   if len(saved) else f'No {currency} readings saved yet. ')
        st.info(f'Deribit keeps no implied volatility history for {currency}, so the dashboard saves its own: each time '
                'this panel loads, and every hour while the alert runner is open. ' + started +
                'A date appears here once there is a reading within 90 minutes of 00:00 UTC, so keep the alert runner '
                'open overnight.')
        return
    picked = st.date_input(
        'Date (UTC = Ghana day)', value=max(today - pd.Timedelta(days=1), earliest).date(),
        min_value=earliest.date(), max_value=today.date(), key=f'iv_hist_day_{currency}',
        help=(f'DVOL starts on {DVOL_START:%Y-%m-%d}; the first {RV_WINDOW + 1} days are needed for realised '
              'volatility.' if currency in DVOL_CURRENCIES else
              f'{currency} implied volatility has been saved since {earliest:%Y-%m-%d}.'),
    )
    try:
        with st.spinner(f'Loading {label} and Deribit prices…'):
            table, row = load_day_cached(currency, utc_day(picked))
    except Exception as exc:
        st.warning(f'Could not load history from Deribit: {exc}')
        return
    if row is None:
        st.warning(f'No {label} reading near 00:00 UTC (or no 00:00 UTC price) for {currency} on {picked:%Y-%m-%d}.'
                   + ('' if currency in DVOL_CURRENCIES else
                      ' Readings are only saved while the dashboard or the alert runner is open.'))
        return

    day = row['Date']
    c1, c2, c3, c4 = st.columns(4)
    finished = row['Result'] != 'in progress'
    dvol_after = 'closed the day at' if finished else 'now at'
    c1.metric(f'{label} at 00:00 UTC', f"{row['DVOL @00:00']:.1%}",
              f"{dvol_after} {row['DVOL close']:.1%}" if np.isfinite(row['DVOL close']) else None,
              delta_color='off', delta_arrow='off')
    c2.metric(f'{label} percentile (year before)', f'{dvol_percentile(dvol, day):.0f}%')
    c3.metric(f'Realised vol ({RV_WINDOW}d) at 00:00', f"{row['Realised 20d']:.1%}"
              if np.isfinite(row['Realised 20d']) else '—')
    c4.metric('Implied − realised', f"{row['IV − RV'] * 100:+.1f} pts" if np.isfinite(row['IV − RV']) else '—')

    m1, m2, m3 = st.columns(3)
    m1.metric('One-day range (from 00:00 open)', _range_text(row['Range low'], row['Range high']),
              f"±{row['Daily move']:.2%} around {price_text(row['Open 00:00'])}", delta_color='off', delta_arrow='off')
    m2.metric('Day close' if finished else 'Price so far today', price_text(row['Close']),
              f"{row['Close (× move)']:+.2f}× the daily move · {row['Result']}", delta_color='off', delta_arrow='off')
    week_done = np.isfinite(row['7-day close'])
    m3.metric('7-day range (from 00:00 open)', _range_text(row['7-day low'], row['7-day high']),
              (f"closed {price_text(row['7-day close'])} after 7 days ({row['7-day (× move)']:+.2f}×)" if week_done
               else '7 days not finished yet'), delta_color='off', delta_arrow='off')
    if finished:
        wicks = 'stayed inside the range all day' if row['Wicks inside'] else 'went outside the range during the day'
        st.caption(f"{day:%A %Y-%m-%d}: high {price_text(row['High'])}, low {price_text(row['Low'])}; price {wicks}.")

    done = table[table['Result'] != 'in progress']
    if len(done):
        inside = (done['Result'] == 'inside').mean()
        st.markdown(
            f"**{len(done)} days ending {day:%Y-%m-%d}:** the close stayed inside the one-day range on "
            f"**{inside:.0%}** of days (a correctly sized range: about 68%), and the whole day including wicks on "
            f"{done['Wicks inside'].mean():.0%}."
        )
    shown = table.sort_values('Date', ascending=False)
    columns = ['Date', 'DVOL @00:00', 'Realised 20d', 'Daily move', 'Open 00:00', 'Range low', 'Range high',
               'High', 'Low', 'Close', 'Close (× move)', 'Result', 'Wicks inside']
    st.dataframe(
        shown[columns].style.format({
            'Date': lambda d: f'{d:%Y-%m-%d %a}', 'DVOL @00:00': '{:.1%}', 'Realised 20d': '{:.1%}',
            'Daily move': '±{:.2%}', 'Open 00:00': price_text, 'Range low': price_text, 'Range high': price_text,
            'High': price_text, 'Low': price_text, 'Close': price_text, 'Close (× move)': '{:+.2f}',
            'Wicks inside': lambda v: '—' if v is None or v != v else ('yes' if v else 'no'),
        }, na_rep='—').map(lambda v: {'inside': 'color: #16a34a', 'closed above': 'color: #dc2626',
                                      'closed below': 'color: #dc2626'}.get(v, ''), subset=['Result']),
        width='stretch', hide_index=True, height=360,
    )
    st.download_button(
        'Download these days (CSV)', shown.assign(Date=shown['Date'].dt.strftime('%Y-%m-%d')).to_csv(index=False).encode(),
        file_name=f'{currency}_implied_volatility_{day:%Y%m%d}.csv', mime='text/csv', key=f'iv_hist_csv_{currency}',
    )
    st.caption(
        f"Each day uses {label} at 00:00 UTC and the perpetual's 00:00 UTC open, exactly like today's range above, "
        'so it is what the panel would have shown that morning. High, low and close come from Deribit hourly '
        'candles for the UTC day. Realised vol is from the 20 daily UTC closes before the day. "Close (× move)" is '
        'how far the close ended from the open in daily moves: between −1 and +1 means inside.'
    )

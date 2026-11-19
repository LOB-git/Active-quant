"""Crypto gamma exposure (GEX) for BTC and ETH from Deribit options, plus an hourly
snapshot log (gex_history.csv) so the signal can be tested once enough history exists.

GEX per option = gamma x open interest x spot^2 x 1%, i.e. the dollars of the coin dealers
would trade for a 1% move. The usual convention is used: dealers are short calls (+) and
long puts (-). That is an assumption: in crypto, call overwriting can leave dealers long
calls, which flips the call side. Deribit's free API has no historical open interest,
which is why snapshots are logged going forward.
"""
import os
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.stats import norm
from live_state import live_cached

HERE = os.path.dirname(os.path.abspath(__file__))
GEX_HISTORY_PATH = os.path.join(HERE, 'gex_history.csv')
DERIBIT = 'https://www.deribit.com/api/v2/public/'
GEX_CURRENCIES = ('BTC', 'ETH')
FLIP_RANGE = 0.20        # look for the zero-gamma level within +/-20% of spot
CHART_RANGE = 0.25       # strikes shown on the chart: within +/-25% of spot
YEAR_SECONDS = 365 * 86400
HISTORY_COLUMNS = ['Snapshot Hour (UTC)', 'Captured At (UTC)', 'Currency', 'Spot', 'Net GEX', 'Call GEX', 'Put GEX',
                   'Next Expiry GEX', 'Share 7d', 'Flip Level', 'Call Wall', 'Put Wall', 'Total OI', 'Options',
                   'Next Expiry', 'Source']


def fetch_option_book(currency, get=requests.get):
    response = get(DERIBIT + 'get_book_summary_by_currency', params={'currency': currency, 'kind': 'option'},
                   timeout=20)
    response.raise_for_status()
    payload = response.json()
    if 'result' not in payload:
        raise RuntimeError(payload.get('error', {}).get('message', 'Deribit returned no result'))
    return pd.DataFrame(payload['result'])


def parse_options(book, now=None):
    """Deribit book summary -> one row per live option with open interest.

    Columns: instrument, expiry, strike, type ('C'/'P'), T (years), iv (fraction), oi, spot.
    Deribit options expire at 08:00 UTC; instrument names look like BTC-27DEC26-80000-C.
    """
    now = now or pd.Timestamp.now(tz='UTC')
    if book is None or book.empty:
        return pd.DataFrame(columns=['instrument', 'expiry', 'strike', 'type', 'T', 'iv', 'oi', 'spot'])
    parts = book['instrument_name'].str.split('-', expand=True)
    frame = pd.DataFrame({
        'instrument': book['instrument_name'],
        'expiry': pd.to_datetime(parts[1], format='%d%b%y', utc=True) + pd.Timedelta(hours=8),
        'strike': pd.to_numeric(parts[2], errors='coerce'),
        'type': parts[3],
        'iv': pd.to_numeric(book['mark_iv'], errors='coerce') / 100,
        'oi': pd.to_numeric(book['open_interest'], errors='coerce'),
        'spot': pd.to_numeric(book['underlying_price'], errors='coerce'),
    })
    frame['T'] = (frame['expiry'] - now).dt.total_seconds() / YEAR_SECONDS
    keep = (frame['T'] > 0) & (frame['iv'] > 0) & (frame['oi'] > 0) & frame['strike'].gt(0) & frame['type'].isin(['C', 'P'])
    return frame[keep].reset_index(drop=True)


def option_gex(options, spot):
    """Dollar GEX per 1% move for each option at `spot` (calls +, puts -)."""
    sig, T, K = options['iv'].to_numpy(), options['T'].to_numpy(), options['strike'].to_numpy()
    d1 = (np.log(spot / K) + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T))
    gamma = norm.pdf(d1) / (spot * sig * np.sqrt(T))
    sign = np.where(options['type'].to_numpy() == 'C', 1.0, -1.0)
    return gamma * options['oi'].to_numpy() * spot ** 2 * 0.01 * sign


def flip_level(options, spot, points=161):
    """Price within +/-FLIP_RANGE of spot where net GEX crosses zero (nearest to spot), or None."""
    if options.empty:
        return None
    grid = np.linspace(spot * (1 - FLIP_RANGE), spot * (1 + FLIP_RANGE), points)
    totals = np.array([option_gex(options, level).sum() for level in grid])
    crossings = np.where(np.diff(np.sign(totals)) != 0)[0]
    if not len(crossings):
        return None
    levels = []
    for i in crossings:  # linear interpolation between the two grid points
        x0, x1, y0, y1 = grid[i], grid[i + 1], totals[i], totals[i + 1]
        levels.append(x0 - y0 * (x1 - x0) / (y1 - y0) if y1 != y0 else x0)
    return float(min(levels, key=lambda level: abs(level - spot)))


def gex_snapshot(options, currency, now=None, spot=None):
    """Summary numbers for one currency. None when there are no live options.

    `spot` defaults to the median underlying price across the options passed in; pass the
    full-book spot when summarising a subset (e.g. one expiry) so both views agree.
    """
    if options.empty:
        return None
    now = now or pd.Timestamp.now(tz='UTC')
    spot = float(spot) if spot else float(options['spot'].median())
    options = options.assign(gex=option_gex(options, spot))
    by_strike = options.groupby('strike')['gex'].sum()
    next_expiry = options['expiry'].min()
    net = float(options['gex'].sum())
    return {
        'currency': currency,
        'captured_at': now,
        'spot': spot,
        'net_gex': net,
        'call_gex': float(options.loc[options['type'] == 'C', 'gex'].sum()),
        'put_gex': float(options.loc[options['type'] == 'P', 'gex'].sum()),
        'next_expiry': next_expiry,
        'next_expiry_gex': float(options.loc[options['expiry'] == next_expiry, 'gex'].sum()),
        'share_7d': float(options.loc[options['T'] < 7 / 365, 'gex'].abs().sum() / options['gex'].abs().sum()),
        'flip': flip_level(options, spot),
        'call_wall': float(by_strike.idxmax()) if (by_strike > 0).any() else None,
        'put_wall': float(by_strike.idxmin()) if (by_strike < 0).any() else None,
        'total_oi': float(options['oi'].sum()),
        'count': int(len(options)),
        'regime': 'POSITIVE' if net > 0 else 'NEGATIVE',
        'options': options,
    }


def snapshot_row(snap, source):
    hour = snap['captured_at'].floor('h')
    return {
        'Snapshot Hour (UTC)': hour.strftime('%Y-%m-%d %H:00'), 'Captured At (UTC)': snap['captured_at'].isoformat(),
        'Currency': snap['currency'], 'Spot': snap['spot'], 'Net GEX': snap['net_gex'], 'Call GEX': snap['call_gex'],
        'Put GEX': snap['put_gex'], 'Next Expiry GEX': snap['next_expiry_gex'], 'Share 7d': snap['share_7d'],
        'Flip Level': snap['flip'], 'Call Wall': snap['call_wall'], 'Put Wall': snap['put_wall'],
        'Total OI': snap['total_oi'], 'Options': snap['count'],
        'Next Expiry': snap['next_expiry'].strftime('%Y-%m-%d %H:%M'), 'Source': source,
    }


def load_gex_history(path=None):
    path = path or GEX_HISTORY_PATH
    if not os.path.exists(path):
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.read_csv(path)


def record_snapshot(snap, source, path=None):
    """Append the snapshot unless this currency already has one for this UTC hour. Returns True if saved."""
    path = path or GEX_HISTORY_PATH
    row = snapshot_row(snap, source)
    history = load_gex_history(path)
    if not history.empty and ((history['Snapshot Hour (UTC)'] == row['Snapshot Hour (UTC)'])
                              & (history['Currency'] == row['Currency'])).any():
        return False
    pd.DataFrame([row], columns=HISTORY_COLUMNS).to_csv(path, mode='a', header=not os.path.exists(path), index=False)
    return True


def log_gex_snapshots(currencies=GEX_CURRENCIES, source='runner', path=None, get=requests.get):
    """Fetch and record one snapshot per currency. Returns {currency: 'saved' | 'exists' | error text}."""
    results = {}
    for currency in currencies:
        try:
            snap = gex_snapshot(parse_options(fetch_option_book(currency, get)), currency)
            if snap is None:
                results[currency] = 'no live options'
            else:
                results[currency] = 'saved' if record_snapshot(snap, source, path) else 'exists'
        except Exception as exc:
            results[currency] = f'failed: {exc}'
    return results


@st.cache_data(ttl=300, show_spinner=False)
def _load_options(currency, fresh_slot=0):
    return parse_options(fetch_option_book(currency)), pd.Timestamp.now(tz='UTC')


load_options = live_cached(_load_options)  # refreshes with the dashboard's live updates


def _money(value):
    sign = '-' if value < 0 else '+'
    value = abs(value)
    return f'{sign}${value / 1e9:,.2f}B' if value >= 1e9 else f'{sign}${value / 1e6:,.1f}M'


def gex_chart(snap, currency):
    options = snap['options']
    spot = snap['spot']
    near = options[(options['strike'] - spot).abs() <= spot * CHART_RANGE]
    by_strike = near.groupby(['strike', 'type'])['gex'].sum().unstack(fill_value=0.0)
    fig = go.Figure()
    for kind, name, color in (('C', 'Calls (+)', '#16a34a'), ('P', 'Puts (−)', '#dc2626')):
        if kind in by_strike:
            fig.add_bar(x=by_strike.index, y=by_strike[kind] / 1e6, name=name, marker_color=color)
    fig.add_vline(x=spot, line_color='#0ea5e9', line_width=2, annotation_text=f'Spot {spot:,.0f}')
    if snap['flip']:
        fig.add_vline(x=snap['flip'], line_dash='dash', line_color='#a855f7',
                      annotation_text=f"Flip {snap['flip']:,.0f}", annotation_position='bottom right')
    fig.update_layout(barmode='relative', title=f'{currency} GEX by strike ($M per 1% move)', height=400,
                      margin={'l': 10, 'r': 10, 't': 50, 'b': 10}, xaxis_title='Strike', yaxis_title='$M',
                      legend={'orientation': 'h', 'y': -0.2}, hovermode='x unified')
    return fig


def render_gex_panel(default_symbol=None):
    st.subheader('🧲 Gamma Exposure (GEX) — Deribit options')
    st.caption(
        'Estimates how much BTC/ETH option dealers must buy or sell to stay hedged as price moves. '
        '**Positive GEX** — dealers buy dips and sell rallies, which tends to dampen moves. **Negative GEX** — '
        'dealers sell drops and buy rallies, which tends to amplify moves. The **flip level** is where it changes sign.'
    )
    base = str(default_symbol or '').upper().replace('-', '/').split('/')[0]
    c1, c2 = st.columns([1, 2])
    currency = c1.radio('Asset', GEX_CURRENCIES, index=1 if base.startswith('ETH') else 0, horizontal=True,
                        key='gex_currency')
    next_only = c2.toggle('Next expiry only', key='gex_next_only',
                          help='Only the options that expire next (Deribit expiries are 08:00 UTC).')
    try:
        options, loaded_at = load_options(currency)
    except Exception as exc:
        st.warning(f'Could not load Deribit options: {exc}')
        return None
    full = gex_snapshot(options, currency, loaded_at)
    if full is None:
        st.warning(f'Deribit returned no live {currency} options.')
        return None
    try:
        record_snapshot(full, 'dashboard')  # log the full-book snapshot, not the filtered view
    except Exception:
        pass
    snap = full
    if next_only:
        snap = gex_snapshot(options[options['expiry'] == full['next_expiry']], currency, loaded_at, full['spot'])

    m1, m2, m3, m4 = st.columns(4)
    plain = {'delta_color': 'off', 'delta_arrow': 'off'}
    m1.metric('Net GEX', _money(snap['net_gex']), 'per 1% move', **plain,
              help='Dollars of the coin dealers would trade to re-hedge after a 1% move.')
    positive = snap['regime'] == 'POSITIVE'
    m2.metric('Regime', 'Positive' if positive else 'Negative', 'dampening' if positive else 'amplifying', **plain)
    flip = snap['flip']
    m3.metric('Gamma flip', f'{flip:,.0f}' if flip else 'none',
              f"{(flip / snap['spot'] - 1):+.1%} from spot" if flip else 'within ±20%', **plain,
              help='Price where net GEX changes sign.')
    m4.metric('Spot', f"{snap['spot']:,.0f}")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric('Call wall', f"{snap['call_wall']:,.0f}" if snap['call_wall'] else '—', help='Strike with the largest +GEX.')
    w2.metric('Put wall', f"{snap['put_wall']:,.0f}" if snap['put_wall'] else '—', help='Strike with the largest −GEX.')
    w3.metric('Expiring ≤ 7d', f"{full['share_7d']:.0%}", help='Share of all GEX that expires within 7 days.')
    w4.metric('Next expiry', f"{full['next_expiry']:%d %b}", f"08:00 UTC · {_money(full['next_expiry_gex'])}", **plain)

    st.plotly_chart(gex_chart(snap, currency), width='stretch')
    st.warning(
        '**Untested — read it as context, not a signal.** GEX assumes dealers are short calls and long puts. In crypto, '
        'funds often *sell* calls for yield, which can leave dealers long calls and flip the call side. It covers '
        'Deribit only (most crypto options, but not OKX, Bybit, Binance or CME), and it can change sharply after each '
        'Friday 08:00 UTC expiry.'
    )

    history = load_gex_history()
    mine = history[history['Currency'] == currency] if not history.empty else history
    with st.expander(f'Snapshot log for testing — {len(mine)} {currency} snapshot(s) saved'):
        st.caption(
            'Deribit has no free historical open interest, so one snapshot per currency per hour is saved to '
            'gex_history.csv whenever this panel is open or the alert runner is running (alert runner.bat). '
            'After roughly 3–6 months it can be tested: does negative GEX come before bigger realised moves, and do '
            'the walls hold?'
        )
        if not mine.empty:
            st.dataframe(mine.tail(48).iloc[::-1], hide_index=True, width='stretch')
    st.caption(f'Deribit book summary loaded {loaded_at:%Y-%m-%d %H:%M} UTC (cached 5 min). Gamma from Black-Scholes '
               'with each option\'s mark IV; open interest is in coins (1 contract = 1 coin).')
    return snap

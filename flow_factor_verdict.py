"""🧮 Flow Factor Verdict — top Binance crypto perpetuals.

Combines every element of "Demand & Supply Analysis — Highest Flow Z Drop / Rise" (Flow,
Momentum, Volatility and Trend Z changes and levels, price change, and the 60/25/15 flow /
momentum / volatility contributions) into one ridge-regression forecast per coin, then gives a
verdict ONLY when that coin's own walk-forward record says the forecast made money out of sample.

    forecast  ŷ_t = β · x_t           (x = standardised factors + interactions, ridge-fitted on
                                        that coin's history before t; refit every REFIT_EVERY bars)
    trade     at a Flow Z rise/drop event (largest ΔF of the trailing 24 bars), side = sign(ŷ)
              when |ŷ| exceeds the round-trip cost in ATR; hold H bars; no overlapping trades
    verdict   LONG / SHORT only if the out-of-sample trades number ≥ MIN_TRADES and their mean
              net return has t ≥ z_(1 − 0.05/N) (Bonferroni for N coins tested at once) and the
              current forecast clears costs. Everything else is NO EDGE.

Pooled research (40 perps, 1h, Jan 2024 – Nov 2026, 4/12/24-bar horizons) found no edge in these
factors, so expect NO EDGE almost everywhere. That is the honest answer, not a bug.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import norm

from qqq_nq_signals import demand_supply_score, historical_z_components

EVENT_WINDOW = 24          # "highest Flow Z rise/drop" over the trailing 24 bars (causal stand-in for a day)
WARMUP = 1000              # bars before the first out-of-sample forecast
REFIT_EVERY = 250          # bars between walk-forward refits
RIDGE = 10.0
MIN_TRADES = 30
FAMILY_ALPHA = 0.05
SCAN_KEY = 'ffv_scan'
TIMEFRAME_DELTA = {'1h': pd.Timedelta(hours=1), '4h': pd.Timedelta(hours=4)}
BASE_FEATURES = ['Flow Z', 'Momentum Z', 'Volatility Z', 'Trend Z', 'Flow Z Change', 'Momentum Z Change',
                 'Volatility Z Change', 'Trend Z Change', 'Price Change (ATR)', 'Flow Contribution',
                 'Momentum Contribution', 'Volatility Contribution']
INTERACTIONS = {
    'Flow Δ × Price Δ': ('Flow Z Change', 'tanh Price'),
    'Flow Δ × Volatility Z': ('Flow Z Change', 'Volatility Z'),
    'Momentum Δ × Flow Δ': ('Momentum Z Change', 'Flow Z Change'),
    'Flow Z × Trend Z': ('Flow Z', 'Trend Z'),
}
FEATURES = BASE_FEATURES + list(INTERACTIONS) + ['Rise Event', 'Drop Event']
TABLE_COLUMNS = [
    'Rank', 'Symbol', 'Verdict', 'Reason', 'Forecast (ATR)', 'Latest Event', 'Flow Z Change',
    'Momentum Z Change', 'Volatility Z Change', 'Price Change (ATR)', 'Flow Contribution',
    'Momentum Contribution', 'Volatility Contribution', 'Direction Score', 'OOS Trades', 'OOS Hit Rate',
    'OOS Net/Trade %', 'OOS t', 'Score Rule Net/Trade %', 'Close',
]


# ----------------------------------------------------------------------------- factors

def completed_candles(history, timeframe, now=None):
    """Drop the still-forming last candle (index is the candle open, UTC)."""
    if history is None or history.empty:
        return history
    data = history.copy()
    data.index = pd.to_datetime(data.index, utc=True)
    data = data[~data.index.duplicated(keep='last')].sort_index()
    now = now or pd.Timestamp.now(tz='UTC')
    return data.loc[data.index + TIMEFRAME_DELTA[timeframe] <= now]


def factor_frame(history, horizon):
    """Every Highest-Flow-Z element per bar, plus the forward return the forecast is judged on."""
    required = {'open', 'high', 'low', 'close', 'volume', 'momentum', 'atr14', 'z_score'}
    if history is None or history.empty or not required.issubset(history.columns):
        return pd.DataFrame()
    data = history
    comps = historical_z_components(data)
    changes = comps.diff()
    close, atr = data['close'].astype(float), data['atr14'].astype(float).replace(0, np.nan)
    price_change = close.diff()
    f = pd.DataFrame(index=data.index)
    f['Flow Z'], f['Momentum Z'] = comps['Flow (z)'], comps['Momentum (z)']
    f['Volatility Z'], f['Trend Z'] = comps['Volatility (z)'], comps['Trend (z)']
    f['Flow Z Change'], f['Momentum Z Change'] = changes['Flow (z)'], changes['Momentum (z)']
    f['Volatility Z Change'], f['Trend Z Change'] = changes['Volatility (z)'], changes['Trend (z)']
    f['Price Change'] = price_change
    f['Price Change (ATR)'] = price_change / atr
    # Same contributions as highest_flow_zone (fractions here; the table shows them x100).
    f['Flow Contribution'] = .60 * np.tanh(f['Flow Z Change'])
    f['Momentum Contribution'] = .25 * np.tanh(f['Momentum Z Change'])
    vol_up = f['Volatility Z Change'] > 0
    f['Volatility Contribution'] = .15 * np.where(
        vol_up, np.sign(price_change) * np.tanh(f['Volatility Z Change'].abs()), 0.0)
    f['Direction Score'] = demand_supply_score(
        f['Flow Z Change'], f['Momentum Z Change'], f['Volatility Z Change'], price_change)
    flow_change = f['Flow Z Change']
    f['Rise Event'] = ((flow_change == flow_change.rolling(EVENT_WINDOW).max()) & (flow_change > 0)).astype(float)
    f['Drop Event'] = ((flow_change == flow_change.rolling(EVENT_WINDOW).min()) & (flow_change < 0)).astype(float)
    f['tanh Price'] = np.tanh(f['Price Change (ATR)'])
    for name, (a, b) in INTERACTIONS.items():
        f[name] = f[a] * f[b]
    f['Close'], f['ATR'] = close, atr
    f['Forward Return'] = close.shift(-horizon) / close - 1
    f['Forward ATR'] = (close.shift(-horizon) - close) / atr
    return f.replace([np.inf, -np.inf], np.nan)


# ----------------------------------------------------------------------------- model

def ridge_fit(X, y, lam=RIDGE):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    Z = (X - mu) / sd
    beta = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ (y - y.mean()))
    return {'mu': mu, 'sd': sd, 'beta': beta, 'intercept': float(y.mean())}


def ridge_predict(model, X):
    return ((X - model['mu']) / model['sd']) @ model['beta'] + model['intercept']


def trade_stats(net):
    net = np.asarray(net, dtype=float)
    n = len(net)
    if n < 2:
        return {'n': n, 'hit': np.nan, 'mean': np.nan, 't': np.nan}
    sd = net.std(ddof=1)
    return {'n': n, 'hit': float((net > 0).mean()), 'mean': float(net.mean()),
            't': float(net.mean() / (sd / np.sqrt(n))) if sd > 0 else np.nan}


def event_trades(f, sides, horizon, cost):
    """Non-overlapping trades at Flow Z events. sides: +1 / -1 / 0 per bar. Returns (positions, net returns)."""
    events = (f['Rise Event'] + f['Drop Event']).to_numpy() > 0
    fwd = f['Forward Return'].to_numpy()
    positions, nets, last = [], [], -10 ** 9
    for i in np.flatnonzero(events & (sides != 0) & np.isfinite(fwd)):
        if i - last < horizon:
            continue
        last = i
        positions.append(i)
        nets.append(sides[i] * fwd[i] - cost)
    return np.array(positions, dtype=int), np.array(nets, dtype=float)


def walk_forward(f, horizon, cost, warmup=WARMUP, refit_every=REFIT_EVERY):
    """Out-of-sample forecasts and trades for one coin, plus the model fitted on all history."""
    valid = f[FEATURES].notna().all(axis=1).to_numpy()
    X = f[FEATURES].fillna(0).clip(-6, 6).to_numpy()
    y = f['Forward ATR'].clip(-10, 10).to_numpy()
    labelled = valid & np.isfinite(y)
    n = len(f)
    pred = np.full(n, np.nan)
    for start in range(warmup, n, refit_every):
        train = labelled.copy()
        train[max(0, start - horizon):] = False   # purge: labels that look past `start`
        if train.sum() < 200:
            continue
        model = ridge_fit(X[train], y[train])
        block = np.arange(start, min(n, start + refit_every))
        block = block[valid[block]]
        pred[block] = ridge_predict(model, X[block])
    cost_atr = (cost * f['Close'] / f['ATR']).to_numpy()
    sides = np.where(np.isfinite(pred) & (np.abs(pred) > cost_atr), np.sign(pred), 0.0)
    model_pos, model_net = event_trades(f, sides, horizon, cost)
    oos = np.isfinite(pred)
    score = f['Direction Score'].to_numpy()
    score_sides = np.where(oos & (score >= 10), 1.0, np.where(oos & (score <= -10), -1.0, 0.0))
    _, score_net = event_trades(f, score_sides, horizon, cost)
    final = ridge_fit(X[labelled], y[labelled]) if labelled.sum() >= 200 else None
    current = float(ridge_predict(final, X[-1:])[0]) if final is not None and valid[-1] else np.nan
    return {
        'pred': pred, 'trades': trade_stats(model_net), 'trade_positions': model_pos, 'trade_net': model_net,
        'score_trades': trade_stats(score_net), 'model': final, 'forecast': current,
        'cost_atr_now': float(cost_atr[-1]) if len(cost_atr) else np.nan,
    }


def required_t(coins, alpha=FAMILY_ALPHA):
    """One-sided Bonferroni threshold: any single coin passing by luck stays below alpha."""
    return float(norm.ppf(1 - alpha / max(int(coins), 1)))


def verdict(result, t_needed, min_trades=MIN_TRADES):
    trades, forecast = result['trades'], result['forecast']
    if not np.isfinite(forecast):
        return 'NO EDGE', 'Not enough clean history for a forecast'
    if trades['n'] < min_trades:
        return 'NO EDGE', f"Only {trades['n']} out-of-sample trades (need {min_trades})"
    if not np.isfinite(trades['t']) or trades['t'] < t_needed:
        return 'NO EDGE', f"Out-of-sample t = {trades['t']:+.2f}, needs ≥ {t_needed:.2f}"
    if abs(forecast) <= result['cost_atr_now']:
        return 'NO EDGE', 'Edge proven, but the current forecast does not clear costs'
    return ('LONG' if forecast > 0 else 'SHORT'), (
        f"Out-of-sample t = {trades['t']:+.2f} over {trades['n']} trades; forecast {forecast:+.2f} ATR")


def highest_flow_events(f, window=EVENT_WINDOW):
    """Highest Flow Z rise and drop of the trailing window, with every element."""
    tail = f.tail(window).dropna(subset=['Flow Z Change'])
    out = {}
    if (tail['Flow Z Change'] > 0).any():
        out['Rise'] = tail.loc[tail['Flow Z Change'].idxmax()]
    if (tail['Flow Z Change'] < 0).any():
        out['Drop'] = tail.loc[tail['Flow Z Change'].idxmin()]
    return out


def analyze_coin(symbol, history, timeframe, horizon, cost, now=None):
    data = completed_candles(history, timeframe, now)
    f = factor_frame(data, horizon)
    if f.empty or len(f) < WARMUP + 200:
        return None
    result = walk_forward(f, horizon, cost)
    last = f.iloc[-1]
    latest_event = 'Rise' if last['Rise Event'] else 'Drop' if last['Drop Event'] else '—'
    return {'symbol': symbol, 'factors': f, 'result': result, 'latest': last, 'latest_event': latest_event}


def scan(symbols, fetch, timeframe, horizon, candles, cost, workers=6, progress=None):
    """Fetch and analyse every symbol. Returns (table, failed, details)."""
    histories, failed = {}, []

    def load(symbol):
        try:
            return symbol, fetch(symbol, timeframe=timeframe, silent=True, limit=candles)
        except Exception:
            return symbol, None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(load, s) for s in symbols]
        for done, future in enumerate(as_completed(futures), start=1):
            symbol, frame = future.result()
            if frame is None or len(frame) < WARMUP + 200:
                failed.append(symbol)
            else:
                histories[symbol] = frame
            if progress:
                progress(done / len(symbols) * 0.6, f'Fetched {symbol}')

    details = {}
    for i, symbol in enumerate(symbols):
        if symbol in histories:
            try:
                coin = analyze_coin(symbol, histories[symbol], timeframe, horizon, cost)
            except Exception:
                coin = None
            if coin is None:
                failed.append(symbol)
            else:
                details[symbol] = coin
        if progress:
            progress(0.6 + 0.4 * (i + 1) / len(symbols), f'Modelled {symbol}')

    t_needed = required_t(len(details))
    rows = []
    for rank, symbol in enumerate(symbols, start=1):
        coin = details.get(symbol)
        if coin is None:
            continue
        res, last = coin['result'], coin['latest']
        call, reason = verdict(res, t_needed)
        coin['verdict'], coin['reason'] = call, reason
        rows.append({
            'Rank': rank, 'Symbol': symbol, 'Verdict': call, 'Reason': reason,
            'Forecast (ATR)': res['forecast'], 'Latest Event': coin['latest_event'],
            'Flow Z Change': last['Flow Z Change'], 'Momentum Z Change': last['Momentum Z Change'],
            'Volatility Z Change': last['Volatility Z Change'], 'Price Change (ATR)': last['Price Change (ATR)'],
            'Flow Contribution': last['Flow Contribution'] * 100,
            'Momentum Contribution': last['Momentum Contribution'] * 100,
            'Volatility Contribution': last['Volatility Contribution'] * 100,
            'Direction Score': last['Direction Score'],
            'OOS Trades': res['trades']['n'], 'OOS Hit Rate': res['trades']['hit'],
            'OOS Net/Trade %': res['trades']['mean'] * 100 if np.isfinite(res['trades']['mean']) else np.nan,
            'OOS t': res['trades']['t'],
            'Score Rule Net/Trade %': (res['score_trades']['mean'] * 100
                                       if np.isfinite(res['score_trades']['mean']) else np.nan),
            'Close': last['Close'],
        })
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    order = table['Verdict'].map({'LONG': 0, 'SHORT': 0}).fillna(1)
    table = table.assign(_o=order).sort_values(['_o', 'Rank'], kind='stable').drop(columns='_o').reset_index(drop=True)
    return table, sorted(set(failed), key=symbols.index), details, t_needed


# ----------------------------------------------------------------------------- UI

FORMULAS = [
    ('Factors at each closed candle',
     r'x_t = \big(F, M, V, T,\ \Delta F, \Delta M, \Delta V, \Delta T,\ \tfrac{\Delta P}{ATR},\ '
     r'c_F, c_M, c_V,\ \Delta F\tanh\tfrac{\Delta P}{ATR},\ \Delta F\,V,\ \Delta M\,\Delta F,\ F\,T,\ '
     r'\mathbb{1}_{rise}, \mathbb{1}_{drop}\big)'),
    ('Contributions (as in Highest Flow Z Drop / Rise)',
     r'c_F = .60\tanh\Delta F,\quad c_M = .25\tanh\Delta M,\quad '
     r'c_V = .15\,\operatorname{sign}(\Delta P)\tanh|\Delta V|\,\mathbb{1}_{\Delta V>0}'),
    ('Forecast of the next H candles, in ATR (ridge, refit walk-forward)',
     r'\hat y_t = \beta^\top \tfrac{x_t-\mu}{\sigma},\qquad '
     r'\beta = \arg\min_\beta \textstyle\sum_{s<t-H}(y_s-\beta^\top z_s)^2 + \lambda\lVert\beta\rVert^2'),
    ('Verdict',
     r'\text{LONG/SHORT} \iff n_{oos}\ge 30,\ \ t_{oos} \ge \Phi^{-1}\!\big(1-\tfrac{0.05}{N}\big),\ \ '
     r'|\hat y_t| > \tfrac{\text{cost}\cdot P_t}{ATR_t}'),
]
VERDICT_STYLE = {'LONG': 'color: #16a34a; font-weight: 700', 'SHORT': 'color: #dc2626; font-weight: 700',
                 'NO EDGE': 'color: #9ca3af'}


def _event_card(label, row):
    st.markdown(f'**Highest Flow Z {label}** — {row.name:%Y-%m-%d %H:%M} UTC')
    cols = st.columns(4)
    cols[0].metric('Flow Z Change', f"{row['Flow Z Change']:+.3f}")
    cols[1].metric('Momentum Z Change', f"{row['Momentum Z Change']:+.3f}")
    cols[2].metric('Volatility Z Change', f"{row['Volatility Z Change']:+.3f}")
    cols[3].metric('Price Change', f"{row['Price Change']:+,.6g}", f"{row['Price Change (ATR)']:+.2f} ATR",
                   delta_color='off', delta_arrow='off')
    cols = st.columns(4)
    cols[0].metric('Flow Contribution', f"{row['Flow Contribution'] * 100:+.1f}")
    cols[1].metric('Momentum Contribution', f"{row['Momentum Contribution'] * 100:+.1f}")
    cols[2].metric('Volatility Contribution', f"{row['Volatility Contribution'] * 100:+.1f}")
    cols[3].metric('Direction Score', f"{row['Direction Score']:+.1f}")


def _weights_chart(model):
    beta = pd.Series(model['beta'], index=FEATURES).sort_values()
    fig = go.Figure(go.Bar(x=beta.values, y=beta.index, orientation='h',
                           marker_color=np.where(beta.values >= 0, '#16a34a', '#dc2626')))
    fig.update_layout(title='Model weights (standardised; ATR per 1σ of each factor)', height=460,
                      margin={'l': 10, 'r': 10, 't': 40, 'b': 10})
    return fig


def _equity_chart(coin):
    res, f = coin['result'], coin['factors']
    if not len(res['trade_net']):
        return None
    curve = pd.Series(np.cumsum(res['trade_net']) * 100, index=f.index[res['trade_positions']])
    fig = go.Figure(go.Scatter(x=curve.index, y=curve.values, mode='lines', line={'color': '#7c3aed'}))
    fig.add_hline(y=0, line_dash='dash', line_color='grey')
    fig.update_layout(title='Out-of-sample model trades: cumulative net return (%)', height=300,
                      margin={'l': 10, 'r': 10, 't': 40, 'b': 10})
    return fig


def render_flow_factor_verdict(top_symbols, fetch):
    """top_symbols(count) -> list of 'BASE/USDT'; fetch(symbol, timeframe, silent, limit) -> candles."""
    st.subheader('🧮 Flow Factor Verdict — Top Crypto Perpetuals')
    st.caption(
        'Every element of Demand & Supply Analysis — Highest Flow Z Drop / Rise, combined into one walk-forward '
        'forecast per coin. A coin only gets LONG or SHORT when its own out-of-sample record proves the forecast '
        'made money after costs, at a bar high enough for the number of coins tested.'
    )
    st.warning(
        '**Expect NO EDGE on most or all coins.** A pooled test of these factors on the top 40 perpetuals '
        '(hourly, Jan 2024 – Nov 2026, holding 4/12/24 hours, 0.10% costs) found no out-of-sample edge: the '
        'forecast\'s rank correlation with the next move was ≈ 0, and the current Direction Score\'s was slightly '
        'negative. This tab tests each coin again, live, so you can see that for yourself rather than trust a label.'
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    count = int(c1.number_input('Perpetuals', 5, 60, 30, 5, key='ffv_count'))
    timeframe = c2.selectbox('Timeframe', list(TIMEFRAME_DELTA), index=0, key='ffv_tf')
    horizon = int(c3.selectbox('Hold (candles)', [4, 12, 24], index=1, key='ffv_h'))
    candles = int(c4.number_input('History (candles)', 1500, 6000, 3000, 500, key='ffv_candles'))
    cost_bps = float(c5.number_input('Round-trip cost (bps)', 0.0, 50.0, 10.0, 1.0, key='ffv_cost'))

    if st.button(f'Run Flow Factor Verdict on the top {count} perpetuals', key='ffv_run'):
        try:
            symbols = top_symbols(count)
        except Exception as exc:
            st.error(f'Could not load the Binance perpetual list: {exc}')
            symbols = []
        if symbols:
            bar = st.progress(0.0, text='Starting…')
            table, failed, details, t_needed = scan(
                symbols, fetch, timeframe, horizon, candles, cost_bps / 1e4,
                progress=lambda share, text: bar.progress(min(share, 1.0), text=text))
            bar.empty()
            st.session_state[SCAN_KEY] = {
                'table': table, 'failed': failed, 'details': details, 't_needed': t_needed,
                'settings': (timeframe, horizon, candles, cost_bps), 'at': pd.Timestamp.now(tz='UTC'),
            }

    run = st.session_state.get(SCAN_KEY)
    if not run:
        st.info('Press the button to run the scan. About 1–2 minutes for 30 perpetuals.')
        with st.expander('How the verdict is calculated'):
            for title, formula in FORMULAS:
                st.markdown(f'**{title}**')
                st.latex(formula)
        return
    table, details = run['table'], run['details']
    timeframe, horizon, candles, cost_bps = run['settings']
    st.caption(
        f"Ran {run['at']:%Y-%m-%d %H:%M} UTC on {len(table)} perpetuals · {timeframe} candles · hold {horizon} · "
        f"{candles} candles of history · {cost_bps:.0f} bps cost · edge bar t ≥ {run['t_needed']:.2f}"
        + (f" · skipped {', '.join(run['failed'])}" if run['failed'] else '')
    )
    if table.empty:
        st.warning('No perpetual returned enough history.')
        return

    counts = table['Verdict'].value_counts()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('LONG', int(counts.get('LONG', 0)))
    m2.metric('SHORT', int(counts.get('SHORT', 0)))
    m3.metric('NO EDGE', int(counts.get('NO EDGE', 0)))
    m4.metric('Positive OOS', int((table['OOS Net/Trade %'] > 0).sum()), f'of {len(table)} coins',
              delta_color='off', delta_arrow='off',
              help='Coins whose out-of-sample model trades made money after costs. With no real edge, costs push '
                   'most coins slightly negative, so well under half is normal; a positive record still needs '
                   'the t bar to count.')

    st.dataframe(
        table.style.format({
            'Forecast (ATR)': '{:+.3f}', 'Flow Z Change': '{:+.2f}', 'Momentum Z Change': '{:+.2f}',
            'Volatility Z Change': '{:+.2f}', 'Price Change (ATR)': '{:+.2f}', 'Flow Contribution': '{:+.1f}',
            'Momentum Contribution': '{:+.1f}', 'Volatility Contribution': '{:+.1f}', 'Direction Score': '{:+.1f}',
            'OOS Hit Rate': '{:.0%}', 'OOS Net/Trade %': '{:+.3f}', 'OOS t': '{:+.2f}',
            'Score Rule Net/Trade %': '{:+.3f}', 'Close': '{:,.6g}',
        }, na_rep='—').map(lambda v: VERDICT_STYLE.get(v, ''), subset=['Verdict']),
        width='stretch', hide_index=True,
    )
    st.caption(
        'Factor columns are for the latest closed candle. Forecast: expected move over the hold, in ATR. '
        'OOS columns: the model\'s own out-of-sample trades at Flow Z events on this coin. Score Rule: the existing '
        'Direction Score (≥ +10 long, ≤ −10 short) traded the same way, for comparison.'
    )
    st.download_button('Download table (CSV)', table.to_csv(index=False).encode(),
                       file_name=f"flow_factor_verdict_{run['at']:%Y%m%d_%H%M}.csv", mime='text/csv',
                       key='ffv_download')

    pick = st.selectbox('Coin detail', table['Symbol'], key='ffv_pick')
    coin = details.get(pick)
    if coin:
        st.markdown(f"### {pick}: **{coin['verdict']}** — {coin['reason']}")
        events = highest_flow_events(coin['factors'])
        if not events:
            st.info('No Flow Z rise or drop in the last 24 candles.')
        for label in ('Drop', 'Rise'):
            if label in events:
                _event_card(label, events[label])
        left, right = st.columns(2)
        if coin['result']['model'] is not None:
            left.plotly_chart(_weights_chart(coin['result']['model']), width='stretch')
        equity = _equity_chart(coin)
        if equity is not None:
            right.plotly_chart(equity, width='stretch')
        trades, score = coin['result']['trades'], coin['result']['score_trades']
        right.markdown(
            f"Model OOS: **{trades['n']}** trades, hit **{trades['hit']:.0%}**, net **{trades['mean'] * 100:+.3f}%** "
            f"per trade, t **{trades['t']:+.2f}**  \nDirection Score rule OOS: **{score['n']}** trades, net "
            f"**{score['mean'] * 100:+.3f}%** per trade, t **{score['t']:+.2f}**"
            if trades['n'] > 1 and score['n'] > 1 else 'Too few out-of-sample trades to summarise.'
        )
    with st.expander('How the verdict is calculated'):
        for title, formula in FORMULAS:
            st.markdown(f'**{title}**')
            st.latex(formula)
        st.caption(
            f'Walk-forward: the first {WARMUP} candles only train; the model is refit every {REFIT_EVERY} candles '
            f'on everything before, leaving out the last {horizon} candles whose outcome was not yet known. '
            f'Events are the largest Flow Z rise or drop of the trailing {EVENT_WINDOW} candles. '
            'Bonferroni: with N coins tested, a coin must reach t ≥ Φ⁻¹(1 − 0.05/N) so that the chance of any coin '
            'passing by pure luck stays under 5%.'
        )

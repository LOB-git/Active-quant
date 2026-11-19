"""Quantitative edge layer for the Top Derivatives Overview.

Every function is pure (candles in, numbers out) so the maths can be tested
without Streamlit or Binance. Candles need open/high/low/close plus the
overview indicators z_score, cmf and atr14.

Nothing here predicts the future. The layer only reports a trade when the
same rule had a positive, cost-adjusted expectancy on the loaded history and
the signal's rank correlation with the next bar's return was positive.
"""

import math

import numpy as np
import pandas as pd


TREND_LOOKBACK = 48      # Bars in the rolling drift t-statistic.
FLOW_LOOKBACK = 20       # Bars in the CMF flow z-score (same as the overview).
VR_HORIZON = 8           # q in the Lo-MacKinlay variance ratio.
MIN_TRADES = 10          # Fewer historical trades than this is not evidence.
MAX_LEVERAGE = 3.0       # Hard cap on the notional the plan will suggest.
Z_95 = -1.6448536269514722


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def log_returns(close):
    close = pd.to_numeric(pd.Series(close), errors='coerce')
    return np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()


def yang_zhang_variance(candles):
    """Per-bar Yang-Zhang variance: overnight + k·open-close + (1-k)·Rogers-Satchell."""
    o, h, l, c = (pd.to_numeric(candles[name], errors='coerce') for name in ('open', 'high', 'low', 'close'))
    frame = pd.DataFrame({
        'gap': np.log(o / c.shift(1)), 'body': np.log(c / o),
        'up': np.log(h / o), 'down': np.log(l / o),
    }).replace([np.inf, -np.inf], np.nan).dropna()
    n = len(frame)
    if n < 3:
        return np.nan
    rogers_satchell = (frame['up'] * (frame['up'] - frame['body'])
                       + frame['down'] * (frame['down'] - frame['body'])).mean()
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    return float(frame['gap'].var() + k * frame['body'].var() + (1 - k) * rogers_satchell)


def hurst_exponent(close, max_lag=64):
    """Slope of log std(x[t+τ] - x[t]) on log τ, x = log price. 0.5 = random walk."""
    x = np.log(pd.to_numeric(pd.Series(close), errors='coerce').dropna().to_numpy(dtype=float))
    if len(x) < 4 * 8:
        return np.nan
    lags = np.unique(np.geomspace(2, min(max_lag, len(x) // 4), 12).astype(int))
    spreads = np.array([np.std(x[lag:] - x[:-lag]) for lag in lags])
    valid = spreads > 0
    if valid.sum() < 3:
        return np.nan
    return float(np.polyfit(np.log(lags[valid]), np.log(spreads[valid]), 1)[0])


def variance_ratio(returns, q=VR_HORIZON):
    """Lo-MacKinlay VR(q) with overlapping sums and its homoskedastic z-statistic."""
    r = pd.Series(returns, dtype=float).dropna().to_numpy()
    n = len(r)
    if n <= 2 * q:
        return np.nan, np.nan
    mu = r.mean()
    var_1 = ((r - mu) ** 2).sum() / (n - 1)
    if var_1 <= 0:
        return np.nan, np.nan
    sums = np.convolve(r, np.ones(q), mode='valid')
    m = q * (n - q + 1) * (1 - q / n)
    ratio = ((sums - q * mu) ** 2).sum() / m / var_1
    z = (ratio - 1) / math.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * n))
    return float(ratio), float(z)


def ou_half_life(close):
    """Ornstein-Uhlenbeck half-life in bars from Δx = α + β·x[t-1]; inf when β ≥ 0."""
    x = np.log(pd.to_numeric(pd.Series(close), errors='coerce').dropna())
    lagged, delta = x.shift(1).iloc[1:], x.diff().iloc[1:]
    variance = lagged.var()
    if len(lagged) < 3 or not variance:
        return np.inf
    beta = lagged.cov(delta) / variance
    return float(-math.log(2) / math.log(1 + beta)) if -1 < beta < 0 else np.inf


def probabilistic_sharpe(returns):
    """Bailey & López de Prado PSR: probability the true per-bar Sharpe exceeds 0."""
    r = pd.Series(returns, dtype=float).dropna()
    n, sigma = len(r), r.std()
    if n < 10 or not sigma:
        return np.nan
    sharpe = r.mean() / sigma
    skew, kurt = float(r.skew()), float(r.kurt()) + 3.0
    denominator = 1 - skew * sharpe + (kurt - 1) / 4 * sharpe ** 2
    if denominator <= 0:
        return np.nan
    return normal_cdf(sharpe * math.sqrt(n - 1) / math.sqrt(denominator))


def tail_risk(returns):
    """95% Cornish-Fisher VaR and historical CVaR, as positive per-bar loss fractions."""
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 20:
        return np.nan, np.nan
    s, k, z = float(r.skew()), float(r.kurt()), Z_95
    z_cf = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
    var = -(r.mean() + z_cf * r.std())
    tail = r[r <= r.quantile(0.05)]
    return float(var), float(-tail.mean())


def clean_wicks(candles, max_wick=5.0, window=50):
    """Cap bad-print wicks and rebuild ATR(14) from the cleaned bars.

    Yahoo's extended-hours bars sometimes carry single prints far outside the real range
    (a 15% hourly range on SPY). A wick longer than max_wick times the rolling median
    bar range is cut back to that length. Genuine volatile periods raise the median
    with them, so real moves survive.
    """
    frame = candles.copy()
    o, h, l, c = (pd.to_numeric(frame[name], errors='coerce') for name in ('open', 'high', 'low', 'close'))
    relative_range = (h - l) / c
    scale = relative_range.rolling(window, min_periods=10).median()
    scale = scale.fillna(relative_range.median())
    cap = max_wick * scale * c
    body_top, body_bottom = np.maximum(o, c), np.minimum(o, c)
    frame['high'] = np.minimum(h, body_top + cap).where(cap.notna(), h)
    frame['low'] = np.maximum(l, body_bottom - cap).where(cap.notna(), l)
    true_range = pd.concat([
        frame['high'] - frame['low'], (frame['high'] - c.shift(1)).abs(), (frame['low'] - c.shift(1)).abs(),
    ], axis=1).max(axis=1)
    frame['atr14'] = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    return frame


def annualized_volatility(yz_variance, returns, bars_per_year, max_ratio=1.5):
    """Yang-Zhang volatility, or close-to-close when noisy wicks inflate Yang-Zhang.

    Both estimate the same quantity, so Yang-Zhang far above close-to-close means the
    high/low data is unreliable (thin extended-hours ETF bars), not that risk is higher.
    """
    if not bars_per_year or not bars_per_year > 0:
        return {'Volatility (ann.)': np.nan, 'Vol Method': 'n/a'}
    close_to_close = float(pd.Series(returns).std()) * math.sqrt(bars_per_year)
    yang_zhang = math.sqrt(yz_variance * bars_per_year) if yz_variance > 0 else np.nan
    if np.isfinite(yang_zhang) and (not close_to_close > 0 or yang_zhang <= max_ratio * close_to_close):
        return {'Volatility (ann.)': yang_zhang, 'Vol Method': 'Yang-Zhang'}
    return {'Volatility (ann.)': close_to_close, 'Vol Method': 'Close-to-close (noisy wicks)'}


def regime_weight(hurst):
    """Trend weight a ∈ [0, 1]: 0 at H ≤ 0.45 (mean reversion), 1 at H ≥ 0.55 (trend)."""
    if hurst is None or pd.isna(hurst):
        return 0.5
    return float(np.clip((hurst - 0.45) / 0.10, 0.0, 1.0))


def signal_series(candles, trend_weight):
    """Q[t] = 0.7·[a·tanh(t_drift/2) − (1−a)·tanh(Z/2)] + 0.3·tanh(FlowZ/2), in [−1, 1].

    Each input uses only data up to and including bar t.
    """
    close = pd.to_numeric(candles['close'], errors='coerce')
    r = np.log(close / close.shift(1))
    drift_t = (r.rolling(TREND_LOOKBACK).mean()
               / (r.rolling(TREND_LOOKBACK).std() / math.sqrt(TREND_LOOKBACK)))
    cmf = pd.to_numeric(candles['cmf'], errors='coerce')
    flow_z = (cmf - cmf.rolling(FLOW_LOOKBACK).mean()) / (cmf.rolling(FLOW_LOOKBACK).std() + 1e-9)
    z = pd.to_numeric(candles['z_score'], errors='coerce')
    # Zero-volume series (^VIX, DXY) have no money flow: treat it as neutral, not missing.
    flow_z = flow_z.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    trend, reversion, flow = np.tanh(drift_t / 2), -np.tanh(z / 2), np.tanh(flow_z / 2)
    q = 0.7 * (trend_weight * trend + (1 - trend_weight) * reversion) + 0.3 * flow
    return q.replace([np.inf, -np.inf], np.nan)


def information_coefficient(signal, close):
    """Spearman rank IC of Q[t] against the next bar's log return, with its t-statistic."""
    close = pd.to_numeric(close, errors='coerce')
    forward = np.log(close.shift(-1) / close)
    pair = pd.DataFrame({'q': signal, 'fwd': forward}).dropna()
    n = len(pair)
    if n < 30:
        return np.nan, np.nan
    ic = float(pair['q'].rank().corr(pair['fwd'].rank()))
    if pd.isna(ic) or abs(ic) >= 1:
        return ic, np.nan
    return ic, ic * math.sqrt((n - 2) / (1 - ic ** 2))


def barrier_backtest(candles, signal, threshold, atr_multiple, reward_risk, max_hold, fee_bps):
    """Triple-barrier replay of the live rule. Returns each trade's net result in R.

    Entry at bar t's close when |Q[t]| ≥ threshold, stop at atr_multiple·ATR, target at
    reward_risk times that, exit at max_hold bars otherwise. Stop counts first when
    one candle touches both. Trades never overlap. Fees are round-trip.
    """
    high = pd.to_numeric(candles['high'], errors='coerce').to_numpy(dtype=float)
    low = pd.to_numeric(candles['low'], errors='coerce').to_numpy(dtype=float)
    close = pd.to_numeric(candles['close'], errors='coerce').to_numpy(dtype=float)
    atr = pd.to_numeric(candles['atr14'], errors='coerce').to_numpy(dtype=float)
    q = pd.Series(signal).to_numpy(dtype=float)
    results, t, last = [], 0, len(close) - 1
    while t < last:
        risk = atr_multiple * atr[t]
        if not (np.isfinite(q[t]) and abs(q[t]) >= threshold and np.isfinite(risk) and risk > 0):
            t += 1
            continue
        side, entry = (1.0 if q[t] > 0 else -1.0), close[t]
        stop, target = entry - side * risk, entry + side * reward_risk * risk
        outcome, exit_at = None, min(t + max_hold, last)
        for j in range(t + 1, exit_at + 1):
            stopped = low[j] <= stop if side > 0 else high[j] >= stop
            if stopped:
                outcome, exit_at = -1.0, j
                break
            if (high[j] >= target) if side > 0 else (low[j] <= target):
                outcome, exit_at = reward_risk, j
                break
        if outcome is None:
            outcome = side * (close[exit_at] - entry) / risk
        results.append(outcome - fee_bps / 10_000 * entry / risk)
        t = exit_at + 1
    return np.array(results, dtype=float)


def trade_statistics(results):
    """Win rate, expectancy, profit factor and the Kelly fraction f* = p − (1−p)/b."""
    n = len(results)
    stats = {'Trades': n, 'Win Rate': np.nan, 'Expectancy R': np.nan, 'Expectancy t': np.nan,
             'Profit Factor': np.nan, 'Kelly f*': np.nan}
    if not n:
        return stats
    wins, losses = results[results > 0], results[results <= 0]
    p = len(wins) / n
    stats['Win Rate'] = p
    stats['Expectancy R'] = float(results.mean())
    if n > 1 and results.std(ddof=1) > 0:
        stats['Expectancy t'] = float(results.mean() / (results.std(ddof=1) / math.sqrt(n)))
    loss_total = -losses.sum()
    stats['Profit Factor'] = float(wins.sum() / loss_total) if loss_total > 0 else np.inf
    if len(wins) and len(losses) and loss_total > 0:
        payoff = wins.mean() / -losses.mean()
        stats['Kelly f*'] = float(p - (1 - p) / payoff) if payoff > 0 else np.nan
    elif len(wins) and not len(losses):
        stats['Kelly f*'] = 1.0
    return stats


def classify_regime(hurst, vr_z):
    if pd.isna(hurst):
        return 'Unknown'
    if hurst >= 0.55 or (hurst > 0.5 and vr_z >= 1.96):
        return 'Trending'
    if hurst <= 0.45 or (hurst < 0.5 and vr_z <= -1.96):
        return 'Mean-Reverting'
    return 'Random Walk'


def bars_per_year_from_index(index):
    """Observed bars per calendar year, so session gaps (nights, weekends) are respected."""
    if not isinstance(index, pd.DatetimeIndex) or len(index) < 2:
        return np.nan
    days = (index[-1] - index[0]) / pd.Timedelta(days=1)
    return (len(index) - 1) / days * 365 if days > 0 else np.nan


def quant_edge_row(symbol, candles, bars_per_day, settings, quote_to_account=1.0):
    """All quant metrics, the backtest verdict and a sized trade plan for one symbol.

    bars_per_day: bars in a 24-hour day for round-the-clock markets, or None to measure
    the bar rate from the candle timestamps (US sessions).
    settings keys: equity, risk_pct, atr_multiple, reward_risk, threshold, max_hold, fee_bps.
    quote_to_account: account-currency value of one unit of the quote currency (1/158 for a
    USD account trading a JPY-quoted pair), so price distances convert to account money.
    """
    candles = clean_wicks(candles)
    returns = log_returns(candles['close'])
    bars_per_year = bars_per_day * 365 if bars_per_day else bars_per_year_from_index(candles.index)
    hurst = hurst_exponent(candles['close'])
    vr, vr_z = variance_ratio(returns)
    weight = regime_weight(hurst)
    signal = signal_series(candles, weight)
    ic, ic_t = information_coefficient(signal, candles['close'])
    var_95, cvar_95 = tail_risk(returns)
    yz = yang_zhang_variance(candles)
    stats = trade_statistics(barrier_backtest(
        candles, signal, settings['threshold'], settings['atr_multiple'],
        settings['reward_risk'], settings['max_hold'], settings['fee_bps'],
    ))

    q_now = float(signal.iloc[-1]) if pd.notna(signal.iloc[-1]) else np.nan
    entry = float(candles['close'].iloc[-1])
    atr = float(candles['atr14'].iloc[-1]) if pd.notna(candles['atr14'].iloc[-1]) else np.nan
    risk_distance = settings['atr_multiple'] * atr
    side = 0 if pd.isna(q_now) or abs(q_now) < settings['threshold'] else (1 if q_now > 0 else -1)

    reasons = []
    if stats['Trades'] < MIN_TRADES:
        reasons.append(f"only {stats['Trades']} historical trades")
    if not stats['Expectancy R'] > 0:
        reasons.append('expectancy ≤ 0 after fees')
    if not stats['Kelly f*'] > 0:
        reasons.append('Kelly ≤ 0')
    if not ic > 0:
        reasons.append('IC ≤ 0')
    if not (np.isfinite(risk_distance) and risk_distance > 0):
        reasons.append('ATR unavailable')
    if not (quote_to_account is not None and quote_to_account > 0):
        reasons.append('no FX rate to size the position')
        quote_to_account = np.nan

    risk_fraction, units = 0.0, 0.0
    if side == 0:
        verdict, reason = 'FLAT', f'|Q| below {settings["threshold"]:.2f}'
    elif reasons:
        verdict, reason = 'NO EDGE', '; '.join(reasons)
    else:
        # Risk the smaller of the user's limit and half-Kelly; cap the notional's leverage.
        risk_fraction = min(settings['risk_pct'] / 100, stats['Kelly f*'] / 2)
        units = settings['equity'] * risk_fraction / (risk_distance * quote_to_account)
        units = min(units, MAX_LEVERAGE * settings['equity'] / (entry * quote_to_account))
        verdict = 'LONG' if side > 0 else 'SHORT'
        reason = f"E[R] {stats['Expectancy R']:+.2f} over {stats['Trades']} trades, IC {ic:+.3f}"

    has_plan = side != 0 and np.isfinite(risk_distance) and risk_distance > 0
    stop = entry - side * risk_distance if has_plan else np.nan
    target = entry + side * settings['reward_risk'] * risk_distance if has_plan else np.nan
    dollar_risk = units * risk_distance * quote_to_account if units else 0.0
    return {
        'Symbol': symbol,
        'Regime': classify_regime(hurst, vr_z),
        'Hurst H': hurst,
        'Variance Ratio': vr,
        'VR z': vr_z,
        'OU Half-Life (bars)': ou_half_life(candles['close']),
        **annualized_volatility(yz, returns, bars_per_year),
        'PSR (drift > 0)': probabilistic_sharpe(returns),
        'CF VaR 95% (bar)': var_95,
        'CVaR 95% (bar)': cvar_95,
        'IC': ic,
        'IC t': ic_t,
        'Quant Score Q': q_now,
        'Trend Weight a': weight,
        **{f'BT {key}': value for key, value in stats.items()},
        'Verdict': verdict,
        # Whether the backtest, IC and ATR checks pass regardless of today's |Q|.
        'Backtest Gate': 'PASS' if not reasons else 'FAIL',
        'Reason': reason,
        'Entry': entry,
        'Stop': stop,
        'Target': target,
        'Risk %': risk_fraction,
        'Size (units)': units,
        'Notional': units * entry * quote_to_account if units else 0.0,
        'Leverage': units * entry * quote_to_account / settings['equity'] if units and settings['equity'] else 0.0,
        'Quote to Account': quote_to_account,
        '$ Risk': dollar_risk,
        '$ Expected': dollar_risk * stats['Expectancy R'] if units else 0.0,
    }

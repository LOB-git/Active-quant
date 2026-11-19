"""Backtest: trade the last hour's taker buy share, but only when the price and CVD lines are close.

Rule (one Binance USD-M perpetual, one candle interval):
  * Lines close: |G| < close_gap, where G is real_flow.price_cvd_gap (price and CVD each scaled to
    their own low..high over the last 200 candles, as the Real Money Flow chart draws them).
  * Buy share = taker buy USDT / total USDT over the last hour of closed candles.
    Buy share > buy_above -> LONG; buy share < 1 - buy_above (sellers above buy_above) -> SHORT.
  * Lines not close -> no trade.
  * Enter at the signal candle's close, hold `hold` (a time span), one trade at a time.
    Fees: fee_bps round trip, taken off every trade.

For comparison the same flow rule is also run when the lines are NOT close, and on every candle.
"""
import numpy as np
import pandas as pd

import real_flow as rf

DEFAULT_COINS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'XRP/USDT', 'BNB/USDT', 'DOGE/USDT']
HOLDS = {'1h': pd.Timedelta(hours=1), '4h': pd.Timedelta(hours=4), '24h': pd.Timedelta(hours=24)}
FEE_BPS = 10.0
VARIANTS = ('Lines close (your rule)', 'Lines NOT close', 'Ignore the lines')


def fetch_history(symbol, interval='5m', days=365, exchange=None, now=None):
    """Binance klines with taker volume for the last `days` days (paged, 1500 per request)."""
    if exchange is None:
        from binance_futures import usdm_exchange
        exchange = usdm_exchange()
    step = rf.BINANCE_INTERVAL_MS[interval]
    end = int((pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)).value // 1_000_000)
    since, rows = end - int(days * 86_400_000), []
    while since < end:
        page = exchange.fapiPublicGetKlines({'symbol': rf.market_id(symbol), 'interval': interval,
                                             'startTime': since, 'limit': 1500})
        if not page:
            break
        rows += page
        following = int(page[-1][0]) + step
        if following <= since:
            break
        since = following
    return rf.klines_frame(rows)


def signals(klines, close_gap=0.10, buy_above=0.51, flow_window='1h', now=None):
    """Per closed candle: gap, buy share of the last `flow_window`, and the signal for each variant."""
    now = pd.Timestamp.now(tz='UTC').tz_localize(None) if now is None else pd.Timestamp(now)
    closed = klines[klines['close_time'] <= now]
    flow = rf.flow_frame(closed)
    gap = rf.price_cvd_gap(flow)['gap']
    buy = flow['buy_usdt'].rolling(flow_window).sum()
    total = (flow['buy_usdt'] + flow['sell_usdt']).rolling(flow_window).sum()
    share = buy / total.replace(0, np.nan)
    side = pd.Series(np.select([share > buy_above, share < 1 - buy_above], [1, -1], 0), index=flow.index)
    side = side.where(share.notna(), 0)
    close = gap.abs() < close_gap
    frame = pd.DataFrame({'close': flow['close'], 'close_time': closed['close_time'], 'gap': gap,
                          'buy_share': share, 'flow_side': side, 'lines_close': close & gap.notna()})
    valid = gap.notna()
    frame[VARIANTS[0]] = side.where(frame['lines_close'], 0)
    frame[VARIANTS[1]] = side.where(valid & ~frame['lines_close'], 0)
    frame[VARIANTS[2]] = side.where(valid, 0)
    return frame


def run_trades(frame, signal_column, hold, fee_bps=FEE_BPS):
    """Non-overlapping trades: enter at the signal candle's close, exit at the close `hold` later."""
    closes = frame['close'].to_numpy()
    exit_times = frame['close_time'].to_numpy()
    sides = frame[signal_column].to_numpy()
    index = frame.index
    targets = frame['close_time'] + hold
    exit_pos = np.searchsorted(exit_times, targets.to_numpy(), side='left')
    trades, position = [], 0
    while position < len(frame):
        hits = np.flatnonzero(sides[position:] != 0)
        if not len(hits):
            break
        entry = position + int(hits[0])
        out = int(exit_pos[entry])   # first candle closing at or after entry + hold
        if out >= len(frame):
            break
        side = int(sides[entry])
        gross = side * (closes[out] / closes[entry] - 1)
        trades.append({'Entry Time': exit_times[entry], 'Side': 'LONG' if side > 0 else 'SHORT',
                       'Buy Share': frame['buy_share'].iloc[entry], 'Gap': frame['gap'].iloc[entry],
                       'Entry': closes[entry], 'Exit': closes[out], 'Exit Time': exit_times[out],
                       'Gross %': gross * 100, 'Net %': (gross - fee_bps / 10_000) * 100, 'open_candle': index[entry]})
        position = out + 1
    return pd.DataFrame(trades)


def summarize(trades):
    if trades.empty:
        return {'Trades': 0, 'Direction right': np.nan, 'Win rate after fees': np.nan, 'Avg gross %': np.nan,
                'Avg net %': np.nan,
                'Total net %': np.nan, 'Profit factor': np.nan, 'Longs': 0, 'Shorts': 0,
                'Long win rate': np.nan, 'Short win rate': np.nan}
    net = trades['Net %']
    wins, losses = net[net > 0].sum(), -net[net < 0].sum()
    longs, shorts = trades[trades['Side'] == 'LONG'], trades[trades['Side'] == 'SHORT']
    return {
        'Trades': int(len(trades)),
        'Direction right': float((trades['Gross %'] > 0).mean()),
        'Win rate after fees': float((net > 0).mean()),
        'Avg gross %': float(trades['Gross %'].mean()),
        'Avg net %': float(net.mean()),
        'Total net %': float(net.sum()),
        'Profit factor': float(wins / losses) if losses > 0 else np.inf,
        'Longs': int(len(longs)), 'Shorts': int(len(shorts)),
        'Long win rate': float((longs['Net %'] > 0).mean()) if len(longs) else np.nan,
        'Short win rate': float((shorts['Net %'] > 0).mean()) if len(shorts) else np.nan,
    }


def backtest(klines, holds=HOLDS, close_gap=0.10, buy_above=0.51, fee_bps=FEE_BPS, now=None):
    """Summary rows (variant x hold) and the trades of the user's rule for each hold."""
    frame = signals(klines, close_gap, buy_above, now=now)
    rows, trades_by_hold = [], {}
    for hold_name, hold in holds.items():
        for variant in VARIANTS:
            trades = run_trades(frame, variant, hold, fee_bps)
            rows.append({'Rule': variant, 'Hold': hold_name, **summarize(trades)})
            if variant == VARIANTS[0]:
                trades_by_hold[hold_name] = trades
    up = np.log(frame['close']).diff().dropna()
    info = {'from': frame.index.min(), 'to': frame.index.max(), 'candles': len(frame),
            'close_share': float(frame['lines_close'].mean()), 'market_change': float(frame['close'].iloc[-1] / frame['close'].iloc[0] - 1) if len(frame) else np.nan,
            'up_candles': float((up > 0).mean()) if len(up) else np.nan}
    return pd.DataFrame(rows), trades_by_hold, info

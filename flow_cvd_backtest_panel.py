"""🧪 Backtest of the 'price and CVD lines close + last hour's buy share' rule, for the Real Money Flow panel."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import flow_cvd_backtest as fb

STATE_KEY = 'flow_cvd_bt'


@st.cache_data(ttl=3600, show_spinner=False)
def load_history(symbol, interval, days):
    return fb.fetch_history(symbol, interval, days)


def equity_figure(trades_by_coin, hold):
    fig = go.Figure()
    for coin, trades in trades_by_coin.items():
        if trades.empty:
            continue
        fig.add_scatter(x=trades['Exit Time'], y=trades['Net %'].cumsum(), name=coin, mode='lines')
    fig.add_hline(y=0, line_color='#94a3b8', line_dash='dash')
    fig.update_layout(title=f'Your rule: running total of net % per trade (hold {hold}, after fees)', height=360,
                      margin={'l': 10, 'r': 10, 't': 50, 'b': 10}, yaxis_title='Sum of net % per trade',
                      hovermode='x unified', legend={'orientation': 'h', 'y': -0.2})
    return fig


def render_flow_rule_backtest():
    st.markdown(
        'Rule tested: when the **price and CVD lines are close** on the chart (|gap| below the limit), check the '
        "**last hour's taker buy share**. Above the buy limit → **LONG**; below 100% minus it (sellers above the "
        'limit) → **SHORT**. Lines not close → no trade. Enter at the candle close, hold for a fixed time, one trade '
        'at a time, fees taken off every trade. The same flow rule is also run when the lines are **not** close and '
        'with the lines **ignored**, to show whether "close" adds anything.'
    )
    c1, c2, c3 = st.columns(3)
    coins = c1.multiselect('Coins', fb.DEFAULT_COINS, fb.DEFAULT_COINS, key='fcb_coins')
    interval = c2.selectbox('Chart candles', ['5m', '15m', '1h'], key='fcb_interval')
    days = int(c3.number_input('Days of history', 30, 365, 365, 30, key='fcb_days'))
    c4, c5, c6 = st.columns(3)
    close_gap = c4.number_input('Lines "close" when |gap| below', 0.01, 0.50, 0.10, 0.01, key='fcb_gap')
    buy_above = c5.number_input('Buy share above (sell share above for SHORT)', 0.501, 0.80, 0.51, 0.01,
                                format='%.3f', key='fcb_buy')
    fee_bps = c6.number_input('Round-trip fees (bps)', 0.0, 50.0, fb.FEE_BPS, 1.0, key='fcb_fee')

    if st.button('Run the backtest', key='fcb_run', disabled=not coins):
        tables, trades, infos, failed = [], {}, {}, []
        bar = st.progress(0.0, text='Downloading Binance taker data…')
        for done, coin in enumerate(coins, start=1):
            try:
                table, by_hold, info = fb.backtest(load_history(coin, interval, days), close_gap=close_gap,
                                                   buy_above=buy_above, fee_bps=fee_bps)
                table.insert(0, 'Coin', coin)
                tables.append(table)
                trades[coin], infos[coin] = by_hold, info
            except Exception as exc:
                failed.append(f'{coin} ({exc})')
            bar.progress(done / len(coins), text=f'Backtested {coin}')
        bar.empty()
        st.session_state[STATE_KEY] = {
            'table': pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(), 'trades': trades,
            'infos': infos, 'failed': failed, 'settings': (interval, days, close_gap, buy_above, fee_bps),
        }

    result = st.session_state.get(STATE_KEY)
    if not result:
        return
    if result['failed']:
        st.warning('Could not backtest: ' + '; '.join(result['failed']))
    table = result['table']
    if table.empty:
        return
    interval, days, close_gap, buy_above, fee_bps = result['settings']
    infos = result['infos']
    first = min(i['from'] for i in infos.values())
    st.caption(
        f'{interval} candles, {first:%Y-%m-%d} → now, |gap| < {close_gap:.2f}, buy share > {buy_above:.1%}, '
        f'{fee_bps:g} bps fees. Lines were close on '
        + ', '.join(f"{c.split('/')[0]} {i['close_share']:.0%}" for c, i in infos.items())
        + ' of candles. Price change over the test: '
        + ', '.join(f"{c.split('/')[0]} {i['market_change']:+.0%}" for c, i in infos.items()) + '.'
    )

    pooled = pooled_summary(table)
    st.markdown('**All coins together**')
    st.dataframe(
        pooled.style.format({'Direction right': '{:.1%}', 'Win rate after fees': '{:.1%}', 'Avg net %': '{:+.3f}%',
                             'Avg gross %': '{:+.3f}%'})
        .map(lambda v: 'color: #16a34a' if v > 0 else 'color: #dc2626', subset=['Avg net %']),
        width='stretch', hide_index=True,
    )
    mine = pooled[pooled['Rule'] == fb.VARIANTS[0]]
    best = mine.loc[mine['Avg net %'].idxmax()]
    if (mine['Avg net %'] > 0).any() and best['Coins profitable'] > len(infos) / 2:
        st.success(f"Your rule made money after fees with a {best['Hold']} hold on {int(best['Coins profitable'])} of "
                   f"{len(infos)} coins. Check it holds on other periods before trusting it.")
    else:
        st.error(
            f"**No edge.** Your rule called the direction right {mine['Direction right'].min():.0%}–"
            f"{mine['Direction right'].max():.0%} of the time (a coin flip is 50%) and lost money after fees on "
            "average for every hold. Compare the rows: whether the lines are close or not makes little difference."
        )

    with st.expander('Per coin'):
        per = table.copy()
        st.dataframe(per.style.format({
            'Direction right': '{:.1%}', 'Win rate after fees': '{:.1%}', 'Avg net %': '{:+.3f}%',
            'Avg gross %': '{:+.3f}%', 'Total net %': '{:+.1f}%', 'Profit factor': '{:.2f}', 'Long win rate': '{:.1%}',
            'Short win rate': '{:.1%}'}, na_rep='—'), width='stretch', hide_index=True)

    holds = list(fb.HOLDS)
    hold = st.radio('Show trades for hold', holds, index=1, horizontal=True, key='fcb_hold')
    by_coin = {coin: trades[hold] for coin, trades in result['trades'].items()}
    st.plotly_chart(equity_figure(by_coin, hold), width='stretch')
    with st.expander(f'Every trade of your rule (hold {hold})'):
        rows = pd.concat([t.assign(Coin=c) for c, t in by_coin.items() if not t.empty], ignore_index=True)
        if not rows.empty:
            rows = rows.drop(columns='open_candle').sort_values('Entry Time', ascending=False)
            st.dataframe(rows[['Coin', 'Entry Time', 'Side', 'Buy Share', 'Gap', 'Entry', 'Exit', 'Exit Time',
                               'Gross %', 'Net %']].style.format({
                'Buy Share': '{:.1%}', 'Gap': '{:+.3f}', 'Entry': '{:,.6g}', 'Exit': '{:,.6g}',
                'Gross %': '{:+.3f}%', 'Net %': '{:+.3f}%'}), width='stretch', hide_index=True, height=320)


def pooled_summary(table):
    """Trade-weighted totals across coins for each rule and hold."""
    rows = []
    for (hold, rule), group in table.groupby(['Hold', 'Rule'], sort=False):
        trades = group['Trades'].sum()
        if not trades:
            continue
        weight = group['Trades'] / trades
        rows.append({
            'Hold': hold, 'Rule': rule, 'Trades': int(trades),
            'Direction right': float((group['Direction right'] * weight).sum()),
            'Win rate after fees': float((group['Win rate after fees'] * weight).sum()),
            'Avg gross %': float((group['Avg gross %'] * weight).sum()),
            'Avg net %': float((group['Avg net %'] * weight).sum()),
            'Coins profitable': int((group['Total net %'] > 0).sum()),
        })
    return pd.DataFrame(rows)

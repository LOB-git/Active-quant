# QQQ + NQ daily-alert confirmation

Implemented in the US Indices panel above Zone Battle Score and in its adjacent backtest.

## Source and timing

The filter reads `daily_ghana_zscore_alerts`, `daily_ghana_zscore_alert_timeframe`
and `daily_ghana_zscore_alert_threshold` directly from Streamlit session state.
These are the actual records behind **Daily Historical Z-Score Alerts — Ghana Time**;
the quantitative panel does not fetch VIX separately or recalculate these daily alerts.
Refresh Indices Data rebuilds this session-scoped source; restarting a session requires a refresh.
The scoring-candle refresh is the separate Generate QQQ + NQ Signals button.

The source uses one-hour candles. Its threshold is the overview's stored threshold
(default 1), with Z-Score and Momentum Z both at or beyond the same signed threshold.
It retains the first positive and first negative event per symbol per Ghana day.
QQQ, NQ=F and ^VIX are the exact source symbols. Other symbols are ignored.

Ghana Time in this source is already candle-close time. The overview excludes
unfinished candles when refreshed. At each scoring candle's close, an exact
backward time lookup finds each asset's latest stored event at or before that time,
restricted to that same Ghana calendar day. No additional hour is added to alerts.
An opposite later event supersedes the earlier daily direction. These event states
are not continuous measurements of the current Z-score. Nothing carries overnight.

## Confirmation rule

The existing demand/supply score remains unchanged for each asset:

`S = 100 × [0.60 tanh(ΔFlow Z) + 0.25 tanh(ΔMomentum Z) + 0.15 × I(ΔVolatility Z > 0) × sign(ΔPrice) × tanh(|ΔVolatility Z|)]`

The weaker aligned QQQ/NQ score is displayed as Confirmation Score. BUY requires
both scores at least the chosen positive score threshold; SELL requires both at
or below the negative threshold. This is the **Base Signal**, not yet permission to trade.

| Final signal | Base signal | QQQ daily alert | NQ daily alert | VIX daily alert |
| --- | --- | --- | --- | --- |
| BUY | BUY | Positive | Positive | Negative |
| SELL | SELL | Negative | Negative | Positive |
| WAIT | BUY or SELL | Missing, unconfirmed, or inconsistent with the required combination | | |

Opposing base direction scores remain CONFLICT. Missing factors and unmatched
QQQ/NQ candle timestamps still prevent trades. The original hourly grids are not
rounded or filled; QQQ and NQ can have different grids. Alerts arriving between
scoring closes are evaluated at the next matching completed scoring candle.

New Alert is recalculated **after** applying the daily filter. A confirmation
enabled by a newly arrived daily event can therefore trigger entry even if the
base score direction has not changed. Every signal row exposes the actual three
source timestamps, Z-scores and Momentum Z values plus a reason.

## Backtest and limitations

The backtest calls the same filtered signal builder. Entry remains the exact
five-minute open at the confirmation time. Stops and targets are monitored using
five-minute bars; the default target remains the nearest already-known one-hour
liquidity zone. Costs, one-position-at-a-time and stop-first intrabar rules are unchanged.
Source changes invalidate saved results. Trades and decision exports include daily
alert evidence; the complete decision table also exposes WAIT and CONFLICT rows.

This is a testable directional hypothesis, not a claim that VIX always moves
oppositely or that the filter improves accuracy. Selectivity can leave very few
trades. Historical source reconstruction does not prove real-time delivery latency,
and revised data can change a rebuilt source. Costs and feed coverage also matter.

## Verification

`tests/test_qqq_nq_daily_alerts.py` verifies signed thresholds, causal matching,
day boundaries, source validation, unchanged inputs, duplicate handling, and
agreement between the panel's signal builder and the backtest. It also checks the
actual overview builder excludes unfinished candles. Panel tests exercise source
evidence, missing-data warnings and changes of stored timeframe.

`scripts/verify_qqq_nq.py` and `scripts/verify_qqq_nq_backtest.py` use public market
history and the real overview daily-alert builder to reconstruct a QA source;
this is not a read of an existing browser session. The latter saves a separately
named `qqq_nq_daily_filtered_*` run, including source alerts and all signal decisions,
without overwriting the earlier unfiltered baseline.

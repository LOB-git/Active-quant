"""'In progress' zone battles: zones price has entered whose reaction window has not finished.

The running score uses the same formula as a finished battle, on the candles so far (including the
still-forming candle, so it moves until that candle closes). It is provisional: the final score can
end up anywhere until the last reaction candle has closed.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROGRESS_COLOR = '#a855f7'


def _utc(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')


def due_in(due, now=None):
    """'1h 05m' until the verdict, or 'due now'."""
    if due is None or pd.isna(due):
        return '—'
    left = _utc(due) - _utc(now or pd.Timestamp.now(tz='UTC'))
    if left <= pd.Timedelta(0):
        return 'due now'
    minutes = int(left.total_seconds() // 60)
    return f'{minutes // 60}h {minutes % 60:02d}m' if minutes >= 60 else f'{minutes}m'


def add_in_progress_markers(fig, progress):
    """Hollow purple markers at the latest candle showing each running score."""
    if progress is None or progress.empty:
        return fig
    fig.add_trace(go.Scatter(
        x=progress['Score Time'], y=progress['Running Score'], mode='markers',
        name='In progress (provisional)',
        marker=dict(symbol='circle-open', size=13, line=dict(width=2, color=PROGRESS_COLOR)),
        customdata=np.c_[progress['Candles Closed'], progress['Candles Needed'],
                         progress['Zone Entry'].astype(str), progress['Leaning']],
        hovertemplate=('IN PROGRESS · running score %{y:+.2f}<br>%{customdata[0]} of %{customdata[1]} '
                       'candles closed<br>Entered zone %{customdata[2]}<br>%{customdata[3]}<extra></extra>'),
    ))
    return fig


def progress_table(progress, now=None):
    view = pd.DataFrame({
        'Zone Type': progress['Zone Type'],
        'Zone Range': [f'{b:,.6g} – {t:,.6g}' for b, t in zip(progress['Zone Bottom'], progress['Zone Top'])],
        'Entered': progress['Zone Entry'],
        'Progress': [f'{c} / {n}' for c, n in zip(progress['Candles Closed'], progress['Candles Needed'])],
        'Candles Left': progress['Candles Left'],
        'Verdict Due (UTC)': [_utc(d).strftime('%Y-%m-%d %H:%M') if pd.notna(d) else '—' for d in progress['Verdict Due']],
        'Due In': [due_in(d, now) for d in progress['Verdict Due']],
        'Running Score': progress['Running Score'],
        'Leaning': progress['Leaning'],
        'Rz': progress['Reaction Rz'], 'Fz': progress['Flow Fz'],
        'Mz': progress['Momentum Mz'], 'Az': progress['Acceptance Az'],
    })
    return view


def render_in_progress(progress, observation_bars, key):
    """Countdown for the newest battle plus a table of every battle in progress."""
    st.markdown('#### ⏳ In progress')
    if progress is None or progress.empty:
        st.caption(f'No battle in progress: price has not entered a zone within the last {int(observation_bars)} candles.')
        return
    latest = progress.iloc[0]
    closed, needed = int(latest['Candles Closed']), int(latest['Candles Needed'])
    st.progress(min(closed / needed, 1.0),
                text=(f"{latest['Zone Type']} zone {latest['Zone Bottom']:,.6g} – {latest['Zone Top']:,.6g}: "
                      f"{closed} of {needed} candles closed · verdict due "
                      f"{_utc(latest['Verdict Due']):%H:%M} UTC ({due_in(latest['Verdict Due'])})"))
    c1, c2, c3 = st.columns(3)
    c1.metric('Running score', f"{latest['Running Score']:+.2f}", latest['Leaning'], delta_color='off', delta_arrow='off')
    c2.metric('Candles left', int(latest['Candles Left']))
    c3.metric('Battles in progress', len(progress))
    st.dataframe(
        progress_table(progress).style.format({
            'Running Score': '{:+.2f}', 'Rz': '{:+.2f}', 'Fz': '{:+.2f}', 'Mz': '{:+.2f}', 'Az': '{:+.2f}',
        }),
        width='stretch', hide_index=True, key=f'{key}_progress_table',
    )
    st.caption(
        'Provisional: the running score uses the same formula as a finished battle, on the candles so far'
        + (', including the candle that is still forming,' if bool(latest['Includes Live Candle']) else '')
        + ' so it changes until the last reaction candle closes. Only then does it become a dot on the chart. '
        'Hollow purple circles on the chart mark these running scores.'
    )

import unittest
from datetime import date

import numpy as np
import pandas as pd

from qqq_nq_signals import (
    build_qqq_nq_signals, combine_direction_histories, demand_supply_score,
    historical_z_components, signals_on_ghana_day,
)


def candle_history(periods=180):
    rng = np.random.default_rng(43)
    idx = pd.date_range('2026-08-01', periods=periods, freq='h', tz='UTC')
    close = 100 + np.cumsum(rng.normal(0, 1, periods))
    df = pd.DataFrame({'close': close, 'open': close + rng.normal(0, 1, periods),
                       'volume': rng.uniform(10, 100, periods),
                       'atr14': rng.uniform(1, 3, periods)}, index=idx)
    df['momentum'] = df['close'].pct_change(30)
    df['z_score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
    return df


def scored(values, index):
    return pd.DataFrame({'Score': values, 'Ready': True}, index=index)


class ConfirmationTests(unittest.TestCase):
    def test_inclusive_thresholds_conflicts_and_weaker_asset(self):
        times = pd.date_range('2026-09-01', periods=6, freq='h', tz='UTC')
        q = scored([10, -10, 90, 9, 20, 20], times)
        n = scored([30, -30, -1, 40, 30, 30], times)
        result = combine_direction_histories(q, n, '1h')
        self.assertEqual(result['Signal'].tolist(), ['BUY', 'SELL', 'CONFLICT', 'WAIT', 'BUY', 'BUY'])
        self.assertEqual(result['Confirmation Score'].tolist(), [10, -10, 0, 9, 20, 20])
        self.assertEqual(result['New Alert'].tolist(), [True, True, False, False, True, False])

    def test_misaligned_and_missing_bars_never_confirm(self):
        times = pd.date_range('2026-09-01', periods=3, freq='h', tz='UTC')
        q = scored([20, 20, 20], times)
        n = scored([40, 40], times[[0, 2]])
        result = combine_direction_histories(q, n, '1h')
        self.assertEqual(result['Signal'].tolist(), ['BUY', 'WAIT', 'BUY'])
        self.assertTrue(result.iloc[-1]['New Alert'])
        result = combine_direction_histories(q, scored([40, 40, 40], times + pd.Timedelta('30min')), '1h')
        self.assertTrue(result['Signal'].eq('WAIT').all())
        self.assertTrue(result['Confirmation Score'].isna().all())

    def test_score_requires_all_factors_and_matches_legacy_formula(self):
        self.assertTrue(np.isnan(demand_supply_score(1, 1, np.nan, 1)))
        for f, m, v, p in [(1, .3, .7, -2), (-.5, .3, -.1, 2), (0, 0, 0, 0)]:
            vd = np.sign(p) * np.tanh(abs(v)) if v > 0 else 0
            expected = 100 * (.6 * np.tanh(f) + .25 * np.tanh(m) + .15 * vd)
            self.assertAlmostEqual(float(demand_supply_score(f, m, v, p)), expected)

    def test_components_match_historical_alert_table(self):
        history = candle_history()
        up = history['close'] >= history['open']
        inflow = history['volume'].where(up, 0).rolling(20).sum()
        outflow = history['volume'].where(~up, 0).rolling(20).sum()
        flow = (inflow - outflow) / (inflow + outflow)
        expected = (flow - flow.rolling(50).mean()) / flow.rolling(50).std()
        pd.testing.assert_series_equal(historical_z_components(history)['Flow (z)'], expected, check_names=False)

    def test_prefix_invariance_excludes_open_candle(self):
        history = candle_history()
        cutoff = history.index[150] + pd.Timedelta('30min')
        result = build_qqq_nq_signals(history, history, as_of=cutoff)
        prefix = build_qqq_nq_signals(history.iloc[:150], history.iloc[:150], as_of=cutoff)
        pd.testing.assert_frame_equal(result, prefix)
        self.assertTrue((result['Signal Time (Ghana)'] <= cutoff).all())
        self.assertFalse(result.index.isin([history.index[150]]).any())

    def test_ghana_day_uses_signal_time_and_preserves_midnight_change(self):
        times = pd.date_range('2026-09-01 18:00', periods=3, freq='h', tz='America/New_York')
        times = times.tz_convert('UTC')
        result = combine_direction_histories(scored([-30, 30, -30], times), scored([-30, 30, -30], times), '1h')
        day = signals_on_ghana_day(result, date(2026, 9, 2))
        self.assertEqual(day['Signal'].tolist(), ['BUY', 'SELL'])
        self.assertEqual(day.iloc[0]['Signal Time (Ghana)'].hour, 0)

    def test_zero_volume_short_or_missing_history_cannot_trade(self):
        history = candle_history()
        no_volume = history.assign(volume=0)
        for nq in [no_volume, history.iloc[:30], None]:
            result = build_qqq_nq_signals(history, nq, as_of='2026-09-01')
            self.assertFalse(result['Signal'].isin(['BUY', 'SELL']).any())


if __name__ == '__main__':
    unittest.main()

import copy
import ast
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from qqq_nq_daily_alerts import (
    alert_source_fingerprint, confirm_with_daily_alerts, prepare_daily_alerts,
    stored_alert_inputs,
)
from qqq_nq_signals import build_qqq_nq_signals, combine_direction_histories
from qqq_nq_backtest import backtest_qqq_nq
from tests.test_qqq_nq_signals import candle_history, scored


def alert(symbol, time, z):
    timestamp = pd.Timestamp(time, tz='UTC')
    return {'Symbol': symbol, 'Ghana Time': timestamp, 'Ghana Date': timestamp.date(),
            'Z-Score': z, 'Momentum Z': z, 'Timeframe': '1h',
            'Alert': 'POSITIVE Z-SCORE' if z > 0 else 'NEGATIVE Z-SCORE'}


def daily_fixture():
    return [alert(symbol, str(day.date()), z)
            for day in pd.date_range('2026-08-01', '2026-08-09')
            for symbol, z in [('QQQ', 1), ('NQ=F', 1), ('^VIX', -1)]]


def base_signals(scores, start='2026-09-01 00:00'):
    times = pd.date_range(start, periods=len(scores), freq='h', tz='UTC')
    return combine_direction_histories(scored(scores, times), scored(scores, times), '1h')


class DailyAlertTests(unittest.TestCase):
    def test_actual_overview_builder_excludes_unfinished_candles(self):
        # Isolate the real function without running the dashboard's unrelated imports.
        source = Path(__file__).resolve().parents[1] / 'Untitled-1.py'
        definition = next(node for node in ast.parse(source.read_text(encoding='utf-8')).body
                          if isinstance(node, ast.FunctionDef) and node.name == 'build_daily_zscore_alert_history')
        namespace = {'pd': pd, 'np': np}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), 'exec'), namespace)
        times = pd.date_range('2026-09-01 01:00', periods=3, freq='h', tz='UTC')
        history = pd.DataFrame({'close': [100., 99., 98.]}, index=times)
        z = pd.Series([1., -1., -2.], index=times)
        rows = namespace['build_daily_zscore_alert_history'](
            history, z, '^VIX', '1h', momentum_z_series=z, as_of='2026-09-01 01:30',
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['Ghana Time'], times[0])
        self.assertEqual(rows[0]['Alert'], 'POSITIVE Z-SCORE')

    def test_alert_arrival_enables_new_confirmation_and_later_direction_replaces_it(self):
        rows = [alert('QQQ', '2026-09-01 00:15', 1), alert('NQ=F', '2026-09-01 00:30', 1),
                alert('^VIX', '2026-09-01 02:00', -1), alert('^VIX', '2026-09-01 04:00', 1)]
        result = confirm_with_daily_alerts(base_signals([20] * 4), rows[::-1], '1h')
        self.assertEqual(result['Signal'].tolist(), ['WAIT', 'BUY', 'BUY', 'WAIT'])
        self.assertEqual(result['New Alert'].tolist(), [False, True, False, False])
        self.assertTrue(pd.isna(result.iloc[0]['VIX Daily Z']))
        self.assertEqual(result.iloc[1]['VIX Alert Time (Ghana)'], rows[2]['Ghana Time'])
        self.assertEqual(result.iloc[-1]['VIX Daily Z'], 1)
        # The join may neither multiply rows nor consume a later alert.
        self.assertEqual(len(result), 4)
        for name in ['QQQ', 'NQ', 'VIX']:
            known = result[f'{name} Alert Time (Ghana)'].dropna()
            self.assertTrue(known.le(result.loc[known.index, 'Signal Time (Ghana)']).all())

    def test_sell_uses_negative_equities_positive_vix_and_base_still_required(self):
        rows = [alert(symbol, '2026-09-01 01:00', z)
                for symbol, z in [('QQQ', -1), ('NQ=F', -1), ('^VIX', 1)]]
        result = confirm_with_daily_alerts(base_signals([-20, 20, -9]), rows, '1h')
        self.assertEqual(result['Signal'].tolist(), ['SELL', 'WAIT', 'WAIT'])
        self.assertEqual(result['Daily Alert Confirmation'].tolist(), ['SELL'] * 3)
        self.assertIn('opposes', result.iloc[1]['Reason'])

    def test_midnight_resets_and_no_future_or_next_day_alert_leakage(self):
        rows = [alert(symbol, '2026-09-01 23:00', z)
                for symbol, z in [('QQQ', 1), ('NQ=F', 1), ('^VIX', -1)]]
        future = [alert(symbol, '2026-09-02 01:00', z)
                  for symbol, z in [('QQQ', 1), ('NQ=F', 1), ('^VIX', -1)]]
        base = base_signals([20] * 3, start='2026-09-01 22:00')
        result = confirm_with_daily_alerts(base, rows + future, '1h')
        self.assertEqual(result['Signal'].tolist(), ['BUY', 'WAIT', 'BUY'])
        self.assertEqual(result['New Alert'].tolist(), [True, False, True])
        prefix = confirm_with_daily_alerts(base.iloc[:2], rows, '1h')
        pd.testing.assert_frame_equal(result.iloc[:2], prefix)

    def test_missing_wrong_timeframe_and_invalid_sources_fail_closed(self):
        rows = [alert(symbol, '2026-09-01', z)
                for symbol, z in [('QQQ', 1), ('NQ=F', 1), ('^VIX', -1)]]
        base = base_signals([20, 20])
        broken = copy.deepcopy(rows)
        broken[-1]['Momentum Z'] = 1
        bad_time = copy.deepcopy(rows)
        bad_time[0]['Ghana Time'] = 'invalid'
        duplicate_conflict = rows + [alert('^VIX', '2026-09-01', 1)]
        for source in [None, [], rows[:2], broken, bad_time, duplicate_conflict]:
            with self.subTest(source=source):
                result = confirm_with_daily_alerts(base, source, '1h')
                self.assertTrue(result['Signal'].eq('WAIT').all())
                self.assertFalse(result['New Alert'].any())
        result = confirm_with_daily_alerts(base, rows, '1h', alert_timeframe='15m')
        self.assertTrue(result['Signal'].eq('WAIT').all())
        self.assertIn('must use 1h', result.iloc[0]['Daily Alert Reason'])

    def test_source_unchanged_duplicates_collapsed_and_unrelated_symbols_ignored(self):
        rows = daily_fixture()
        original = copy.deepcopy(rows)
        clean, issue = prepare_daily_alerts(rows + rows + [{'Symbol': 'SPY'}])
        self.assertFalse(issue)
        self.assertEqual(len(clean), len(rows))
        self.assertEqual(rows, original)
        inputs = stored_alert_inputs({'daily_ghana_zscore_alerts': rows,
                                      'daily_ghana_zscore_alert_timeframe': '1h'})
        fingerprint = alert_source_fingerprint(inputs)
        inputs['daily_alerts'] = list(reversed(rows))
        self.assertEqual(fingerprint, alert_source_fingerprint(inputs))
        inputs['daily_alerts'] = rows[:-1]
        self.assertNotEqual(fingerprint, alert_source_fingerprint(inputs))

    def test_backtest_and_panel_function_use_identical_filtered_signals(self):
        history = candle_history()
        times = pd.date_range('2026-08-01', '2026-08-09', freq='5min', tz='UTC')
        execution = pd.DataFrame({'open': 100., 'close': 100., 'high': 101., 'low': 99., 'volume': 100}, index=times)
        rows = daily_fixture()
        arguments = dict(daily_alerts=rows, alert_timeframe='1h', alert_threshold=1.)
        expected = build_qqq_nq_signals(history, history, as_of='2026-08-09', **arguments)
        result = backtest_qqq_nq(
            history, history, execution, None, start='2026-08-01', end_exclusive='2026-08-09',
            as_of='2026-08-09', target_mode='fixed_rr', **arguments,
        )
        pd.testing.assert_frame_equal(result['signals'], expected)
        self.assertGreater(len(result['trades']), 0)
        self.assertTrue(result['trades']['VIX Daily Z'].eq(-1).all())
        self.assertTrue(result['trades']['QQQ Daily Z'].eq(1).all())
        for label in ('VIX', 'NQ', 'QQQ'):
            self.assertTrue(result['trades'][f'{label} Alert Time (Ghana)'].le(result['trades']['Entry Time (Ghana)']).all())


if __name__ == '__main__':
    unittest.main()

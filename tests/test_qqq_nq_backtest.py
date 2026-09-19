import unittest

import numpy as np
import pandas as pd

from qqq_nq_backtest import execute_nq_confirmations, summarize_trades


def execution_bars():
    return pd.DataFrame({'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 100.},
                        index=pd.date_range('2026-09-01', periods=40, freq='5min', tz='UTC'))


def confirmations(bars, positions=(15,), directions=('BUY',)):
    return pd.DataFrame({'Signal Time (Ghana)': bars.index[list(positions)],
                         'Signal': list(directions), 'New Alert': True,
                         'QQQ Score': 30., 'NQ Score': 40., 'Confirmation Score': 30.})


class ExecutionTests(unittest.TestCase):
    def run_bt(self, bars, signals=None, **kwargs):
        options = dict(target_mode='fixed_rr', slippage_points=0., commission_points=0.,
                       start='2026-09-01', end_exclusive='2026-09-02', as_of='2026-09-02')
        options.update(kwargs)
        liquidity = options.pop('liquidity_history', None)
        return execute_nq_confirmations(confirmations(bars) if signals is None else signals,
                                        bars, liquidity, **options)

    def test_entry_at_signal_open_target_on_entry_bar_atr_uses_past(self):
        bars = execution_bars()
        bars.iloc[15, bars.columns.get_loc('high')] = 105
        result = self.run_bt(bars)
        trade = result['trades'].iloc[0]
        self.assertEqual(trade['Entry Time (Ghana)'], bars.index[15])
        self.assertEqual(trade['ATR14'], 2.)
        self.assertEqual(trade['Net R'], 2.)
        self.assertEqual(trade['Exit Reason'], 'Take profit')

    def test_short_and_stop_first_for_ambiguous_bar(self):
        bars = execution_bars()
        bars.loc[bars.index[15], ['low', 'high']] = [95, 105]
        for side in ('BUY', 'SELL'):
            result = self.run_bt(bars, confirmations(bars, directions=(side,)))
            self.assertEqual(result['trades'].iloc[0]['Net R'], -1.)
            self.assertEqual(result['trades'].iloc[0]['Exit Reason'], 'Stop (both touched)')
        bars.loc[bars.index[15], 'high'] = 101
        result = self.run_bt(bars, confirmations(bars, directions=('SELL',)))
        self.assertEqual(result['trades'].iloc[0]['Net R'], 2.)

    def test_gapped_stop_fills_at_open_and_costs_reconcile(self):
        bars = execution_bars()
        bars.loc[bars.index[16], ['open', 'high', 'low', 'close']] = [95, 96, 94, 95]
        result = self.run_bt(bars, slippage_points=.25, commission_points=.5)
        trade = result['trades'].iloc[0]
        self.assertEqual(trade['Exit Reason'], 'Stop gap')
        self.assertEqual(trade['Exit Price'], 94.75)
        self.assertAlmostEqual(trade['Net Points'], trade['Gross Points'] - trade['Cost Points'])
        self.assertLess(trade['Net R'], -1.)
        self.assertEqual(result['summary']['Max Closed Drawdown R'], -trade['Net R'])

    def test_missing_entry_is_skipped_instead_of_using_stale_candle(self):
        bars = execution_bars()
        signals = confirmations(bars)
        result = self.run_bt(bars.drop(bars.index[15]), signals)
        self.assertEqual(result['summary']['Trades'], 0)
        self.assertIn('exact signal time', result['audit'].iloc[0]['Reason'])

    def test_open_position_skips_signal_and_reenters_after_exit_bar(self):
        bars = execution_bars()
        bars.loc[bars.index[18], 'high'] = 105
        result = self.run_bt(bars, confirmations(bars, (15, 16, 19), ('BUY', 'BUY', 'BUY')))
        self.assertEqual(result['summary']['Trades'], 2)
        self.assertEqual(result['audit']['Status'].tolist(), ['Traded', 'Skipped', 'Traded'])

    def test_unchanged_confirmation_does_not_reenter(self):
        bars = execution_bars()
        signals = confirmations(bars, (15, 16), ('BUY', 'BUY'))
        signals.loc[1, 'New Alert'] = False
        bars.loc[bars.index[15], 'high'] = 105
        result = self.run_bt(bars, signals)
        self.assertEqual(result['summary']['New confirmations'], 1)

    def test_liquidity_only_receives_closed_history_and_target_is_nearest(self):
        bars = execution_bars()
        entry = bars.index[15]
        liquidity = pd.DataFrame({'open': 100., 'high': 110., 'low': 90., 'close': 100., 'volume': 10.},
                                 index=pd.date_range(end=entry.floor('h') + pd.Timedelta('5h'), periods=80, freq='h'))
        def zones(history, **kwargs):
            self.assertTrue(((history.index + pd.Timedelta('1h')) <= entry).all())
            self.assertGreaterEqual(len(history), 50)
            return pd.DataFrame({'Zone Bottom': [108., 103., 93.], 'Zone Top': [109., 104., 94.]}), None
        bars.loc[bars.index[15], 'high'] = 105
        result = self.run_bt(bars, target_mode='liquidity', liquidity_history=liquidity, zone_calculator=zones)
        self.assertEqual(result['trades'].iloc[0]['Target Price'], 103.)
        self.assertEqual(result['trades'].iloc[0]['Net R'], 1.5)

    def test_no_liquidity_target_is_explicit_skip(self):
        bars = execution_bars()
        liquidity = bars.copy()
        liquidity.index = pd.date_range(end=bars.index[0], periods=len(bars), freq='h')
        result = self.run_bt(bars, target_mode='liquidity', liquidity_history=liquidity,
                             zone_calculator=lambda *a, **kw: (pd.DataFrame(), None))
        self.assertEqual(result['summary']['Trades'], 0)
        self.assertIn('50 completed', result['audit'].iloc[0]['Reason'])

    def test_future_exit_cannot_change_test_period_and_gap_is_flagged(self):
        bars = execution_bars()
        cutoff = bars.index[20]
        result = self.run_bt(bars, end_exclusive=cutoff)
        bars.loc[bars.index[20], 'high'] = 10000
        changed = self.run_bt(bars, end_exclusive=cutoff)
        pd.testing.assert_frame_equal(result['trades'], changed['trades'])
        self.assertEqual(result['trades'].iloc[0]['Exit Time (Ghana)'], cutoff)
        self.assertEqual(result['trades'].iloc[0]['Exit Reason'], 'End of period')
        gaps = self.run_bt(bars.drop(bars.index[17]))
        self.assertTrue(gaps['trades'].iloc[0]['Bar Gap'])

    def test_initial_loss_counts_toward_drawdown(self):
        trades = pd.DataFrame({'Net Points': [-10., -20., 5.], 'Net R': [-1., -2., .5], 'Bar Gap': False})
        summary = summarize_trades(trades, pd.DataFrame())
        self.assertEqual(summary['Max Closed Drawdown R'], 3.)
        self.assertAlmostEqual(summary['Profit Factor (points)'], 5/30)


if __name__ == '__main__':
    unittest.main()

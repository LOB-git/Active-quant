import unittest

import numpy as np

from historical_flow_analysis import analyze_ghana_flow_events, highest_flow_zone
from tests.test_qqq_nq_signals import candle_history


class HistoricalFlowAnalysisTests(unittest.TestCase):
    def test_selected_ghana_day_uses_shared_components_and_extreme_flow_events(self):
        history = candle_history(240)
        history['high'] = history[['open', 'close']].max(axis=1) + 1
        history['low'] = history[['open', 'close']].min(axis=1) - 1
        selected_day = history.index[-1].date()
        components, changes, events = analyze_ghana_flow_events(history, 'TEST/USDT', selected_day)
        self.assertFalse(components.empty)
        self.assertEqual(components.index.tolist(), changes.index.tolist())
        self.assertTrue(events['Ghana Time'].dt.date.eq(selected_day).all())
        self.assertTrue(events['Flow Z Change'].ne(0).all())
        self.assertTrue(set(events['Alert']).issubset({'Flow Z Rise Alert', 'Flow Z Drop Alert'}))
        for direction, sign in [('rise', 1), ('drop', -1)]:
            zone = highest_flow_zone(events, direction)
            candidates = events.loc[events['Flow Z Change'] * sign > 0, 'Flow Z Change']
            if candidates.empty:
                self.assertIsNone(zone)
            else:
                self.assertIsNotNone(zone)
                expected = candidates.max() if direction == 'rise' else candidates.min()
                self.assertEqual(zone['Flow Z Change'], expected)
                self.assertAlmostEqual(zone['Event Size'], abs(expected))
                self.assertIn(zone['Classification'], {
                    'Candidate Demand Zone', 'Candidate Supply Zone', 'Unresolved / Balanced Zone',
                })

    def test_requires_complete_price_and_component_input(self):
        history = candle_history(100)
        history['high'] = history[['open', 'close']].max(axis=1) + 1
        history['low'] = history[['open', 'close']].min(axis=1) - 1
        history = history.drop(columns=['atr14'])
        components, changes, events = analyze_ghana_flow_events(history, 'TEST/USDT', history.index[-1].date())
        self.assertTrue(components.empty)
        self.assertTrue(changes.empty)
        self.assertTrue(events.empty)
        self.assertIsNone(highest_flow_zone(events, 'rise'))


if __name__ == '__main__':
    unittest.main()

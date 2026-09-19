import unittest
from streamlit.testing.v1 import AppTest


SCRIPT = '''
from tests.test_qqq_nq_signals import candle_history
from qqq_nq_panel import render_qqq_nq_panel
import streamlit as st
def fetch(symbol, timeframe, silent):
    st.session_state['calls'] = st.session_state.get('calls', []) + [(symbol, timeframe)]
    return candle_history()
render_qqq_nq_panel(fetch)
'''


class PanelTests(unittest.TestCase):
    def test_local_button_chart_table_and_timeframe_guard(self):
        app = AppTest.from_string(SCRIPT, default_timeout=20).run()
        self.assertEqual(len(app.exception), 0)
        self.assertNotIn('calls', app.session_state.filtered_state)
        app.button(key='qqq_nq_generate').click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.session_state['calls'], [('QQQ', '1h'), ('NQ=F', '1h')])
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual(len(app.get('plotly_chart')), 1)
        self.assertTrue(app.dataframe[-1].value['Signal'].eq('WAIT').all())
        self.assertTrue(any('No stored daily alerts' in item.value for item in app.warning))
        app.number_input(key='qqq_nq_threshold').set_value(40.0).run()
        self.assertEqual(len(app.session_state['calls']), 2)
        app.selectbox(key='qqq_nq_timeframe').set_value('15m').run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.dataframe), 0)
        self.assertIn('15m', app.info[0].value)

    def test_stored_alert_evidence_and_source_changes_update_signals(self):
        from tests.test_qqq_nq_daily_alerts import daily_fixture
        app = AppTest.from_string(SCRIPT, default_timeout=20)
        app.session_state['daily_ghana_zscore_alerts'] = daily_fixture()
        app.session_state['daily_ghana_zscore_alert_timeframe'] = '1h'
        app.run().button(key='qqq_nq_generate').click().run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.error), 0)
        self.assertEqual(len(app.dataframe), 2)
        table = app.dataframe[-1].value
        self.assertTrue(table['VIX Daily Z'].eq(-1).all())
        self.assertTrue(table['QQQ Daily Z'].eq(1).all())
        self.assertTrue(table['Daily Alert Confirmation'].eq('BUY').all())
        self.assertTrue(table['Signal'].eq('BUY').any())
        app.session_state['daily_ghana_zscore_alert_timeframe'] = '15m'
        app.run()
        self.assertTrue(app.dataframe[-1].value['Signal'].eq('WAIT').all())
        self.assertEqual(len(app.session_state['calls']), 2)


if __name__ == '__main__':
    unittest.main()

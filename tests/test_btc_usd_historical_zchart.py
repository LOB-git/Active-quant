"""Regression coverage for BTC/USD in US Indices historical component analysis."""

import ast
from datetime import datetime, timedelta
import os
from pathlib import Path
import unittest

import numpy as np
import pandas as pd


SOURCE = Path(__file__).resolve().parents[1] / 'Untitled-1.py'


class FakeYahoo:
    def __init__(self, frame):
        self.frame = frame
        self.calls = []

    def download(self, symbol, **kwargs):
        self.calls.append((symbol, kwargs))
        return self.frame.copy()

    def Ticker(self, symbol):  # pragma: no cover - download is populated in this test
        raise AssertionError('The populated Yahoo response should not need a fallback ticker call')


def yahoo_frame():
    index = pd.date_range('2026-09-01', periods=240, freq='h', tz='UTC')
    values = np.arange(len(index), dtype=float) + 100_000
    base = pd.DataFrame({
        'Open': values, 'High': values + 20, 'Low': values - 20,
        'Close': values + 5, 'Volume': np.full(len(index), 3.),
    }, index=index)
    base.columns = pd.MultiIndex.from_product([base.columns, ['BTC-USD']])
    return base


def historical_functions(fake_yahoo):
    tree = ast.parse(SOURCE.read_text(encoding='utf-8'))
    definitions = []
    wanted = {'fetch_yahoo_ohlcv', 'fetch_and_analyze'}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in wanted:
            node.decorator_list = []
            definitions.append(node)
    namespace = {
        'pd': pd, 'np': np, 'yf': fake_yahoo, 'datetime': datetime,
        'timedelta': timedelta, 'fetch_fear_and_greed_history': lambda: None,
        'ccxt': None, 'os': os,
        'YFINANCE_CACHE_DIR': str(SOURCE.parent / '.yfinance-cache-test'),
    }
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace


class BtcUsdHistoricalChartTests(unittest.TestCase):
    def test_btc_usd_uses_yahoo_not_binance_and_has_zscore_warmup(self):
        yahoo = FakeYahoo(yahoo_frame())
        functions = historical_functions(yahoo)
        result = functions['fetch_and_analyze']('BTC/USD', timeframe='1h', silent=True)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(len(result), 200)
        self.assertEqual(yahoo.calls[0][0], 'BTC-USD')
        self.assertEqual(yahoo.calls[0][1]['interval'], '60m')
        self.assertTrue({'open', 'high', 'low', 'close', 'volume', 'momentum', 'atr14', 'z_score'}.issubset(result.columns))
        self.assertGreater(result['z_score'].notna().sum(), 0)

    def test_btc_usd_12h_is_resampled_from_hourly_yahoo_candles(self):
        yahoo = FakeYahoo(yahoo_frame())
        functions = historical_functions(yahoo)
        candles = functions['fetch_yahoo_ohlcv']('BTC-USD', timeframe='12h')
        self.assertEqual(yahoo.calls[0][0], 'BTC-USD')
        self.assertEqual(yahoo.calls[0][1]['interval'], '60m')
        self.assertEqual(len(candles), 20)
        self.assertEqual(candles.iloc[0]['volume'], 36.)
        self.assertIsNone(candles.index.tz)


if __name__ == '__main__':
    unittest.main()

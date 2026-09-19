import ccxt, requests, time

exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
print('Using exchange', exchange.id)
try:
    markets = exchange.fetch_markets()
    swap_symbols = [m.get('symbol') for m in markets if m.get('symbol') and ((m.get('type') or '').lower() == 'swap' or m.get('contract') is True or m.get('symbol').endswith(':USDT') or m.get('symbol').endswith('/USDT'))]
    print('Found swap symbols:', len(swap_symbols))
    # group by base
    market_map = {m.get('symbol'): m for m in markets if m.get('symbol')}
    from collections import defaultdict
    groups = defaultdict(list)
    for s in swap_symbols:
        m = market_map.get(s, {})
        base = m.get('base') or (m.get('info') or {}).get('baseCoin') or ''
        mtype = (m.get('type') or m.get('info', {}).get('type') or 'swap').lower()
        groups[(mtype, base)].append(s)

    tickers = {}
    for (mtype, base), syms in groups.items():
        for i in range(0, len(syms), 80):
            batch = syms[i:i+80]
            try:
                part = exchange.fetch_tickers(batch)
                tickers.update(part or {})
            except Exception as e:
                # try per-symbol
                for s in batch:
                    try:
                        t = exchange.fetch_ticker(s)
                        if t:
                            tickers[s] = t
                    except Exception:
                        pass
            time.sleep(0.12)
    print('Collected tickers:', len(tickers))
    sample = list(tickers.keys())[:20]
    print('Sample:', sample)
except Exception as e:
    print('Error:', e)
    try:
        r = requests.get('https://api.bybit.com/v5/market/instruments-info?category=linear', timeout=10)
        print('HTTP fallback status', r.status_code)
    except Exception as e2:
        print('HTTP fallback failed', e2)

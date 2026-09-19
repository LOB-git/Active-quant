import ccxt
import traceback

def main():
    exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    try:
        markets = exchange.fetch_markets()
        print(f"Fetched {len(markets)} markets from {exchange.id}")
        swap_symbols = [m.get('symbol') for m in markets if m.get('symbol') and ((m.get('type') or '').lower() == 'swap' or m.get('contract') is True or m.get('symbol').endswith(':USDT') or m.get('symbol').endswith('/USDT'))]
        print(f"Identified {len(swap_symbols)} swap symbols (showing first 20): {swap_symbols[:20]}")

        # Group symbols by type and base currency, then fetch in safe batches
        market_map = {m.get('symbol'): m for m in markets if m.get('symbol')}
        from collections import defaultdict
        groups = defaultdict(list)
        for s in swap_symbols:
            m = market_map.get(s, {})
            mtype = (m.get('type') or m.get('info', {}).get('type') or 'swap').lower()
            base = m.get('base') or (m.get('info') or {}).get('baseCoin') or ''
            groups[(mtype, base)].append(s)

        tickers = {}
        import time
        for (mtype, base), syms in groups.items():
            print(f"Fetching {len(syms)} symbols for type={mtype} base={base}")
            batch_size = 80
            for i in range(0, len(syms), batch_size):
                batch = syms[i:i+batch_size]
                try:
                    part = exchange.fetch_tickers(batch)
                    print(f"  fetched batch {i}-{i+len(batch)} -> {len(part)} tickers")
                    tickers.update(part)
                except Exception as e:
                    print('  batch fetch failed:', e)
                    traceback.print_exc()
                    try:
                        all_t = exchange.fetch_tickers()
                        for k, v in all_t.items():
                            if k in batch:
                                tickers[k] = v
                        print(f"  fallback filtered from all tickers for batch -> {len([k for k in batch if k in all_t])} hit")
                    except Exception as e2:
                        print('  fetch_tickers() fallback failed:', e2)
                        traceback.print_exc()
                time.sleep(0.15)

        print(f"Total tickers collected: {len(tickers)}")

        print(f"Total tickers collected: {len(tickers)}")
    except Exception as e:
        print('fetch_markets failed:', e)
        traceback.print_exc()

if __name__ == '__main__':
    main()

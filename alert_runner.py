"""Background Telegram alert runner for the Quant Scalper dashboard.

Runs the dashboard's overview Z-score alert checks on a timer, without a
browser or Streamlit page open. It reuses the dashboard's own functions
(fetch_and_analyze, send_telegram_alert, build_overview_row,
new_threshold_alerts), so alerts match what the dashboard would send.

It also saves the Top 200 Risk & Leverage Watch List to watchlist_history.csv
every hour (the "watchlist" group), so each day has a full record even when the
dashboard is closed. That part works without Telegram details.

The "gex" group saves an hourly BTC/ETH gamma-exposure snapshot from Deribit to
gex_history.csv (no Telegram needed), building the history GEX needs to be tested.

The "session_flow" group records, every 5 minutes, each trading session's (Asia from the
New York close, London, New York) Highest Flow Z Rise and Drop for BTC/USD and the top crypto
perpetuals to session_flow_history.csv (no Telegram needed).

The "zone_flow_tracker" group forward-tests Zone Battle + Highest Flow Z Rise/Drop signals on
BTC/USDT and BTC/USD across 5m, 15m, 1h, 4h and 1d every 5 minutes, logging only signals made
after tracking began to zone_flow_tracker_log.csv (no Telegram needed).

Usage:
    python alert_runner.py            # run forever
    python alert_runner.py --once     # one check, then exit
    python alert_runner.py --test     # send a test message to Telegram

Telegram credentials come from environment variables TELEGRAM_BOT_TOKEN and
TELEGRAM_CHAT_ID, or from alert_secrets.env next to this file:
    TELEGRAM_BOT_TOKEN=123456:ABC...
    TELEGRAM_CHAT_ID=123456789
Settings (groups, timeframe, thresholds, interval) live in alert_config.json.
"""
import argparse
import importlib.util
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, 'alert_config.json')
SECRETS_PATH = os.path.join(HERE, 'alert_secrets.env')
STATE_PATH = os.path.join(HERE, '.alert_runner_state.json')
LOG_PATH = os.path.join(HERE, 'alert_runner.log')

DEFAULT_CONFIG = {
    'interval_seconds': 300,
    'groups': {
        'us_indices': {
            'enabled': True,
            'timeframe': '1h',
            'z_threshold': 1.0,
            'symbols': ['SPY', 'QQQ', 'NQ=F', 'DIA', 'YM=F', '^VIX', 'DX-Y.NYB'],
        },
        'derivatives': {
            'enabled': True,
            'timeframe': '1h',
            'z_threshold': 1.0,
            'symbols': None,  # None = the dashboard's derivative overview list
        },
        # Saves the Risk & Leverage Watch List to watchlist_history.csv. No Telegram needed.
        'watchlist': {
            'enabled': True,
            'every_seconds': 3600,
            'timeframe': '1h',
            'scan_size': 200,
            'risk_usd': 10.0,
            'max_leverage': 0.4,
            # Quant Edge Settings, as in the dashboard. equity x risk_pct gives the $ risk.
            'equity': 1000.0,
            'risk_pct': 1.0,
            'atr_multiple': 1.5,
            'reward_risk': 2.0,
            'threshold': 0.25,
            'max_hold': 48,
            'fee_bps': 10.0,
        },
        # Saves BTC/ETH gamma exposure from Deribit to gex_history.csv. No Telegram needed.
        'gex': {
            'enabled': True,
            'every_seconds': 3600,
            'currencies': ['BTC', 'ETH'],
        },
        # Saves 30-day implied volatility for coins without a DVOL index to iv_snapshots.csv. No Telegram needed.
        'implied_vol': {
            'enabled': True,
            'every_seconds': 3600,
            'coins': ['SOL', 'XRP', 'AVAX', 'HYPE', 'TRX'],
        },
        # Records each session's Highest Flow Z Rise/Drop to session_flow_history.csv. No Telegram needed.
        'session_flow': {
            'enabled': True,
            'every_seconds': 300,
            'timeframe': '5m',
            'symbols': ['BTC/USD'],   # Yahoo BTC-USD, as in the US Indices tab
            'top_perpetuals': 10,     # plus the top N Binance crypto perpetuals by 24h volume
        },
        # Forward test of Zone Battle + Highest Flow Z Rise/Drop on every timeframe. No Telegram needed.
        'zone_flow_tracker': {
            'enabled': True,
            'every_seconds': 300,
            'symbols': ['BTC/USDT', 'BTC/USD', 'BTC/USDT (real flow)'],
            'timeframes': ['5m', '15m', '1h', '4h', '1d'],
        },
    },
}
TELEGRAM_GROUPS = {'us_indices', 'derivatives'}
QUANT_SETTING_KEYS = ('equity', 'risk_pct', 'atr_multiple', 'reward_risk', 'threshold', 'max_hold', 'fee_bps')

log = logging.getLogger('alert_runner')


def setup_logging():
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s', '%Y-%m-%d %H:%M:%S')
    log.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_PATH, encoding='utf-8')):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    # Streamlit warns about every st.* call made outside `streamlit run`; those are expected here.
    for noisy in ('streamlit', 'streamlit.runtime', 'yfinance', 'peewee'):
        logging.getLogger(noisy).setLevel(logging.ERROR)


def load_config():
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding='utf-8') as fh:
            user = json.load(fh)
        config['interval_seconds'] = user.get('interval_seconds', config['interval_seconds'])
        for name, group in user.get('groups', {}).items():
            config['groups'].setdefault(name, {}).update(group)
        if config != user:  # New settings were added in an update: show them in the file.
            with open(CONFIG_PATH, 'w', encoding='utf-8') as fh:
                json.dump(config, fh, indent=2)
            log.info('Added new default settings to %s', CONFIG_PATH)
    else:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as fh:
            json.dump(DEFAULT_CONFIG, fh, indent=2)
        log.info('Wrote default settings to %s', CONFIG_PATH)
    return config


def load_secrets():
    values = {}
    if os.path.exists(SECRETS_PATH):
        with open(SECRETS_PATH, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    values[key.strip()] = value.strip().strip('"').strip("'")
    token = os.environ.get('TELEGRAM_BOT_TOKEN') or values.get('TELEGRAM_BOT_TOKEN', '')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID') or values.get('TELEGRAM_CHAT_ID', '')
    return token, chat_id


def load_state():
    try:
        with open(STATE_PATH, encoding='utf-8') as fh:
            data = json.load(fh)
        return {name: {tuple(key) for key in keys} for name, keys in data.items()}
    except (OSError, ValueError):
        return {}


def save_state(state):
    # Keep the most recent keys only; old candles can never alert again.
    data = {name: sorted(keys, key=lambda k: k[2])[-500:] for name, keys in state.items()}
    tmp = STATE_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(data, fh)
    os.replace(tmp, STATE_PATH)


def load_dashboard():
    """Import dashboard.py as a module. Its main() only runs under `streamlit run`."""
    sys.path.insert(0, HERE)
    spec = importlib.util.spec_from_file_location('quant_dashboard', os.path.join(HERE, 'dashboard.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Streamlit configures its own loggers on import; silence its bare-mode warnings.
    for name in list(logging.root.manager.loggerDict):
        if name.startswith('streamlit'):
            logging.getLogger(name).setLevel(logging.ERROR)
    return module


def telegram_post(token, chat_id, text):
    import requests
    response = requests.post(
        f'https://api.telegram.org/bot{token}/sendMessage',
        json={'chat_id': chat_id, 'text': text}, timeout=15,
    )
    if not response.ok:
        raise RuntimeError(f'Telegram returned {response.status_code}: {response.text[:200]}')


def check_group(name, group, dash, panel, token, chat_id, seen):
    timeframe = group['timeframe']
    threshold = float(group['z_threshold'])
    if name == 'derivatives':
        symbols = group.get('symbols') or panel.OVERVIEW_SYMBOLS
        fetch = lambda sym: dash.fetch_and_analyze(sym, timeframe=timeframe, silent=True, market='usdm')  # noqa: E731
        price_text = panel.format_price
    else:
        symbols = group['symbols']
        fetch = lambda sym: dash.fetch_and_analyze(sym, timeframe=timeframe, silent=True)  # noqa: E731
        price_text = None

    rows, missing = [], []
    for symbol in symbols:
        try:
            candles = fetch(symbol)
        except Exception as exc:  # one bad symbol must not stop the rest
            log.warning('%s: %s fetch failed: %s', name, symbol, exc)
            candles = None
        if candles is None or candles.empty or len(candles) < 2:
            missing.append(symbol)
            continue
        rows.append(panel.build_overview_row(symbol, candles, timeframe))

    fresh = panel.new_threshold_alerts(rows, threshold, timeframe, seen)
    for alert in fresh:
        price = price_text(alert['Price']) if price_text else alert['Price']
        dash.send_telegram_alert(
            token, chat_id, alert['Symbol'], alert['Alert'], price,
            f"Z-Score {alert['Z-Score']:+.2f}",
            [f'Threshold: +/-{threshold:.2f}',
             f"Z-Score Change: {alert['Z-Score Change']:+.2f}",
             f'Timeframe: {timeframe}',
             f"Candle: {alert['Candle Time']:%Y-%m-%d %H:%M} UTC",
             f'Source: {name.replace("_", " ")} (background runner)'],
        )
        log.info('ALERT %s %s %s z=%+.2f', name, alert['Symbol'], alert['Alert'], alert['Z-Score'])
    log.info('%s: %d checked, %d new alert(s)%s', name, len(rows), len(fresh),
             f", no data for {', '.join(missing)}" if missing else '')
    return len(fresh)


def save_watchlist(group, dash, panel, dry_run=False):
    """Scan the top Binance perpetuals and store today's Risk & Leverage Watch List matches."""
    import pandas as pd
    import watchlist_history
    from binance_futures import top_usdm_symbols

    timeframe = group['timeframe']
    settings = {key: float(group[key]) for key in QUANT_SETTING_KEYS}
    settings['max_hold'] = int(settings['max_hold'])
    risk_usd, max_leverage = float(group['risk_usd']), float(group['max_leverage'])
    symbols = top_usdm_symbols(int(group['scan_size']))
    fetch = lambda sym, **options: dash.fetch_and_analyze(sym, market='usdm', **options)  # noqa: E731
    scanned_at = pd.Timestamp.now(tz='UTC')
    table, failed, _ = panel.scan_quant_edges(symbols, fetch, timeframe, settings)
    table = panel.rank_quant_scan(table)
    matches, funnel = panel.risk_leverage_funnel(table, risk_usd, max_leverage)
    log.info('watchlist: %d/%d scanned; %s', len(table), len(symbols),
             ' -> '.join(f'{label}: {count}' for label, count in funnel))
    if dry_run:
        log.info('[dry-run] would save: %s', ', '.join(matches['Symbol']) or 'nothing')
        return 0
    saved = watchlist_history.record_matches(matches, scanned_at, timeframe, risk_usd, max_leverage)
    log.info('watchlist: saved %d match(es) for %s: %s', saved,
             scanned_at.tz_convert(watchlist_history.GHANA_TZ).strftime('%Y-%m-%d'),
             ', '.join(matches['Symbol']) or 'none')
    return saved


def save_gex(group, dry_run=False):
    """Record this hour's GEX snapshot for each currency."""
    import gex_panel

    if dry_run:
        log.info('[dry-run] would save GEX snapshots for %s', ', '.join(group['currencies']))
        return
    results = gex_panel.log_gex_snapshots(group['currencies'], source='runner')
    log.info('gex: %s', ', '.join(f'{currency} {status}' for currency, status in results.items()))


def save_implied_vol(group, dry_run=False):
    """Save each coin's live 30-day implied volatility (Deribit keeps no history for these coins)."""
    import alt_iv

    if dry_run:
        log.info('[dry-run] would save implied volatility for %s', ', '.join(group['coins']))
        return
    results = {}
    for coin in group['coins']:
        try:
            reading = alt_iv.live_iv(coin)
            saved = alt_iv.record_snapshot(coin, reading)
            results[coin] = f"{reading['iv']:.1%}" + ('' if saved else ' (recent one kept)') if reading else 'no data'
        except Exception as exc:
            results[coin] = f'failed: {exc}'
    log.info('implied_vol: %s', ', '.join(f'{coin} {status}' for coin, status in results.items()))


def save_session_flow(group, dash, dry_run=False):
    """Record the current and previous trading day's session Highest Flow Z Rise/Drop."""
    import session_flow_recorder
    from session_flow_panel import session_symbols, session_fetcher

    symbols = session_symbols(group['symbols'], int(group['top_perpetuals']))
    if dry_run:
        log.info('[dry-run] would record session flow for %s', ', '.join(symbols))
        return
    results = session_flow_recorder.record_sessions(
        symbols, session_fetcher(dash.fetch_and_analyze, group['timeframe']), group['timeframe'])
    failed = {s: r for s, r in results.items() if r.startswith('failed') or r.startswith('no ')}
    log.info('session_flow: recorded %d/%d symbols%s', len(results) - len(failed), len(results),
             f"; problems: {failed}" if failed else '')


def save_zone_flow_tracker(group, dash, dry_run=False):
    """Log new confirmed Zone Battle + Flow signals for every symbol and timeframe."""
    import zone_flow_tracker

    if dry_run:
        log.info('[dry-run] would update the zone flow tracker')
        return
    results = zone_flow_tracker.update_all(dash.fetch_and_analyze, dash.calculate_zone_battle_scores,
                                           group['symbols'], group['timeframes'])
    new = sum(int(r.split()[0]) for r in results.values() if r.endswith('signal(s)'))
    problems = {f'{s} {tf}': r for (s, tf), r in results.items() if not r.endswith('signal(s)')}
    log.info('zone_flow_tracker: %d new signal(s)%s', new, f'; problems: {problems}' if problems else '')


def run_cycle(config, dash, panel, token, chat_id, state, last_run=None, dry_run=False):
    """Run every enabled group that is due. Groups with every_seconds run at that pace."""
    total, last_run = 0, last_run if last_run is not None else {}
    now = time.monotonic()
    for name, group in config['groups'].items():
        if not group.get('enabled'):
            continue
        every = group.get('every_seconds')
        if every and name in last_run and now - last_run[name] < float(every):
            continue
        if name in TELEGRAM_GROUPS and not (token and chat_id):
            continue
        last_run[name] = now
        try:
            if name == 'watchlist':
                save_watchlist(group, dash, panel, dry_run)
            elif name == 'gex':
                save_gex(group, dry_run)
            elif name == 'implied_vol':
                save_implied_vol(group, dry_run)
            elif name == 'session_flow':
                save_session_flow(group, dash, dry_run)
            elif name == 'zone_flow_tracker':
                save_zone_flow_tracker(group, dash, dry_run)
            else:
                total += check_group(name, group, dash, panel, token, chat_id, state.setdefault(name, set()))
        except Exception:
            log.exception('%s: check failed', name)
    save_state(state)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--once', action='store_true', help='run one check and exit')
    parser.add_argument('--test', action='store_true', help='send a Telegram test message and exit')
    parser.add_argument('--dry-run', action='store_true',
                        help='log alerts instead of sending them, and do not record them as sent')
    args = parser.parse_args()

    setup_logging()
    token, chat_id = load_secrets()
    if args.dry_run:
        token, chat_id = token or 'dry-run', chat_id or 'dry-run'

    if args.test:
        if not token or not chat_id:
            log.error('Telegram token/chat ID missing. Put TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in %s', SECRETS_PATH)
            return 2
        telegram_post(token, chat_id, f'Quant alert runner test from {os.uname().nodename if hasattr(os, "uname") else "PC"} '
                                      f'at {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC')
        log.info('Test message sent.')
        return 0

    config = load_config()
    enabled = [n for n, g in config['groups'].items() if g.get('enabled')]
    if not (token and chat_id):
        skipped = [n for n in enabled if n in TELEGRAM_GROUPS]
        enabled = [n for n in enabled if n not in TELEGRAM_GROUPS]
        if skipped:
            log.warning('No Telegram token/chat ID in %s: skipping %s. Watch list, GEX, session flow and zone flow tracking still run.',
                        SECRETS_PATH, ', '.join(skipped))
    if not enabled:
        log.error('Nothing to run. Add Telegram details to %s or enable the watchlist, gex, session_flow or zone_flow_tracker group.', SECRETS_PATH)
        return 2
    log.info('Loading dashboard functions...')
    dash = load_dashboard()
    import derivative_crypto_panel as panel
    if args.dry_run:
        dash.send_telegram_alert = lambda *a, **k: log.info('[dry-run] would send: %s %s price=%s %s', a[2], a[3], a[4], a[5])
        global save_state
        save_state = lambda state: None  # noqa: E731
    state, last_run = load_state(), {}
    log.info('Runner started: groups=%s, checking every %ss', ', '.join(enabled), config['interval_seconds'])

    while True:
        started = time.monotonic()
        run_cycle(config, dash, panel, token, chat_id, state, last_run, args.dry_run)
        if args.once:
            return 0
        # fetch_and_analyze caches for 300 s; clear it so each cycle sees fresh candles.
        try:
            dash.fetch_and_analyze.clear()
        except Exception:
            pass
        time.sleep(max(30, config['interval_seconds'] - (time.monotonic() - started)))


if __name__ == '__main__':
    sys.exit(main())

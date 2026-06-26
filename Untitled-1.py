import streamlit as st
import ccxt
import requests
import pandas as pd
import yfinance as yf
import numpy as np
import time
import concurrent.futures
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg') # Fix for Streamlit/Matplotlib GUI errors
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import streamlit.components.v1 as components


# Set page config at the top level to avoid errors and define layout
st.set_page_config(page_title="Quant Scalper 1h", layout="wide")

@st.cache_data
def fetch_fear_and_greed_history():
    """
    Fetches historical Fear and Greed Index data (Proxy for Market News Sentiment).
    """
    url = "https://api.alternative.me/fng/?limit=0"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()['data']
        df = pd.DataFrame(data)
        df['value'] = pd.to_numeric(df['value'])
        # Convert timestamp to datetime
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.set_index('date').sort_index()
        return df[['value']]
    except Exception as e:
        return None

@st.cache_data(ttl=300)
def fetch_and_analyze(symbol='BTC/USDT', timeframe='1h', start_date=None, end_date=None, silent=False):
    """
    Fetches Crypto Data (via ccxt/Binance) and calculates Multi-Strategy Factors.
    """
    try:
        stock_index_symbols = ['SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM', 'GBPUSD=X', '^FTSE', 'XAUUSD=X']
        is_stock_index = symbol in stock_index_symbols

        # Handle Symbol Formatting (e.g., BTC-USD -> BTC/USDT for Binance)
        if not is_stock_index and symbol.endswith('-USD'):
            symbol = symbol.replace('-USD', '/USDT')
        elif not is_stock_index and '-' in symbol:
            symbol = symbol.replace('-', '/')
        
        df = pd.DataFrame()
        
        # Check if symbol is a Stock/ETF to fetch via yfinance
        if is_stock_index:
            # Fetch Stock Data via yfinance
            print(f"Fetching data for {symbol} via yfinance...")
            # yfinance uses minute-based intervals for intraday data; map/hourly intervals where needed
            yf_interval = timeframe
            if timeframe == '1h':
                yf_interval = '60m'
            # For 4-hour, fetch 60m data then resample to 4H below
            if timeframe == '4h':
                yf_interval = '60m'

            # Choose period: intraday intervals are limited to ~60 days of history
            intraday_intervals = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '4h']

            # yfinance intraday data is limited to the last 60 days.
            # If an intraday timeframe is selected with a start_date older than that,
            # yfinance will fail. We adjust the start_date if necessary.
            effective_start_date = start_date
            if start_date and timeframe in intraday_intervals:
                sixty_days_ago = (datetime.now() - timedelta(days=59)).strftime('%Y-%m-%d')
                if start_date < sixty_days_ago:
                    effective_start_date = sixty_days_ago

            if start_date:
                try:
                    df = yf.download(symbol, start=effective_start_date, interval=yf_interval, progress=False, prepost=True)
                except Exception:
                    df = pd.DataFrame()
            else:
                period = '60d' if timeframe in intraday_intervals else '2y'
                try:
                    df = yf.download(symbol, period=period, interval=yf_interval, progress=False, prepost=True)
                except Exception:
                    df = pd.DataFrame()
                if df.empty:
                    df = yf.Ticker(symbol).history(period=period, interval=yf_interval, prepost=True) # Fallback
            
            # Flatten MultiIndex columns (yfinance v0.2+)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                
            # Normalize columns
            df.columns = [c.lower() for c in df.columns]
            df.index.name = 'date'
            
            # Strip timezone to avoid merge_asof crashes with Fear & Greed data
            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_convert('UTC').tz_localize(None)
            # If we fetched 60m data but the user requested 4h, resample to 4H candles
            if timeframe == '4h' and not df.empty:
                df = df.resample('4H').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()
            
        else:
            # Fetch Crypto Data via Binance (CCXT)
            print(f"Fetching data for {symbol} via Binance...")
            exchange = ccxt.binance({'enableRateLimit': True})
            limit = 1000 # Binance API limit per request
            all_ohlcv = []

            if start_date:
                # Backtesting mode: Fetch all data since start_date in a loop
                since = exchange.parse8601(start_date + 'T00:00:00Z')
                while True:
                    print(f"Fetching historical data chunk for {symbol} since {datetime.fromtimestamp(since/1000)}...")
                    ohlcv_chunk = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                    if not ohlcv_chunk:
                        break # No more data
                    
                    all_ohlcv.extend(ohlcv_chunk)
                    
                    if len(ohlcv_chunk) < limit:
                        break # Reached the end of available history for the period
                    
                    # Set 'since' to the timestamp of the last candle + 1ms for the next chunk
                    since = ohlcv_chunk[-1][0] + 1 
            else:
                # Live analysis mode: Fetch latest N candles
                print(f"Fetching latest {limit} candles for live analysis of {symbol}...")
                all_ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if all_ohlcv:
                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                df = df.set_index('date')

        if df.empty:
             if not silent:
                 st.warning(f"No data returned for {symbol}. Check if the ticker is correct (e.g., BTC/USDT) and internet connection.")
             return None
            
        # Remove potential duplicates from overlapping fetches and sort
        if 'timestamp' in df.columns:
            df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
        else:
            df = df[~df.index.duplicated(keep='first')]
        df.sort_index(inplace=True) # Ensure data is chronological
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

        # --- STRATEGY 1: Trend Following (Moving Averages) ---
        df['sma_200'] = df['close'].rolling(window=200).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # --- STRATEGY 2: Mean Reversion (Bollinger Bands & RSI) ---
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (2 * df['bb_std'])
        df['bb_lower'] = df['bb_mid'] - (2 * df['bb_std'])

        # --- STRATEGY 3: Statistical Arbitrage (Simplified) ---
        # Using Volatility (Standard Deviation) as a proxy for mean-reverting regimes
        df['volatility'] = df['close'].rolling(window=20).std()
        df['z_score'] = (df['close'] - df['bb_mid']) / df['bb_std']

        # --- STRATEGY 4: Factor Investing (Momentum) ---
        # Rate of Change (30 days for Crypto)
        df['momentum'] = df['close'].pct_change(periods=30)
        
        # --- STRATEGY 5: Sentiment Analysis (High Impact News Proxy) ---
        # We merge Fear & Greed index. If news is bad (Fear), we look for bottoms.
        fg_df = fetch_fear_and_greed_history()
        if fg_df is not None:
            # Merge using merge_asof to align daily sentiment with candle times
            # We align backward to use the most recent known sentiment value
            df = pd.merge_asof(df, fg_df, left_index=True, right_index=True, direction='backward')
            df.rename(columns={'value': 'sentiment'}, inplace=True)
        else:
            df['sentiment'] = 50 # Default Neutral if API fails

        # --- STRATEGY 6: Institutional Order Flow (Chaikin Money Flow) ---
        # CMF measures buying/selling pressure over a period
        ad = np.where(df['high'] == df['low'], 0, (2 * df['close'] - df['high'] - df['low']) / (df['high'] - df['low']))
        df['vol_ad'] = ad * df['volume']
        df['cmf'] = df['vol_ad'].rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

        # --- STRATEGY 7: Order Blocks (Support/Resistance Zones) ---
        # We identify the lowest low (Demand/Bullish Block) and highest high (Supply/Bearish Block)
        # of the last 50 candles to find where institutions placed orders.
        df['ob_bull'] = df['low'].rolling(window=50).min().shift(1) # Recent swing low
        df['ob_bear'] = df['high'].rolling(window=50).max().shift(1) # Recent swing high
        
        # --- STRATEGY 8: Volume Spikes (Squeeze Detection) ---
        df['vol_ma'] = df['volume'].rolling(window=20).mean()
        
        # --- STRATEGY 9: ADX (Trend Strength & Regime Filter) ---
        # ADX identifies if the market is trending or chopping
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
        df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        df['up_move'] = df['high'] - df['high'].shift(1)
        df['down_move'] = df['low'].shift(1) - df['low']
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        # Wilder's Smoothing (Window 14)
        df['tr_s'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
        df['plus_dm_s'] = df['plus_dm'].ewm(alpha=1/14, adjust=False).mean()
        df['minus_dm_s'] = df['minus_dm'].ewm(alpha=1/14, adjust=False).mean()
        df['plus_di'] = 100 * (df['plus_dm_s'] / df['tr_s'])
        df['minus_di'] = 100 * (df['minus_dm_s'] / df['tr_s'])
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].ewm(alpha=1/14, adjust=False).mean()

        # --- STRATEGY 10: Fair Value Gaps (FVG) ---
        # Identifying imbalances (gaps) left by aggressive orders
        # Bullish FVG: Low of candle i > High of candle i-2
        df['is_fvg_bull'] = (df['low'] > df['high'].shift(2)) & (df['close'] > df['open'])
        df['is_fvg_bear'] = (df['high'] < df['low'].shift(2)) & (df['close'] < df['open'])
        
        # Track the most recent FVG zones (Forward Fill)
        df['last_bull_fvg_top'] = df['low'].where(df['is_fvg_bull']).ffill()
        df['last_bull_fvg_bottom'] = df['high'].shift(2).where(df['is_fvg_bull']).ffill()
        df['last_bear_fvg_bottom'] = df['high'].where(df['is_fvg_bear']).ffill()
        df['last_bear_fvg_top'] = df['low'].shift(2).where(df['is_fvg_bear']).ffill()
        
        # Fill NaN values created by rolling windows
        df['cmf'] = df['cmf'].fillna(0)
        df['ob_bull'] = df['ob_bull'].fillna(df['low'])
        df['ob_bear'] = df['ob_bear'].fillna(df['high'])
        df['adx'] = df['adx'].fillna(0)
        df['last_bull_fvg_top'] = df['last_bull_fvg_top'].fillna(0)
        df['last_bull_fvg_bottom'] = df['last_bull_fvg_bottom'].fillna(0)
        df['last_bear_fvg_top'] = df['last_bear_fvg_top'].fillna(10000000)
        df['last_bear_fvg_bottom'] = df['last_bear_fvg_bottom'].fillna(10000000)

        # --- STRATEGY 11: Rising Momentum & Volume (User Request) ---
        # 12-Hour Volume (Assuming 1h candles, 12h = 12 periods)
        # We check if the total volume traded in the last 12 hours is increasing
        df['vol_12h'] = df['volume'].rolling(window=12).sum()
        df['vol_12h_prev'] = df['vol_12h'].shift(1)
        df['momentum_prev'] = df['momentum'].shift(1)
        
        return df

    except Exception as e:
        if not silent:
            st.error(f"Data Fetch Error: {e}")
        print(f"Error fetching data for {symbol}: {e}")
        return None

def generate_signal(df, symbol, start_hour=0, end_hour=24):
    """
    Generates a Composite Score based on the 5 quantitative strategies.
    """
    if df is None or df.empty:
        return
    
    # Get the latest data point
    current = df.iloc[-1]
    
    # Logic Variables
    price = current['close']
    ema_50 = current['ema_50']
    rsi = current['rsi']
    z_score = current['z_score']
    momentum = current['momentum']
    sentiment = current.get('sentiment', 50)
    cmf = current['cmf']
    ob_bull = current['ob_bull']
    ob_bear = current['ob_bear']
    volume = current['volume']
    vol_ma = current['vol_ma']
    adx = current['adx']
    vol_12h = current.get('vol_12h', 0)
    vol_12h_prev = current.get('vol_12h_prev', 0)
    momentum_prev = current.get('momentum_prev', 0)
    
    # FVG Zones
    price = current['close']
    in_bull_fvg = (price <= current['last_bull_fvg_top']) and (price >= current['last_bull_fvg_bottom'])
    in_bear_fvg = (price >= current['last_bear_fvg_bottom']) and (price <= current['last_bear_fvg_top'])
    
    # --- TIME FILTER ---
    # Check if current candle is within active trading hours
    current_hour = current.name.hour
    is_active_time = False
    if start_hour <= end_hour:
        is_active_time = (start_hour <= current_hour < end_hour)
    else: # Spans midnight (e.g., 22:00 to 06:00)
        is_active_time = (current_hour >= start_hour or current_hour < end_hour)


    # --- COMPOSITE SCORING SYSTEM ---
    score = 0
    reasons = []
    
    # 1. Trend Following (+1 if Bullish)
    if price > ema_50:
        score += 1
        reasons.append("Trend Bullish")
    else:
        score -= 1
        reasons.append("Trend Bearish")
        
    # 2. Mean Reversion (+1 if Oversold, -1 if Overbought)
    if rsi < 30:
        score += 1
        reasons.append("RSI Oversold")
    elif rsi > 70:
        score -= 1
        reasons.append("RSI Overbought")
        
    # 3. Statistical Arbitrage / Mean Rev (Z-Score extreme)
    if z_score < -2:
        score += 1
        reasons.append("Price < 2 StdDev (Cheap)")
    elif z_score > 2:
        score -= 1
        reasons.append("Price > 2 StdDev (Expensive)")
        
    # 4. Factor Investing (Momentum)
    if momentum > 0:
        score += 0.5
        reasons.append("Pos Momentum")
    else:
        score -= 0.5
        reasons.append("Neg Momentum")
    
    # 5. Sentiment / News Analysis
    # Buy when others are fearful (Bad News Overreaction)
    if sentiment < 20:
        score += 1
        reasons.append(f"Extreme Fear ({sentiment})")
    # Sell when others are greedy (Good News Hype)
    elif sentiment > 80:
        score -= 1
        reasons.append(f"Extreme Greed ({sentiment})")
        
    # 6. Order Flow (CMF) - Tracking Smart Money
    if cmf > 0.1:
        score += 1
        reasons.append(f"Inst. Accumulation (CMF {cmf:.2f})")
    elif cmf < -0.1:
        score -= 1
        reasons.append(f"Inst. Distribution (CMF {cmf:.2f})")
        
    # 7. Order Blocks (Re-testing Liquidity Zones)
    # If price is within 1% of the Bullish Order Block (Support)
    if 0 <= (price - ob_bull) / price <= 0.01:
        score += 1.5 # High weight for support bounces
        reasons.append("Testing Bullish Order Block")
    elif 0 <= (ob_bear - price) / price <= 0.01:
        score -= 1.5 # High weight for resistance rejection
        reasons.append("Testing Bearish Order Block")
        
    # 8. Short Squeeze / Blow-off Top Detector (Fade the "Fake" Spike)
    # Logic: Price > 3 StdDevs (Extreme) + RSI Hot + Massive Volume Spike = Liquidation Wick
    if z_score > 3 and rsi > 75 and volume > (vol_ma * 3):
        score -= 2.0 # Strong Sell signal (expecting rapid reversion)
        reasons.append("Short Squeeze / Fake Pump Detected")
        
    # 9. Regime Filter (ADX)
    if adx > 25:
        score *= 1.1 # Amplify score if trend is strong
        reasons.append(f"Strong Trend (ADX {adx:.1f})")
    elif adx < 20:
        score *= 0.5 # Reduce score confidence in chop
        reasons.append(f"Weak Trend/Chop (ADX {adx:.1f})")
        
    # 10. Fair Value Gaps (Sniper Entries)
    if in_bull_fvg:
        score += 2.0
        reasons.append("In Bullish FVG Zone (Buy Zone)")
    elif in_bear_fvg:
        score -= 2.0
        reasons.append("In Bearish FVG Zone (Sell Zone)")
        
    # 11. Rising Momentum & Volume (Dominant Factor)
    # "Mainly based on rising momentum and rising 12-hour volume"
    volume_rising = vol_12h > vol_12h_prev
    
    if volume_rising:
        # Bullish: Positive Momentum that is getting stronger
        if momentum > 0 and momentum > momentum_prev:
            score += 2.5
            reasons.append("Rising Mom & 12H Vol (Strong Buy)")
        # Bearish: Negative Momentum that is getting stronger (falling price speeding up)
        elif momentum < 0 and momentum < momentum_prev:
            score -= 2.5
            reasons.append("Rising Bearish Mom & 12H Vol (Strong Sell)")

    # --- Prediction Logic ---
    signal = "NEUTRAL"
    
    # Market Neutral Approach:
    # We only take positions if multiple factors align (High Confidence)
    if not is_active_time:
        signal = "NEUTRAL (Outside Hours)"
        reasons.insert(0, f"Inactive Time (UTC {current_hour}:00)")
    elif score >= 1.5: # Lowered slightly to show more activity
        signal = "BUY / LONG"
    elif score <= -1.5:
        signal = "SELL / SHORT"
    else:
        signal = "NEUTRAL / HEDGE"

    # Output formatting
    return {
        "symbol": symbol,
        "price": price,
        "score": score,
        "factors": reasons,
        "signal": signal,
        "date": current.name.date()
    }

def plot_backtest(df, trade_history, symbol):
    """
    Plots the price, indicators, and trades from the backtest.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Price and EMAs
    ax.plot(df.index, df['close'], label=f'{symbol} (Price)', color='black', alpha=0.5)
    ax.plot(df.index, df['ema_50'], label='Trend (50 EMA)', color='blue', alpha=0.7)
    ax.plot(df.index, df['bb_upper'], label='BB Upper', color='gray', linestyle='--', alpha=0.3)
    ax.plot(df.index, df['bb_lower'], label='BB Lower', color='gray', linestyle='--', alpha=0.3)
    ax.plot(df.index, df['ob_bull'], label='Bullish Order Block', color='green', linestyle=':', alpha=0.6)
    ax.plot(df.index, df['ob_bear'], label='Bearish Order Block', color='red', linestyle=':', alpha=0.6)
    
    # Plot FVG Zones (Fair Value Gaps)
    # Replace initialization values with NaN so they don't mess up the chart scale
    bull_top = df['last_bull_fvg_top'].replace(0, np.nan)
    bull_bot = df['last_bull_fvg_bottom'].replace(0, np.nan)
    bear_top = df['last_bear_fvg_top'].replace(10000000, np.nan)
    bear_bot = df['last_bear_fvg_bottom'].replace(10000000, np.nan)
    
    ax.fill_between(df.index, bull_top, bull_bot, color='green', alpha=0.15, label='Bullish FVG')
    ax.fill_between(df.index, bear_top, bear_bot, color='red', alpha=0.15, label='Bearish FVG')

    # Plot Trades
    for trade in trade_history:
        entry_date = trade['entry_date']
        exit_date = trade['exit_date']
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        
        color = 'green' if trade['type'] == 'LONG' else 'red'
        marker = '^' if trade['type'] == 'LONG' else 'v'
            
        # Mark Entry and Exit
        ax.scatter(entry_date, entry_price, color=color, marker=marker, s=100, zorder=5)
        ax.scatter(exit_date, exit_price, color='black', marker='x', s=50, zorder=5)
        ax.plot([entry_date, exit_date], [entry_price, exit_price], color=color, linestyle='--', alpha=0.5)

    ax.set_title(f'Backtest Analysis: {symbol}')
    ax.legend()
    plt.close(fig) # Prevent memory leaks
    return fig

def send_telegram_alert(token, chat_id, symbol, signal, price, score, reasons):
    """
    Sends a formatted trade alert to Telegram.
    """
    if not token or not chat_id:
        return

    emoji = "🟢" if "BUY" in signal else "🔴"
    msg = (
        f"{emoji} *TRADE ALERT: {symbol}*\n"
        f"*Signal:* {signal}\n"
        f"*Price:* ${price:,.2f}\n"
        f"*Score:* {score}\n\n"
        f"*Drivers:*\n• " + "\n• ".join(reasons)
    )
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"}, timeout=5)
    except Exception as e:
        print(f"Telegram Error: {e}")

def backtest_strategy(df, symbol, start_hour=0, end_hour=24, rsi_lower=30, rsi_upper=70):
    """
    Simple backtest to verify strategy performance on historical data.
    """
    if df is None or df.empty:
        return

    initial_balance = 10000
    balance = initial_balance
    position = None # None, 'LONG', 'SHORT'
    entry_price = 0
    entry_date = None
    trades = []
    trade_history = []
    peak_balance = initial_balance
    max_drawdown = 0
    
    status_text = f"Running Backtest for {symbol} ({len(df)} days)..."
    
    # Iterate through history (skip first 50 for EMA warmup)
    for i in range(50, len(df) - 1):
        current = df.iloc[i]
        # We execute trades at the OPEN of the next candle based on CURRENT signals
        next_open = df.iloc[i+1]['open']
        
        # Logic Variables
        price = current['close']
        ema_50 = current['ema_50']
        rsi = current['rsi']
        z_score = current['z_score']
        momentum = current['momentum']
        sentiment = current.get('sentiment', 50)
        cmf = current['cmf']
        ob_bull = current['ob_bull']
        ob_bear = current['ob_bear']
        volume = current['volume']
        vol_ma = current['vol_ma']
        adx = current['adx']
        vol_12h = current.get('vol_12h', 0)
        vol_12h_prev = current.get('vol_12h_prev', 0)
        momentum_prev = current.get('momentum_prev', 0)
        
        # FVG
        in_bull_fvg = (price <= current['last_bull_fvg_top']) and (price >= current['last_bull_fvg_bottom'])
        in_bear_fvg = (price >= current['last_bear_fvg_bottom']) and (price <= current['last_bear_fvg_top'])
        
        # Time Filter Logic
        current_hour = current.name.hour
        is_active_time = False
        if start_hour <= end_hour:
            is_active_time = (start_hour <= current_hour < end_hour)
        else:
            is_active_time = (current_hour >= start_hour or current_hour < end_hour)

        # Calculate Composite Score
        score = 0
        if price > ema_50: score += 1
        else: score -= 1
        if rsi < rsi_lower: score += 1
        elif rsi > rsi_upper: score -= 1
        if z_score < -2: score += 1
        elif z_score > 2: score -= 1
        if momentum > 0: score += 0.5
        else: score -= 0.5
        
        # Sentiment Logic
        if sentiment < 20: score += 1
        elif sentiment > 80: score -= 1
        
        # Order Flow
        if cmf > 0.1: score += 1
        elif cmf < -0.1: score -= 1
        
        # Order Blocks (Bounce trading)
        if 0 <= (price - ob_bull) / price <= 0.01: score += 1.5
        elif 0 <= (ob_bear - price) / price <= 0.01: score -= 1.5
        
        # Short Squeeze Detector
        if z_score > 3 and rsi > 75 and volume > (vol_ma * 3): score -= 2.0
        
        # ADX Filter
        if adx > 25: score *= 1.1
        elif adx < 20: score *= 0.5
        
        # FVG
        if in_bull_fvg: score += 2.0
        elif in_bear_fvg: score -= 2.0
        
        # 11. Rising Momentum & Volume (Dominant Factor)
        volume_rising = vol_12h > vol_12h_prev
        if volume_rising:
            if momentum > 0 and momentum > momentum_prev:
                score += 2.5
            elif momentum < 0 and momentum < momentum_prev:
                score -= 2.5
        
        # Exit Conditions
        if position == 'LONG' and score < 1:
            pnl = (next_open - entry_price) / entry_price
            balance *= (1 + pnl)
            trades.append(pnl)
            trade_history.append({
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': df.index[i+1],
                'exit_price': next_open,
                'type': 'LONG'
            })
            position = None
        elif position == 'SHORT' and score > -1:
            pnl = (entry_price - next_open) / entry_price
            balance *= (1 + pnl)
            trades.append(pnl)
            trade_history.append({
                'entry_date': entry_date,
                'entry_price': entry_price,
                'exit_date': df.index[i+1],
                'exit_price': next_open,
                'type': 'SHORT'
            })
            position = None
            
        # Track Drawdown
        if balance > peak_balance:
            peak_balance = balance
        drawdown = (peak_balance - balance) / peak_balance
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            
        # Entry Conditions
        if position is None and is_active_time:
            # Strong Buy Signal (Multiple factors align)
            if score >= 1.5:
                position = 'LONG'
                entry_price = next_open
                entry_date = df.index[i+1]
            # Strong Sell Signal (Trend Down + Overbought + High Vol)
            elif score <= -1.5:
                position = 'SHORT'
                entry_price = next_open
                entry_date = df.index[i+1]
                
    # Final PnL
    roi = ((balance - initial_balance) / initial_balance) * 100
    win_rate = (len([t for t in trades if t > 0]) / len(trades) * 100) if trades else 0
    
    gross_profit = sum([t for t in trades if t > 0])
    gross_loss = abs(sum([t for t in trades if t < 0]))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
    
    return {
        "final_balance": balance,
        "roi": roi,
        "total_trades": len(trades),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "history": trade_history
    }

def backtest_composite_derivative(symbol, timeframe='1h', flow_timeframe=None, start_date=None, end_date=None, lookback_days=30):
    """
    Backtests composite derivative entry/exit signals on historical OHLCV data.
    Uses all derivative strategies: MACD, ATR, MA20, money flow, volume ratio.
    Supports date range or lookback period.
    """
    if flow_timeframe is None:
        flow_timeframe = timeframe
    
    try:
        exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
        
        # Determine limit based on timeframe and lookback period
        timeframe_minutes = {'5m': 5, '15m': 15, '1h': 60, '4h': 240}
        minutes = timeframe_minutes.get(timeframe, 60)
        limit = max(300, int((lookback_days * 24 * 60) / minutes) + 50)
        
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=min(limit, 1000))
        
        if not ohlcv:
            return None
            
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Filter by date range if provided
        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]
        
        if len(df) < 60:
            st.error("Not enough data in selected date range.")
            return None
        
        # Calculate all indicators
        df['ma20'] = df['close'].rolling(window=20).mean()
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_hist'] = macd - signal
        
        prev_close = df['close'].shift(1)
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr14'] = tr.rolling(window=14).mean()
        
        # RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Money flow signal
        df['is_up'] = df['close'] >= df['open']
        df['money_flow_vol'] = df.loc[df['is_up'], 'volume'].rolling(window=14).sum()
        df['money_flow_vol'] = df['money_flow_vol'].fillna(0)
        df['counter_flow_vol'] = df.loc[~df['is_up'], 'volume'].rolling(window=14).sum()
        df['counter_flow_vol'] = df['counter_flow_vol'].fillna(0)
        df['net_flow'] = df['money_flow_vol'] - df['counter_flow_vol']
        df['total_flow'] = df['money_flow_vol'] + df['counter_flow_vol']
        df['money_flow_signal'] = df['net_flow'] / (df['total_flow'] + 1e-9)
        
        # Volume ratio
        df['vol_ma20'] = df['volume'].rolling(window=20).mean()
        df['vol_ratio'] = df['volume'] / (df['vol_ma20'] + 1e-9)
        
        # Generate signals
        initial_balance = 10000
        balance = initial_balance
        position = None
        entry_price = 0
        entry_date = None
        trades = []
        trade_history = []
        peak_balance = initial_balance
        max_drawdown = 0
        
        for i in range(50, len(df) - 1):
            current = df.iloc[i]
            next_row = df.iloc[i + 1]
            
            # Entry conditions
            price = float(current['close'])
            ma20 = float(current['ma20']) if not pd.isna(current['ma20']) else price
            macd_hist = float(current['macd_hist']) if not pd.isna(current['macd_hist']) else 0
            atr = float(current['atr14']) if not pd.isna(current['atr14']) and float(current['atr14']) > 0 else 1.0
            rsi = float(current['rsi']) if not pd.isna(current['rsi']) else 50.0
            money_flow_sig = float(current['money_flow_signal']) if not pd.isna(current['money_flow_signal']) else 0
            vol_ratio = float(current['vol_ratio']) if not pd.isna(current['vol_ratio']) else 1.0
            
            # Normalized composite entry logic
            normalized_macd = np.tanh(macd_hist / (atr + 1e-9))
            normalized_rsi = (rsi - 50.0) / 50.0
            flow_score = money_flow_sig
            vol_strength = min(vol_ratio, 2.0) / 2.0
            
            entry_score = (0.4 * normalized_macd) + (0.3 * normalized_rsi) + (0.2 * flow_score) + (0.1 * vol_strength)
            trend_ok = price > ma20
            
            buy_signal = entry_score > 0.25 and trend_ok
            sell_signal = entry_score < 0.1 or not trend_ok
            
            # Position management
            if position is None and buy_signal:
                position = 'LONG'
                entry_price = float(next_row['open'])
                entry_date = i + 1
            elif position == 'LONG' and sell_signal:
                pnl = (float(next_row['open']) - entry_price) / entry_price
                balance *= (1 + pnl)
                trades.append(pnl)
                trade_history.append({
                    'entry': entry_date,
                    'exit': i + 1,
                    'price_entry': entry_price,
                    'price_exit': float(next_row['open']),
                    'pnl': pnl
                })
                position = None
            
            # Track drawdown
            if balance > peak_balance:
                peak_balance = balance
            drawdown = ((peak_balance - balance) / peak_balance) * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Final stats
        roi = ((balance - initial_balance) / initial_balance) * 100
        win_rate = (len([t for t in trades if t > 0]) / len(trades) * 100) if trades else 0
        gross_profit = sum([t for t in trades if t > 0])
        gross_loss = abs(sum([t for t in trades if t < 0]))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        return {
            "final_balance": balance,
            "roi": roi,
            "total_trades": len(trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "trade_history": trade_history
        }
    except Exception as e:
        st.error(f"Backtest error: {e}")
        return None

def scan_and_rank_crypto():
    """
    Dynamically fetches the Top 50 Crypto assets (by 24h volume) and ranks them by Momentum.
    """
    exchange = ccxt.binance({'enableRateLimit': True})
    
    try:
        # Fetch all live tickers from Binance
        all_tickers = exchange.fetch_tickers()
        # Filter for active USDT pairs and extract quote volume
        usdt_pairs = [
            data for symbol, data in all_tickers.items() 
            if symbol.endswith('/USDT') and data.get('quoteVolume') is not None and data.get('active', True)
        ]
        # Sort by 24h quote volume to get the most liquid/top assets
        usdt_pairs.sort(key=lambda x: x['quoteVolume'], reverse=True)
        # Grab the top 50 symbols, plus our standard stocks
        tickers = [x['symbol'] for x in usdt_pairs[:50]] + ['SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM']
    except Exception as e:
        st.error(f"Could not fetch dynamic tickers: {e}")
        tickers = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM'] # Fallback
        
    # --- Fetch Futures Metrics (Funding, Volume, OI) ---
    funding_rates = {}
    futures_24h_vol = {}
    futures_oi = {}
    swap_sym_map = {}
    exchange_swap = ccxt.binance({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    try:
        funding_rates_data = exchange_swap.fetch_funding_rates()
        swap_tickers_data = exchange_swap.fetch_tickers()
        
        try:
            oi_data = exchange_swap.fetch_open_interests()
        except Exception:
            oi_data = {}
            
        for swap_sym, data in funding_rates_data.items():
            # Convert 'BTC/USDT:USDT' to 'BTC/USDT' to match spot tickers
            spot_sym = swap_sym.split(':')[0]
            fr = data.get('fundingRate')
            funding_rates[spot_sym] = fr if fr is not None else 0.0
            swap_sym_map[spot_sym] = swap_sym
            
        for swap_sym, data in swap_tickers_data.items():
            spot_sym = swap_sym.split(':')[0]
            vol = data.get('quoteVolume')
            futures_24h_vol[spot_sym] = vol if vol is not None else 0.0
            
        for swap_sym, data in oi_data.items():
            spot_sym = swap_sym.split(':')[0]
            futures_oi[spot_sym] = data
    except Exception as e:
        st.warning(f"Could not fetch futures metrics: {e}")
        
    stats = []
    progress_bar = st.progress(0)
    total_tickers = len(tickers)
    
    def process_ticker(sym):
        df = fetch_and_analyze(sym, timeframe='1h', silent=True)
        if df is not None and not df.empty:
            mom = df['momentum'].iloc[-1]
            price = df['close'].iloc[-1]
            fr = funding_rates.get(sym, 0.0)
            vol_24h = futures_24h_vol.get(sym, 0.0)
            
            vol_12h = 0.0
            if sym in swap_sym_map:
                try:
                    ohlcv_1h = exchange_swap.fetch_ohlcv(swap_sym_map[sym], timeframe='1h', limit=12)
                    if ohlcv_1h:
                        vol_12h = sum(c[5] * c[4] for c in ohlcv_1h)
                except Exception:
                    pass
                    
            oi_info = futures_oi.get(sym, {})
            notional_oi = oi_info.get('openInterestValue')
            if notional_oi is None:
                base_oi = oi_info.get('openInterest', 0.0)
                notional_oi = base_oi * price
                
            trend = df['close'].tail(12).tolist()
            vol_profile = df['volume'].tail(12).tolist()
            return {
                'symbol': sym, 
                'momentum': mom, 
                'price': price, 
                'funding_rate': fr, 
                'notional_oi': notional_oi,
                'futures_12h_vol': vol_12h,
                'futures_24h_vol': vol_24h,
                '12h_trend': trend, 
                '1h_volume': vol_profile
            }
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(process_ticker, sym) for sym in tickers]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result:
                stats.append(result)
            # Update progress bar
            progress_bar.progress((i + 1) / total_tickers)
            
    # Sort descending
    stats.sort(key=lambda x: x['momentum'], reverse=True)
    
    return pd.DataFrame(stats)

@st.cache_data(ttl=300)
def scan_top_derivative_assets(timeframe='1h', flow_timeframe=None, volume_timeframe='1h', top_n=30):
    """
    Scan top derivative (swap) pairs and compute momentum, z-score, inflow/outflow, and liquidity.
    `timeframe` is used for momentum/z-score; `flow_timeframe` is used to compute inflow/outflow.
    `volume_timeframe` is used for the short-term volume component in liquidity ratio.
    """
    if flow_timeframe is None:
        flow_timeframe = timeframe

    def fetch_market_caps(symbols):
        market_caps = {}
        try:
            cg_list = requests.get("https://api.coingecko.com/api/v3/coins/list", timeout=10).json()
            symbol_to_id = {}
            for coin in cg_list:
                symbol = coin.get('symbol', '').upper()
                if symbol and symbol not in symbol_to_id:
                    symbol_to_id[symbol] = coin.get('id')

            ids = [symbol_to_id[symbol] for symbol in symbols if symbol in symbol_to_id]
            if ids:
                chunk_size = 100
                for i in range(0, len(ids), chunk_size):
                    batch = ids[i:i + chunk_size]
                    params = {
                        'vs_currency': 'usd',
                        'ids': ','.join(batch),
                        'order': 'market_cap_desc',
                        'per_page': len(batch),
                        'page': 1,
                        'sparkline': 'false'
                    }
                    response = requests.get("https://api.coingecko.com/api/v3/coins/markets", params=params, timeout=10)
                    for item in response.json():
                        symbol = item.get('symbol', '').upper()
                        market_caps[symbol] = item.get('market_cap', 0.0) or 0.0
        except Exception:
            pass
        return market_caps

    exchange = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    try:
        tickers = exchange.fetch_tickers()
    except Exception as e:
        st.error(f"Could not fetch derivative tickers: {e}")
        return pd.DataFrame()

    swap_pairs = []
    for pair, data in tickers.items():
        if not data.get('active', True):
            continue
        if pair.endswith(':USDT') or pair.endswith('/USDT'):
            if data.get('quoteVolume') is not None:
                swap_pairs.append((pair, data))

    if not swap_pairs:
        return pd.DataFrame()

    swap_pairs.sort(key=lambda x: x[1].get('quoteVolume', 0), reverse=True)
    swap_pairs = swap_pairs[:top_n]

    funding_rates = {}
    try:
        fr_data = exchange.fetch_funding_rates()
        for sym, d in fr_data.items():
            base_sym = sym.split(':')[0]
            funding_rates[base_sym] = d.get('fundingRate', 0.0)
    except Exception:
        pass

    # Fetch open interest data for swap pairs (map base symbol -> oi info)
    oi_map = {}
    try:
        oi_data = exchange.fetch_open_interests()
        for k, v in oi_data.items():
            base = k.split(':')[0]
            oi_map[base] = v
    except Exception:
        oi_map = {}

    stats = []
    progress_bar = st.progress(0)

    def process_pair(pair_data):
        full_symbol, ticker = pair_data
        base_symbol = full_symbol.split(':')[0]
        try:
            ohlcv = exchange.fetch_ohlcv(full_symbol, timeframe=timeframe, limit=100)
            if not ohlcv:
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['open'] = pd.to_numeric(df['open'], errors='coerce')
            df['volume'] = pd.to_numeric(df['volume'], errors='coerce')

            df['momentum'] = df['close'].pct_change(periods=1)
            df['ma20'] = df['close'].rolling(window=20).mean()
            df['std20'] = df['close'].rolling(window=20).std()
            df['z_score'] = (df['close'] - df['ma20']) / df['std20']

            # Use a separate timeframe for inflow/outflow if requested
            inflow = outflow = net_flow = 0.0
            money_flow_signal = 0.0
            vol_ratio = 1.0

            # --- Entry Analysis Components ---
            # MACD histogram and ATR
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            df['macd_hist'] = macd - signal

            prev_close = df['close'].shift(1)
            tr1 = df['high'] - df['low']
            tr2 = (df['high'] - prev_close).abs()
            tr3 = (df['low'] - prev_close).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            df['atr14'] = tr.rolling(window=14).mean()
            
            # RSI (14)
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            df['ma20'] = df['close'].rolling(window=20).mean()

            try:
                ohlcv_flow = exchange.fetch_ohlcv(full_symbol, timeframe=flow_timeframe, limit=100)
                if ohlcv_flow:
                    df_flow = pd.DataFrame(ohlcv_flow, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_flow['open'] = pd.to_numeric(df_flow['open'], errors='coerce')
                    df_flow['close'] = pd.to_numeric(df_flow['close'], errors='coerce')
                    df_flow['volume'] = pd.to_numeric(df_flow['volume'], errors='coerce')
                    df_flow['is_up'] = df_flow['close'] >= df_flow['open']
                    inflow = float(df_flow.loc[df_flow['is_up'], 'volume'].sum())
                    outflow = float(df_flow.loc[~df_flow['is_up'], 'volume'].sum())
                    net_flow = inflow - outflow
                    total_flow = inflow + outflow
                    if total_flow > 0:
                        money_flow_signal = net_flow / total_flow

                    avg_flow_vol = float(df_flow['volume'].rolling(window=20).mean().iloc[-1]) if len(df_flow) >= 20 else float(df_flow['volume'].mean())
                    last_flow_vol = float(df_flow['volume'].iloc[-1])
                    if avg_flow_vol > 0:
                        vol_ratio = last_flow_vol / avg_flow_vol
            except Exception:
                inflow = outflow = net_flow = 0.0
                money_flow_signal = 0.0
                vol_ratio = 1.0

            current = df.iloc[-1]

            oi_info = oi_map.get(base_symbol, {})
            open_interest = oi_info.get('openInterestValue') or oi_info.get('openInterest') or ticker.get('openInterest', 0.0)

            # Funding signal: normalize funding rate into a score -1..+1 range
            funding_rate = funding_rates.get(base_symbol, ticker.get('fundingRate', 0.0))
            funding_signal = np.tanh(funding_rate * 100)

            # Z-score signal: positive bullish momentum, negative bearish
            z_signal = float(current['z_score']) if np.isfinite(current['z_score']) else 0.0
            z_score_signal = np.tanh(z_signal / 3)

            # Trend Probability Score
            tps = (0.4 * z_score_signal) + (0.3 * money_flow_signal) + (0.3 * funding_signal)

            # Normalized composite entry rule
            try:
                macd_hist_val = float(df['macd_hist'].iloc[-1])
                atr_val = float(df['atr14'].iloc[-1]) if not pd.isna(df['atr14'].iloc[-1]) and float(df['atr14'].iloc[-1]) > 0 else 1.0
                rsi_val = float(df['rsi'].iloc[-1]) if not pd.isna(df['rsi'].iloc[-1]) else 50.0
            except Exception:
                macd_hist_val = 0.0
                atr_val = 1.0
                rsi_val = 50.0
            
            # Normalized signals: all scaled to -1..+1 or 0..1
            normalized_macd = np.tanh(macd_hist_val / (atr_val + 1e-9))
            normalized_rsi = (rsi_val - 50.0) / 50.0
            flow_score = money_flow_signal  # already -1..+1
            vol_strength = min(vol_ratio, 2.0) / 2.0  # cap at 2.0, scale to 0-1
            
            # Weighted composite score
            entry_score = (0.4 * normalized_macd) + (0.3 * normalized_rsi) + (0.2 * flow_score) + (0.1 * vol_strength)
            
            # Trend filter
            price = float(current['close'])
            ma20 = float(df['ma20'].iloc[-1]) if not pd.isna(df['ma20'].iloc[-1]) else price
            trend_ok = price > ma20
            
            entry_signal = 'BUY' if entry_score > 0.25 and trend_ok else ''

            # Fetch volumes at different timeframes
            vol_5m = vol_15m = vol_1h = vol_4h = 0.0
            try:
                for tf, vol_var in [('5m', 'vol_5m'), ('15m', 'vol_15m'), ('1h', 'vol_1h'), ('4h', 'vol_4h')]:
                    try:
                        vol_ohlcv = exchange.fetch_ohlcv(full_symbol, timeframe=tf, limit=1)
                        if vol_ohlcv:
                            vol_val = float(vol_ohlcv[-1][5])  # volume is index 5
                            if vol_var == 'vol_5m':
                                vol_5m = vol_val
                            elif vol_var == 'vol_15m':
                                vol_15m = vol_val
                            elif vol_var == 'vol_1h':
                                vol_1h = vol_val
                            elif vol_var == 'vol_4h':
                                vol_4h = vol_val
                    except Exception:
                        pass
            except Exception:
                pass

            # Dedicated 15-minute RSI for quick short-term derivative momentum checks
            rsi_15m = 50.0
            try:
                ohlcv_15m = exchange.fetch_ohlcv(full_symbol, timeframe='15m', limit=100)
                if ohlcv_15m:
                    df_15m = pd.DataFrame(ohlcv_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_15m['close'] = pd.to_numeric(df_15m['close'], errors='coerce')
                    delta_15m = df_15m['close'].diff()
                    gain_15m = delta_15m.where(delta_15m > 0, 0).rolling(window=14).mean()
                    loss_15m = (-delta_15m.where(delta_15m < 0, 0)).rolling(window=14).mean()
                    rs_15m = gain_15m / (loss_15m + 1e-9)
                    rsi_series_15m = 100 - (100 / (1 + rs_15m))
                    latest_rsi_15m = rsi_series_15m.iloc[-1]
                    if not pd.isna(latest_rsi_15m):
                        rsi_15m = float(latest_rsi_15m)
            except Exception:
                rsi_15m = 50.0

            # Liquidity ratio uses short-term volume over market cap, times 24h volume
            liquidity_ratio = 0.0
            try:
                base_sym = base_symbol.upper().replace('USDT', '')
                market_caps = fetch_market_caps([base_sym])
                market_cap = market_caps.get(base_sym, 0.0)
                selected_vol = {
                    '5m': vol_5m,
                    '15m': vol_15m,
                    '1h': vol_1h,
                    '4h': vol_4h
                }.get(volume_timeframe, vol_1h)
                if market_cap > 0:
                    liquidity_ratio = (selected_vol / market_cap) * float(ticker.get('quoteVolume', 0.0))
            except Exception:
                liquidity_ratio = 0.0

            return {
                'symbol': base_symbol,
                'price': current['close'],
                'momentum': current['momentum'],
                'z_score': current['z_score'],
                'funding_rate': funding_rate,
                'funding_signal': funding_signal,
                'money_flow_signal': money_flow_signal,
                'tps': tps,
                'open_interest': open_interest,
                '24h_volume': ticker.get('quoteVolume', 0.0),
                'vol_5m': vol_5m,
                'vol_15m': vol_15m,
                'vol_1h': vol_1h,
                'vol_4h': vol_4h,
                'rsi_15m': rsi_15m,
                'liquidity_ratio': liquidity_ratio,
                'inflow': inflow,
                'outflow': outflow,
                'net_flow': net_flow,
                'entry_score': entry_score,
                'entry_signal': entry_signal
            }
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_pair, pair_data) for pair_data in swap_pairs]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            result = future.result()
            if result:
                stats.append(result)
            progress_bar.progress((i + 1) / len(swap_pairs))

    stats.sort(key=lambda x: x['momentum'] if x['momentum'] is not None else -999, reverse=True)
    return pd.DataFrame(stats)


def optimize_parameters(df, symbol, active_hours):
    """
    Runs a Grid Search to find the best RSI parameters.
    """
    rsi_lowers = [20, 25, 30, 35]
    rsi_uppers = [65, 70, 75, 80]
    results = []
    
    total_iterations = len(rsi_lowers) * len(rsi_uppers)
    progress_bar = st.progress(0)
    iteration = 0
    
    for lower in rsi_lowers:
        for upper in rsi_uppers:
            # Run backtest with specific params
            stats = backtest_strategy(df, symbol, start_hour=active_hours[0], end_hour=active_hours[1], rsi_lower=lower, rsi_upper=upper)
            stats['rsi_lower'] = lower
            stats['rsi_upper'] = upper
            results.append(stats)
            iteration += 1
            progress_bar.progress(iteration / total_iterations)
            
    return pd.DataFrame(results)

def color_metrics(val):
    if isinstance(val, (int, float)):
        color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
        return f'color: {color}'
    return ''

def format_large_number(x):
    try:
        if pd.isna(x):
            return "$0"
        x = float(x)
        is_negative = x < 0
        x = abs(x)
        
        if x >= 1e9:
            val = f"${x/1e9:.2f}B"
        elif x >= 1e6:
            val = f"${x/1e6:.2f}M"
        elif x >= 1e3:
            val = f"${x/1e3:.2f}K"
        else:
            val = f"${x:.2f}"
        return f"-{val}" if is_negative else val
    except (ValueError, TypeError):
        return "$0"

def plot_volatility_surface(df, symbol):
    """
    Computes and plots the "quantum" volatility surface and classical distribution.
    Includes a marker for the most recent data point.
    """
    if df is None or 'close' not in df.columns or len(df) < 50:
        st.warning("Not enough historical data to generate volatility surface.")
        return None

    # 1. Calculate volatility (e.g., rolling 20-period stdev of log returns)
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['log_ret'].rolling(window=20).std() * np.sqrt(252) # Annualized
    df.dropna(inplace=True)

    if df['volatility'].empty:
        st.warning("Could not compute volatility.")
        return None

    # 2. Define a "wave function" psi from the volatility series
    vol_series = df['volatility'].values
    vol_change = df['volatility'].diff().fillna(0).values
    psi = vol_series + 1j * vol_change

    # 3. Get probability |psi|^2 and phase Arg(psi)
    prob_density = np.abs(psi)**2
    phase = np.angle(psi)

    # 4. Create the 2D grid for the surface plot
    x = vol_series
    y = vol_change
    z = prob_density

    # 5. Create the plots using Matplotlib
    fig = plt.figure(figsize=(15, 7))
    
    # 3D Quantum Probability Surface
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_trisurf(x, y, z, cmap='viridis', antialiased=True, alpha=0.8)
    surf.set_array(phase)
    surf.set_clim(-np.pi, np.pi)
    ax1.set_title(f'Quantum Volatility Surface for {symbol}\n(Color = Phase)', fontsize=12)
    ax1.set_xlabel('Volatility', fontsize=10)
    ax1.set_ylabel('Volatility Change (Momentum)', fontsize=10)
    ax1.set_zlabel('Quantum Probability |ψ|²', fontsize=10)
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5, label='Phase Angle (Arg(ψ))')

    # Add a "You Are Here" marker for the latest point
    ax1.scatter(x[-1], y[-1], z[-1], color='red', s=100, edgecolor='black', depthshade=True, label='Current State', zorder=10)
    ax1.legend()

    price_prediction_data = {}
    # --- Draw Predicted Path (Probability Current) ---
    try:
        # 1. Create a KD-Tree for efficient nearest neighbor search in the (vol, vol_change) plane
        all_points = np.vstack((x, y)).T
        tree = cKDTree(all_points)

        # 2. Find the 10 nearest neighbors to the current point
        current_pos = np.array([x[-1], y[-1]])
        distances, indices = tree.query(current_pos, k=10)

        # 3. Calculate the gradient of the phase in this local neighborhood
        # We use linear regression (least squares) to fit a plane to the phase data of the neighbors
        neighbor_points = all_points[indices]
        neighbor_phases = phase[indices]
        
        A = np.c_[neighbor_points, np.ones(len(indices))]
        # Fit a plane: z = a*x + b*y + c. The gradient is (a, b)
        gradient, _, _, _ = np.linalg.lstsq(A, neighbor_phases, rcond=None)
        grad_x, grad_y = gradient[0], gradient[1]

        # 4. Draw the gradient vector as an arrow on the plot
        arrow_length_factor = 0.1 # Adjust to make arrow longer/shorter
        ax1.quiver(x[-1], y[-1], z[-1], grad_x, grad_y, 0, length=arrow_length_factor, normalize=True, color='magenta', linewidth=3, label='Predicted Path')
        
        # Store data for price prediction
        price_prediction_data['vol_grad_x'] = grad_x

    except Exception as e:
        price_prediction_data['vol_grad_x'] = 0

    # 2D Classical Histogram
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.hist(df['volatility'], bins=50, orientation='horizontal', density=True, color='skyblue', edgecolor='black')
    ax2.set_title(f'Classical Volatility Distribution', fontsize=12)
    ax2.set_xlabel('Empirical Density', fontsize=10)
    ax2.set_ylabel('Volatility', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    return fig, price_prediction_data

def main():
    st.title("Quantitative Scalping Dashboard (1h) 📈")
    
    # Sidebar
    if st.sidebar.button("🔄 Force Data Refresh"):
        st.cache_data.clear()
        st.sidebar.success("Cache cleared! Data will be fetched fresh on next action.")
        
    st.sidebar.header("Configuration")
    # Dropdown list for asset selection
    asset_options = [
        'BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT', 'DOGE/USDT', 'ADA/USDT', 
        'TRX/USDT', 'AVAX/USDT', 'SHIB/USDT', 'DOT/USDT', 'LINK/USDT', 'BCH/USDT', 'NEAR/USDT', 
        'LTC/USDT', 'MATIC/USDT', 'UNI/USDT', 'APT/USDT', 'ICP/USDT', 'FIL/USDT',
        'TON/USDT', 'XLM/USDT', 'ETC/USDT', 'XMR/USDT', 'OKB/USDT', 'ATOM/USDT', 'HBAR/USDT', 
        'VET/USDT', 'CRO/USDT', 'AR/USDT', 'MNT/USDT', 'OP/USDT', 'INJ/USDT', 'RNDR/USDT', 
        'GRT/USDT', 'IMX/USDT', 'STX/USDT', 'THETA/USDT', 'EGLD/USDT', 'FTM/USDT', 'ALGO/USDT', 
        'TIA/USDT', 'AAVE/USDT', 'FLOW/USDT', 'QNT/USDT', 'SNX/USDT',
        'SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM'
    ]
    symbol = st.sidebar.selectbox("Ticker Symbol", options=asset_options)
    # Default backtest to 30 days for 1h timeframe to avoid huge data loads
    backtest_start = st.sidebar.date_input("Backtest Start", value=datetime.now() - timedelta(days=30))
    
    st.sidebar.header("Strategy Settings")
    
    # Dynamically adjust hours based on asset type
    us_stocks_indices = ['SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM']
    is_stock = symbol in us_stocks_indices
    
    default_hours = (8, 21) if is_stock else (0, 23)
    label = "Active Hours (UTC) - Stocks" if is_stock else "Active Hours (UTC) - Crypto (24/7)"
    active_hours = st.sidebar.slider(label, 0, 23, default_hours)
    
    st.sidebar.header("Notifications")
    tg_token = st.sidebar.text_input("Telegram Bot Token", type="password")
    tg_chat_id = st.sidebar.text_input("Telegram Chat ID")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["📊 Market Overview", "⚡ Top Crypto Ranking", "🔥 Derivatives Trend Scan", "🛠️ Backtest Engine", "🏛️ US Indices", "🎯 Composite Derivative Backtest", "🌌 Volatility Quantum Analysis"])

    with tab1:
        st.subheader(f"Live Analysis: {symbol}")
        if st.button("Analyze Current Market"):
            df = fetch_and_analyze(symbol)
            if df is not None:
                sig = generate_signal(df, symbol, start_hour=active_hours[0], end_hour=active_hours[1])
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${sig['price']:.2f}")
                col2.metric("Composite Score", f"{sig['score']}")
                col3.metric("Signal", sig['signal'])
                st.write(f"**Factors Driving Signal:** {', '.join(sig['factors'])}")
                
                # Send Alert if Strong Signal detected during analysis
                if ("BUY" in sig['signal'] or "SELL" in sig['signal']) and tg_token and tg_chat_id:
                    send_telegram_alert(tg_token, tg_chat_id, symbol, sig['signal'], sig['price'], sig['score'], sig['factors'])
                    st.success(f"Telegram Alert sent to {tg_chat_id}!")
            else:
                st.error(f"Could not load data for {symbol}. Please check the ticker.")

    with tab2:
        st.subheader("Top Crypto Momentum Ranking")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            scan_btn = st.button("Scan Top Crypto")
        with col2:
            auto_scan = st.checkbox("🔄 Auto-Scan (Refresh every 5 mins)")
            
        if scan_btn or auto_scan:
            with st.spinner("Scanning crypto markets..."):
                df_rank = scan_and_rank_crypto()
                
                styler = df_rank.style.format({
                    "momentum": "{:.2%}", 
                    "price": "${:.2f}", 
                    "funding_rate": "{:.4%}",
                    "notional_oi": format_large_number,
                    "futures_12h_vol": format_large_number,
                    "futures_24h_vol": format_large_number
                })
                
                if hasattr(styler, 'map'):
                    styler = styler.map(color_metrics, subset=['momentum', 'funding_rate'])
                else:
                    styler = styler.applymap(color_metrics, subset=['momentum', 'funding_rate'])
                    
                styler = styler.background_gradient(subset=['futures_12h_vol', 'futures_24h_vol'], cmap='Blues')

                st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.dataframe(
                    styler,
                    column_config={
                        "notional_oi": st.column_config.Column(
                            "Open Interest", 
                            help="Notional Open Interest (USDT)"
                        ),
                        "futures_12h_vol": st.column_config.Column(
                            "12H Futures Vol", 
                            help="Rolling 12-Hour Futures Trading Volume (USDT)"
                        ),
                        "futures_24h_vol": st.column_config.Column(
                            "24H Futures Vol", 
                            help="24-Hour Futures Trading Volume (USDT)"
                        ),
                        "12h_trend": st.column_config.LineChartColumn(
                            "12H Trend", help="Price movement over the last 12 hours"
                        ),
                        "1h_volume": st.column_config.BarChartColumn(
                            "1h Volume Profile", help="1-hour volume bars over the last 12 hours"
                        )
                    }
                )
                
            if auto_scan:
                time.sleep(301) # Wait slightly over 5m to clear the 300s cache TTL
                if hasattr(st, 'rerun'):
                    st.rerun()
                else:
                    st.experimental_rerun()

        # --- 12-HOUR MICRO-MOMENTUM BREAKDOWN ---
        st.divider()
        st.subheader("🔍 12-Hour Micro-Momentum Breakdown")
        st.write("Breaks down the past 12 hours into 1-hour returns to spot building bullish or bearish pressure.")
        
        micro_options = st.session_state.get('scanned_tickers', asset_options)
        micro_idx = micro_options.index(symbol) if symbol in micro_options else 0
        micro_sym = st.selectbox("Select Asset to Analyze", options=micro_options, index=micro_idx, key='micro_sym')
        
        micro_df = fetch_and_analyze(micro_sym, timeframe='1h', silent=True)
        if micro_df is not None and len(micro_df) >= 13:
            # Calculate 1h returns before slicing so the first candle has a valid return
            micro_df['1h_return'] = micro_df['close'].pct_change() * 100
            last_12h = micro_df.tail(12).copy()
            
            # Format index for better chart display (e.g., just the time '14:35')
            last_12h.index = last_12h.index.strftime('%H:%M')
            
            # Plotting bar chart of 1h returns
            st.bar_chart(last_12h['1h_return'])
            
            # Summary metrics
            bullish_candles = (last_12h['1h_return'] > 0).sum()
            bearish_candles = (last_12h['1h_return'] < 0).sum()
            net_12h_return = ((micro_df['close'].iloc[-1] - micro_df['close'].iloc[-13]) / micro_df['close'].iloc[-13]) * 100
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Net 12H Price Change", f"{net_12h_return:.2f}%")
            c2.metric("Bullish 1h Candles", int(bullish_candles))
            c3.metric("Bearish 1h Candles", int(bearish_candles))
            
            # Momentum Verdict
            if net_12h_return > 0 and bullish_candles > bearish_candles:
                st.success(f"**Verdict:** Momentum is steadily building **BULLISH** 📈")
            elif net_12h_return < 0 and bearish_candles > bullish_candles:
                st.error(f"**Verdict:** Momentum is steadily building **BEARISH** 📉")
            else:
                st.warning(f"**Verdict:** Momentum is currently **MIXED / CONSOLIDATING** ⚖️")

    with tab3:
        st.subheader("Top Derivatives Trend Scan")
        st.write("Scan the top Binance USDT perpetual contract derivatives and compare momentum with Z-score.")
        timeframe_deriv = st.selectbox("Select timeframe", ["5m", "15m", "1h", "4h"], index=2)
        flow_timeframe = st.selectbox("Inflow/Outflow timeframe", ["5m", "15m", "1h", "4h"], index=2)
        volume_timeframe = st.selectbox("Short-term volume timeframe", ["5m", "15m", "1h", "4h"], index=2)

        if st.button("Scan Top Derivatives"):
            with st.spinner("Scanning top derivative assets..."):
                df_deriv = scan_top_derivative_assets(timeframe=timeframe_deriv, flow_timeframe=flow_timeframe, volume_timeframe=volume_timeframe, top_n=100)
                if df_deriv is not None and not df_deriv.empty:
                    df_deriv['momentum'] = df_deriv['momentum'].fillna(0)
                    df_deriv['z_score'] = df_deriv['z_score'].fillna(0)
                    df_deriv['funding_rate'] = df_deriv['funding_rate'].fillna(0)
                    df_deriv['funding_signal'] = df_deriv['funding_signal'].fillna(0)
                    df_deriv['money_flow_signal'] = df_deriv['money_flow_signal'].fillna(0)
                    df_deriv['tps'] = df_deriv['tps'].fillna(0)
                    df_deriv['inflow'] = df_deriv['inflow'].fillna(0)
                    df_deriv['outflow'] = df_deriv['outflow'].fillna(0)
                    df_deriv['net_flow'] = df_deriv['net_flow'].fillna(0)
                    if 'rsi_15m' not in df_deriv.columns:
                        df_deriv['rsi_15m'] = 50.0
                    df_deriv['rsi_15m'] = df_deriv['rsi_15m'].fillna(50)
                    df_deriv['entry_score'] = df_deriv.get('entry_score', 0).fillna(0)
                    df_deriv['entry_signal'] = df_deriv.get('entry_signal', '').fillna('')
                    st.caption("TPS = (0.4 × Z-score Signal) + (0.3 × Money Flow Signal) + (0.3 × Funding Signal)")
                    styler = df_deriv.style.format({
                        "price": "${:.2f}",
                        "momentum": "{:.2%}",
                        "z_score": "{:.2f}",
                        "funding_rate": "{:.4%}",
                        "funding_signal": "{:.2f}",
                        "money_flow_signal": "{:.2f}",
                        "tps": "{:.2f}",
                        "liquidity_ratio": "{:.6f}",
                        "rsi_15m": "{:.2f}",
                        "entry_score": "{:.3f}",
                        "entry_signal": "{}",
                        "open_interest": format_large_number,
                        "24h_volume": format_large_number,
                        "inflow": format_large_number,
                        "outflow": format_large_number,
                        "net_flow": format_large_number
                    })
                    if hasattr(styler, 'map'):
                        styler = styler.map(color_metrics, subset=['momentum', 'z_score', 'money_flow_signal', 'funding_signal', 'tps', 'liquidity_ratio', 'net_flow', 'rsi_15m', 'entry_score'])
                    else:
                        styler = styler.applymap(color_metrics, subset=['momentum', 'z_score', 'money_flow_signal', 'funding_signal', 'tps', 'liquidity_ratio', 'net_flow', 'rsi_15m', 'entry_score'])
                    st.dataframe(styler, width='stretch')
                    # Top-10 upside candidates by TPS with Z-score
                    try:
                        top10 = df_deriv.sort_values(by='tps', ascending=False).head(10).reset_index(drop=True)
                        if not top10.empty:
                            st.subheader("Top 10 Upside Candidates (by TPS)")
                            st.write("Bars = TPS (higher = more probable upside). Line = Z-score.")
                            top10_plot = top10.set_index('symbol')
                            st.bar_chart(top10_plot['tps'])
                            st.line_chart(top10_plot['z_score'])
                    except Exception as e:
                        st.warning(f"Could not render top-10 chart: {e}")
                else:
                    st.warning("No derivative asset data returned. Try again in a moment.")

    with tab4:
        st.subheader(f"Strategy Backtest: {symbol}")
        if st.button("Run Backtest"):
            df = fetch_and_analyze(symbol, start_date=backtest_start.strftime('%Y-%m-%d'))
            if df is not None:
                with st.spinner("Simulating strategy..."):
                    stats = backtest_strategy(df, symbol, start_hour=active_hours[0], end_hour=active_hours[1])
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Final Balance", f"${stats['final_balance']:.2f}")
                    m2.metric("ROI", f"{stats['roi']:.2f}%")
                    m3.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
                    m4.metric("Win Rate", f"{stats['win_rate']:.2f}%")
                    
                    fig = plot_backtest(df, stats['history'], symbol)
                    st.pyplot(fig)
            else:
                st.error(f"Could not load backtest data for {symbol}.")

        st.divider()
        st.subheader("🤖 AI Parameter Optimizer")
        st.write("Automatically find the best RSI thresholds for this asset and timeframe.")
        
        if st.button("Run Optimization Loop"):
            df = fetch_and_analyze(symbol, start_date=backtest_start.strftime('%Y-%m-%d'))
            if df is not None:
                with st.spinner("Testing 16 parameter combinations..."):
                    results_df = optimize_parameters(df, symbol, active_hours)
                    
                    # Find best result by ROI
                    best = results_df.sort_values(by='roi', ascending=False).iloc[0]
                    
                    st.success(f"💎 Best Parameters Found: RSI < {best['rsi_lower']} and RSI > {best['rsi_upper']}")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Optimized ROI", f"{best['roi']:.2f}%")
                    col2.metric("Win Rate", f"{best['win_rate']:.2f}%")
                    col3.metric("Total Trades", best['total_trades'])
                    
                    top_results = results_df[['rsi_lower', 'rsi_upper', 'roi', 'win_rate', 'profit_factor']].sort_values(by='roi', ascending=False).head(5)
                    styler = top_results.style.format({
                        "roi": "{:.2f}%",
                        "win_rate": "{:.2f}%",
                        "profit_factor": "{:.2f}"
                    })
                    
                    if hasattr(styler, 'map'):
                        styler = styler.map(color_metrics, subset=['roi'])
                    else:
                        styler = styler.applymap(color_metrics, subset=['roi'])
                        
                    st.dataframe(styler)

    with tab5:
        st.subheader("🏛️ Top US Indices & VIX Overview")
        st.write("Tracking S&P 500 (SPY), Nasdaq 100 (QQQ), Dow Jones (DIA), Volatility Index (^VIX), and US Dollar Index (DX-Y.NYB).")
        # Timeframe selector for indices/stocks (15m, 1h, 4h)
        timeframe = st.selectbox("Select timeframe", ["15m", "1h", "4h"], index=1)
        
        indices = ['SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB']
        index_stats = []
        missing_indices = []
        
        if st.button("Refresh Indices Data"):
            with st.spinner("Fetching US Indices..."):
                for sym in indices:
                    df_idx = fetch_and_analyze(sym, timeframe=timeframe, silent=True)
                    if df_idx is not None and not df_idx.empty:
                        current = df_idx.iloc[-1]
                        previous = df_idx.iloc[-2] if len(df_idx) > 1 else current
                        is_advancing = current['close'] > previous['close']
                        is_declining = current['close'] < previous['close']
                        # Estimate daily volume (last 7 1-hour bars = 7 trading hours)
                        est_vol = df_idx['volume'].tail(7).sum()
                        z_score = float(current['z_score']) if pd.notna(current['z_score']) else 0.0
                        vol_ma = float(current['vol_ma']) if pd.notna(current['vol_ma']) and float(current['vol_ma']) > 0 else np.nan
                        volume_ratio = float(current['volume']) / vol_ma if pd.notna(vol_ma) else 0.0
                        cmf_mean = df_idx['cmf'].rolling(window=20).mean().iloc[-1]
                        cmf_std = df_idx['cmf'].rolling(window=20).std().iloc[-1]
                        flow_z_score = (float(current['cmf']) - float(cmf_mean)) / float(cmf_std + 1e-9) if pd.notna(cmf_mean) and pd.notna(cmf_std) else 0.0
                        signal_score = (-z_score) + np.log1p(max(volume_ratio, 0.0)) + flow_z_score
                        
                        index_stats.append({
                            "Symbol": sym,
                            "Price": current['close'],
                            "Momentum": current['momentum'],
                            "RSI": current['rsi'],
                            "Trend": "Bullish 🟢" if current['close'] > current['ema_50'] else "Bearish 🔴",
                            "Advancing": is_advancing,
                            "Declining": is_declining,
                            "Est. Daily Volume": est_vol,
                            "Z-Score": z_score,
                            "Volume Ratio": volume_ratio,
                            "Flow Z-Score": flow_z_score,
                            "Signal Score": signal_score
                        })
                    else:
                        missing_indices.append(sym)
                        
                if index_stats:
                    df_ind = pd.DataFrame(index_stats)
                    advancing_count = int(df_ind['Advancing'].sum())
                    declining_count = int(df_ind['Declining'].sum())
                    total_count = len(df_ind)
                    breadth_ratio = advancing_count / declining_count if declining_count > 0 else None
                    if declining_count > 0:
                        breadth_ratio_label = f"{breadth_ratio:.2f} ({advancing_count}/{declining_count})"
                    else:
                        breadth_ratio_label = f"All advancing ({advancing_count}/{declining_count})" if advancing_count > 0 else f"No decliners ({advancing_count}/{declining_count})"
                    breadth_percent = advancing_count / total_count if total_count > 0 else 0.0
                    df_ind['Breadth Ratio'] = breadth_ratio_label
                    df_ind['Breadth %'] = breadth_percent
                    df_display = df_ind.drop(columns=['Advancing', 'Declining'])
                    df_display = df_display[[
                        "Symbol", "Breadth Ratio", "Breadth %", "Price", "Momentum", "RSI",
                        "Trend", "Signal Score", "Volume Ratio", "Flow Z-Score",
                        "Z-Score", "Est. Daily Volume"
                    ]]
                    if missing_indices:
                        st.warning(f"Could not load: {', '.join(missing_indices)}")
                    st.caption("Breadth Ratio = Advancing / Declining. Breadth % = Advancing / Total. Signal Score = -Z-Score + ln(1 + Volume Ratio) + Flow Z-Score")
                    styler = df_display.style.format({
                        "Price": "${:.2f}",
                        "Momentum": "{:.2%}",
                        "RSI": "{:.2f}",
                        "Est. Daily Volume": format_large_number,
                        "Z-Score": "{:.2f}",
                        "Volume Ratio": "{:.2f}x",
                        "Flow Z-Score": "{:.2f}",
                        "Signal Score": "{:.2f}",
                        "Breadth %": "{:.0%}"
                    })
                    
                    if hasattr(styler, 'map'):
                        styler = styler.map(color_metrics, subset=['Momentum', 'Z-Score', 'Flow Z-Score', 'Signal Score', 'Breadth %'])
                    else:
                        styler = styler.applymap(color_metrics, subset=['Momentum', 'Z-Score', 'Flow Z-Score', 'Signal Score', 'Breadth %'])
                        
                    st.dataframe(styler, width='stretch')
        else:
            st.info("Click 'Refresh Indices Data' to load the latest metrics for US Markets.")
            
        st.divider()
        st.subheader("🏢 Top 10 US Stocks Overview")
        st.write("Tracking the top 10 US companies by market cap.")
        
        top_stocks = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM']
        stock_stats = []
        
        if st.button("Refresh Stocks Data"):
            with st.spinner("Fetching US Stocks..."):
                bullish_trends = 0
                positive_momentum = 0
                valid_stocks = 0
                
                for sym in top_stocks:
                    df_stock = fetch_and_analyze(sym, timeframe=timeframe, silent=True)
                    if df_stock is not None and not df_stock.empty:
                        current = df_stock.iloc[-1]
                        est_vol = df_stock['volume'].tail(7).sum()
                        
                        is_bullish = current['close'] > current['ema_50']
                        if is_bullish: bullish_trends += 1
                        if current['momentum'] > 0: positive_momentum += 1
                        valid_stocks += 1
                        
                        stock_stats.append({
                            "Symbol": sym,
                            "Price": current['close'],
                            "Momentum": current['momentum'],
                            "RSI": current['rsi'],
                            "Trend": "Bullish 🟢" if is_bullish else "Bearish 🔴",
                            "Est. Daily Volume": est_vol,
                            "Z-Score": current['z_score']
                        })
                        
                if stock_stats:
                    # --- Market Health Score UI ---
                    health_score = (bullish_trends / valid_stocks) * 100
                    health_status = "🟢 STRONG BULL" if health_score >= 70 else "🔴 BEARISH" if health_score <= 30 else "🟡 NEUTRAL"
                    
                    hc1, hc2, hc3 = st.columns(3)
                    hc1.metric("Stocks in Bullish Trend", f"{bullish_trends} / {valid_stocks}")
                    hc2.metric("Positive Momentum", f"{positive_momentum} / {valid_stocks}")
                    hc3.metric("Overall Health", health_status)
                    
                    st.write("") # Spacer
                    
                    df_stocks = pd.DataFrame(stock_stats)
                    styler_stocks = df_stocks.style.format({
                        "Price": "${:.2f}",
                        "Momentum": "{:.2%}",
                        "RSI": "{:.2f}",
                        "Est. Daily Volume": format_large_number,
                        "Z-Score": "{:.2f}"
                    })
                    
                    if hasattr(styler_stocks, 'map'):
                        styler_stocks = styler_stocks.map(color_metrics, subset=['Momentum', 'Z-Score'])
                    else:
                        styler_stocks = styler_stocks.applymap(color_metrics, subset=['Momentum', 'Z-Score'])
                        
                    st.dataframe(styler_stocks, width='stretch')
        else:
            st.info("Click 'Refresh Stocks Data' to load the latest metrics for Top US Stocks.")

    with tab6:
        st.subheader("🎯 Composite Derivative Backtest")
        st.write("Backtest all composite derivative entry/exit signals: MACD, ATR, MA20, money flow, and volume ratio.")
        
        # Asset selection for backtest
        deriv_symbol = st.text_input("Enter swap pair (e.g., BTC/USDT:USDT)", value="BTC/USDT:USDT")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            backtest_timeframe = st.selectbox("Backtest timeframe", ["5m", "15m", "1h", "4h"], index=2, key="bt_tf")
        with col2:
            backtest_flow_timeframe = st.selectbox("Flow timeframe", ["5m", "15m", "1h", "4h"], index=2, key="bt_flow_tf")
        with col3:
            lookback_option = st.radio("Date Range", ["Last N Days", "Custom Range"], index=0, key="bt_lookback_option")
        
        if lookback_option == "Last N Days":
            lookback_days = st.slider("Lookback days", min_value=7, max_value=365, value=30, step=1)
            start_date_param = None
            end_date_param = None
        else:
            col_date1, col_date2 = st.columns(2)
            with col_date1:
                start_date_param = st.date_input("Start date", value=(datetime.now() - timedelta(days=30)))
            with col_date2:
                end_date_param = st.date_input("End date", value=datetime.now())
            lookback_days = 30
        
        if st.button("Run Composite Derivative Backtest"):
            if deriv_symbol.strip():
                with st.spinner(f"Running backtest for {deriv_symbol}..."):
                    # Convert dates to datetime if using custom range
                    if lookback_option == "Custom Range":
                        start_date_param = pd.Timestamp(start_date_param) if start_date_param else None
                        end_date_param = pd.Timestamp(end_date_param) if end_date_param else None
                    else:
                        start_date_param = None
                        end_date_param = None
                    
                    stats = backtest_composite_derivative(
                        deriv_symbol, 
                        timeframe=backtest_timeframe, 
                        flow_timeframe=backtest_flow_timeframe,
                        start_date=start_date_param,
                        end_date=end_date_param,
                        lookback_days=lookback_days
                    )
                    
                    if stats:
                        # Display metrics
                        m1, m2, m3, m4, m5 = st.columns(5)
                        m1.metric("Final Balance", f"${stats['final_balance']:.2f}")
                        m2.metric("ROI", f"{stats['roi']:.2f}%")
                        m3.metric("Total Trades", int(stats['total_trades']))
                        m4.metric("Win Rate", f"{stats['win_rate']:.1f}%")
                        m5.metric("Profit Factor", f"{stats['profit_factor']:.2f}")
                        
                        m6, m7 = st.columns(2)
                        m6.metric("Max Drawdown", f"{stats['max_drawdown']:.2f}%")
                        
                        st.divider()
                        st.subheader("Trade History")
                        if stats['trade_history']:
                            trade_df = pd.DataFrame(stats['trade_history'])
                            trade_df['pnl_pct'] = trade_df['pnl'] * 100
                            styler = trade_df.style.format({
                                'price_entry': '${:.2f}',
                                'price_exit': '${:.2f}',
                                'pnl_pct': '{:.2f}%'
                            })
                            if hasattr(styler, 'map'):
                                styler = styler.map(color_metrics, subset=['pnl_pct'])
                            else:
                                styler = styler.applymap(color_metrics, subset=['pnl_pct'])
                            st.dataframe(styler, width='stretch')
                        else:
                            st.warning("No trades generated during backtest period.")
                    else:
                        st.error(f"Could not run backtest for {deriv_symbol}. Check symbol format and try again.")
            else:
                st.warning("Please enter a valid swap pair (e.g., BTC/USDT:USDT).")

    with tab7:
        st.subheader("🌌 Volatility Quantum Analysis")
        st.info("""
        This tab visualizes the volatility term structure using a "quantum probability surface" as described.
        - **Left Plot (3D Surface)**: Shows the quantum probability `|ψ|²` of the market being in a specific volatility state (x-axis) with a certain momentum (y-axis). The color represents the phase `Arg(ψ)`, indicating the direction of probability flow. Peaks are metastable states.
        - **Right Plot (2D Histogram)**: Shows the classical, empirical distribution of volatility for comparison. It captures where the system has been, but not the complex phase relationships between states.
        """)
        
        col_q1, col_q2, col_q3 = st.columns([2, 2, 1])
        with col_q1:
            quantum_symbol = st.selectbox("Select Index for Analysis", options=['SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB', '^FTSE', 'XAUUSD=X', 'GBPUSD=X'], index=0, key="quantum_sym")
        with col_q2:
            quantum_timeframe = st.selectbox("Select Timeframe", ["15m", "1h", "4h"], index=1, key="quantum_tf")
        with col_q3:
            st.write("") # Spacer
            st.write("") # Spacer
            auto_refresh_quantum = st.checkbox("🔄 Auto-Refresh", key="quantum_refresh")

        if st.button(f"Generate Volatility Surface for {quantum_symbol}") or auto_refresh_quantum:
            with st.spinner(f"Performing quantum analysis on {quantum_symbol}..."):
                # Fetch the last year of data for a meaningful surface
                df_quantum = fetch_and_analyze(quantum_symbol, timeframe=quantum_timeframe, start_date=(datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), silent=True)
                fig, prediction_data = plot_volatility_surface(df_quantum, quantum_symbol)
                if fig:
                    st.pyplot(fig)
                    
                    # --- Market Direction Prediction Section ---
                    st.divider()
                    st.subheader("🔮 Market Direction Prediction")
                    
                    # 1. Volatility Prediction
                    vol_prediction = "NEUTRAL"
                    if prediction_data.get('vol_grad_x', 0) > 0:
                        vol_prediction = "INCREASE (Breakout/Trend Likely)"
                    elif prediction_data.get('vol_grad_x', 0) < 0:
                        vol_prediction = "DECREASE (Consolidation Likely)"

                    # 2. Price Trend & Momentum
                    current_data = df_quantum.iloc[-1]
                    price_trend = "BULLISH" if current_data['close'] > current_data['ema_50'] else "BEARISH"
                    price_momentum = "POSITIVE" if current_data['rsi'] > 50 else "NEGATIVE"

                    # 3. Final Prediction Logic
                    final_prediction = "CONSOLIDATE / CHOP"
                    if "INCREASE" in vol_prediction:
                        if price_trend == "BULLISH" and price_momentum == "POSITIVE":
                            final_prediction = "MOVE UP ⬆️"
                        elif price_trend == "BEARISH" and price_momentum == "NEGATIVE":
                            final_prediction = "MOVE DOWN ⬇️"
                    
                    p_col1, p_col2, p_col3 = st.columns(3)
                    p_col1.metric("Volatility Prediction", vol_prediction)
                    p_col2.metric("Underlying Price Trend", price_trend)
                    p_col3.metric("Short-Term Momentum", price_momentum)
                    st.info(f"**Final Prediction:** The analysis suggests the market is most likely to **{final_prediction}**.")
                else:
                    st.warning(f"Could not generate volatility surface for {quantum_symbol}.")
            
            if auto_refresh_quantum:
                time.sleep(301) # Wait 5 minutes
                st.rerun()

if __name__ == "__main__":
     try:
         main()
     except Exception:
         import traceback
         traceback.print_exc()
         input("Press Enter to exit...")

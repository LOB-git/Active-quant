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
import plotly.graph_objects as go
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
def fetch_and_analyze(symbol='BTC/USDT', timeframe='1h', start_date=None, end_date=None, silent=False, limit=None):
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
            exchange = ccxt.binance({'enableRateLimit': True}) # Using binance.com for yfinance consistency
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
                fetch_limit = limit if limit is not None else 1000
                print(f"Fetching latest {fetch_limit} candles for live analysis of {symbol}...")
                all_ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=fetch_limit)
            
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
        df['atr14'] = df['tr'].ewm(alpha=1/14, adjust=False).mean()
        
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
        df['atr14'] = df['atr14'].fillna(0)
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

def get_quantum_signal_for_candle(df_slice):
    """
    Calculates the quantum volatility signal for a given historical slice of data.
    This is a non-plotting version of the logic in plot_volatility_surface.
    """
    try:
        # This function requires at least 50 data points for rolling calculations
        if len(df_slice) < 50:
            return "CONSOLIDATE / CHOP"

        # 1. Calculate volatility
        df = df_slice.copy()
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_ret'].rolling(window=20).std() * np.sqrt(252) # Annualized
        df.dropna(inplace=True)

        if df['volatility'].empty or len(df) < 2:
            return "CONSOLIDATE / CHOP"

        # 2. Define a "wave function" psi
        vol_series = df['volatility'].values
        vol_change = df['volatility'].diff().fillna(0).values
        psi = vol_series + 1j * vol_change
        phase = np.angle(psi)

        # 3. Get phase gradient to predict path
        grad_x = 0
        try:
            all_points = np.vstack((vol_series, vol_change)).T
            tree = cKDTree(all_points)
            current_pos = np.array([vol_series[-1], vol_change[-1]])
            distances, indices = tree.query(current_pos, k=min(10, len(all_points)))
            
            if len(indices) > 2:
                neighbor_points = all_points[indices]
                neighbor_phases = phase[indices]
                A = np.c_[neighbor_points, np.ones(len(indices))]
                gradient, _, _, _ = np.linalg.lstsq(A, neighbor_phases, rcond=None)
                grad_x = gradient[0]
        except Exception:
            grad_x = 0

        # --- Market Direction Prediction Logic ---
        # 1. Volatility Prediction
        vol_prediction = "NEUTRAL"
        if grad_x > 0:
            vol_prediction = "INCREASE"
        elif grad_x < 0:
            vol_prediction = "DECREASE"

        # 2. Price Trend & Momentum
        current_data = df.iloc[-1]
        price_trend = "BULLISH" if current_data['close'] > current_data['ema_50'] else "BEARISH"
        price_momentum = "POSITIVE" if current_data['rsi'] > 50 else "NEGATIVE"

        # 3. Final Prediction Logic
        final_prediction = "CONSOLIDATE / CHOP"
        if "INCREASE" in vol_prediction:
            if price_trend == "BULLISH" and price_momentum == "POSITIVE":
                final_prediction = "MOVE UP"
            elif price_trend == "BEARISH" and price_momentum == "NEGATIVE":
                final_prediction = "MOVE DOWN"
        
        return final_prediction

    except Exception as e:
        # Return neutral signal if any error occurs during calculation
        return "CONSOLIDATE / CHOP"

def backtest_composite_derivative(df_full, symbol, risk_reward_ratio=None, **kwargs):
    """
    Backtests using the Volatility Quantum Analysis signal.
    - Enters LONG on "MOVE UP" signal.
    - Enters SHORT on "MOVE DOWN" signal.
    - Exits any position on "CONSOLIDATE / CHOP" signal.
    - Also exits on stop-loss (1.5 * ATR) or take-profit (stop_loss_distance * R:R ratio).
    """
    if df_full is None or df_full.empty or len(df_full) < 100:
        st.warning("Not enough data for a meaningful backtest (requires at least 100 candles).")
        return None

    initial_balance = 10000
    balance = initial_balance
    position = None # None, 'LONG', 'SHORT'
    entry_price = 0
    entry_date = None
    stop_loss_price = 0
    take_profit_price = 0
    trades = []
    trade_history = []
    peak_balance = initial_balance
    max_drawdown = 0
    
    # Start backtest after a warmup period for indicators
    warmup_period = 50
    
    # Iterate through history
    progress_bar = st.progress(0)
    total_steps = len(df_full) - warmup_period - 1
    for i in range(warmup_period, len(df_full) - 1):
        # For each step, we use all historical data up to that point to make a decision
        df_slice = df_full.iloc[:i+1]
        
        # Get the signal for the current candle
        signal = get_quantum_signal_for_candle(df_slice)
        
        # We execute trades at the OPEN of the next candle
        next_open = df_full.iloc[i+1]['open']
        candle_high = df_full.iloc[i+1]['high']
        candle_low = df_full.iloc[i+1]['low']
        
        # --- Stop-Loss & Take-Profit Logic ---
        if position == 'LONG':
            # Check for stop-loss or take-profit hit during the candle's lifetime
            if candle_low <= stop_loss_price:
                pnl = (stop_loss_price - entry_price) / entry_price
                balance *= (1 + pnl)
                trades.append(pnl)
                trade_history.append({'entry_date': entry_date, 'entry_price': entry_price, 'exit_date': df_full.index[i+1], 'exit_price': stop_loss_price, 'type': 'LONG', 'pnl': pnl, 'exit_reason': 'Stop-Loss'})
                position = None
            elif risk_reward_ratio and candle_high >= take_profit_price:
                pnl = (take_profit_price - entry_price) / entry_price
                balance *= (1 + pnl)
                trades.append(pnl)
                trade_history.append({'entry_date': entry_date, 'entry_price': entry_price, 'exit_date': df_full.index[i+1], 'exit_price': take_profit_price, 'type': 'LONG', 'pnl': pnl, 'exit_reason': 'Take-Profit'})
                position = None
        elif position == 'SHORT':
            if candle_high >= stop_loss_price:
                pnl = (entry_price - stop_loss_price) / entry_price
                balance *= (1 + pnl)
                trades.append(pnl)
                trade_history.append({'entry_date': entry_date, 'entry_price': entry_price, 'exit_date': df_full.index[i+1], 'exit_price': stop_loss_price, 'type': 'SHORT', 'pnl': pnl, 'exit_reason': 'Stop-Loss'})
                position = None
            elif risk_reward_ratio and candle_low <= take_profit_price:
                pnl = (entry_price - take_profit_price) / entry_price
                balance *= (1 + pnl)
                trades.append(pnl)
                trade_history.append({'entry_date': entry_date, 'entry_price': entry_price, 'exit_date': df_full.index[i+1], 'exit_price': take_profit_price, 'type': 'SHORT', 'pnl': pnl, 'exit_reason': 'Take-Profit'})
                position = None
        
        # --- Exit Logic ---
        if position and signal == "CONSOLIDATE / CHOP":
            if position == 'LONG':
                pnl = (next_open - entry_price) / entry_price
                balance *= (1 + pnl)
                trades.append(pnl)
                trade_history.append({
                    'entry_date': entry_date, 'entry_price': entry_price,
                    'exit_date': df_full.index[i+1], 'exit_price': next_open,
                    'type': 'LONG', 'pnl': pnl, 'exit_reason': 'Consolidate Signal'
                })
                position = None
            elif position == 'SHORT':
                pnl = (entry_price - next_open) / entry_price
                balance *= (1 + pnl)
                trades.append(pnl)
                trade_history.append({
                    'entry_date': entry_date, 'entry_price': entry_price,
                    'exit_date': df_full.index[i+1], 'exit_price': next_open,
                    'type': 'SHORT', 'pnl': pnl, 'exit_reason': 'Consolidate Signal'
                })
                position = None
        
        # --- Entry Logic ---
        if position is None:
            if signal == "MOVE UP":
                position = 'LONG'
                entry_price = next_open
                entry_date = df_full.index[i+1]
                atr_val = df_full.iloc[i]['atr14']
                stop_loss_price = entry_price - (atr_val * 1.5)
                if risk_reward_ratio:
                    risk_distance = entry_price - stop_loss_price
                    take_profit_price = entry_price + (risk_distance * risk_reward_ratio)
            elif signal == "MOVE DOWN":
                position = 'SHORT'
                entry_price = next_open
                entry_date = df_full.index[i+1]
                atr_val = df_full.iloc[i]['atr14']
                stop_loss_price = entry_price + (atr_val * 1.5)
                if risk_reward_ratio:
                    risk_distance = stop_loss_price - entry_price
                    take_profit_price = entry_price - (risk_distance * risk_reward_ratio)

        # Track Drawdown
        if balance > peak_balance:
            peak_balance = balance
        drawdown = (peak_balance - balance) / peak_balance
        if drawdown > max_drawdown:
            max_drawdown = drawdown
            
        progress_bar.progress((i - warmup_period) / total_steps)

    # Final PnL & Stats
    roi = ((balance - initial_balance) / initial_balance) * 100
    win_rate = (len([t for t in trades if t > 0]) / len(trades) * 100) if trades else 0
    gross_profit = sum([t for t in trades if t > 0])
    gross_loss = abs(sum([t for t in trades if t < 0]))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
    
    return {
        "final_balance": balance, "roi": roi, "total_trades": len(trades),
        "win_rate": win_rate, "profit_factor": profit_factor,
        "max_drawdown": max_drawdown * 100, "trade_history": trade_history,
        "rr_ratio": risk_reward_ratio if risk_reward_ratio else "None"
    }

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
    # Use Bybit for derivatives as it has fewer geographical restrictions than Binance
    exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
    try:
        tickers = exchange.fetch_tickers()
    except Exception as e:
        st.error(f"Could not fetch derivative tickers: {e}")
        st.error(f"Could not fetch derivative tickers from {exchange.id}: {e}")
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
            base_sym = sym.replace(':USDT', '') # Bybit uses 'BTC/USDT:USDT', Binance uses 'BTC/USDT'
            funding_rates[base_sym] = d.get('fundingRate', 0.0)
    except Exception:
        pass

    # Fetch open interest data for swap pairs (map base symbol -> oi info)
    oi_map = {}
    try:
        oi_data = exchange.fetch_open_interests()
        for k, v in oi_data.items():
            base = k.split(':')[0]
            base = k.replace(':USDT', '')
            oi_map[base] = v
    except Exception:
        oi_map = {}

    stats = []
    progress_bar = st.progress(0)

    def process_pair(pair_data):
        full_symbol, ticker = pair_data
        base_symbol = full_symbol.split(':')[0]
        base_symbol = full_symbol.replace(':USDT', '')
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
            if df.empty: # Add a check here
                return None
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-9)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Add EMA50 for quantum prediction function
            df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
            df.dropna(inplace=True) # Drop NaNs from indicators before quantum analysis

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
                'quantum_verdict': get_quantum_signal_for_candle(df),
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
                'entry_signal': entry_signal,
                'vol_ratio': vol_ratio,
                'atr14': float(current['atr14']) if not pd.isna(current['atr14']) else 0.0
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
        ax1.legend() # Add legend again to include the quiver plot label
        
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
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["📊 Market Overview", "⚡ Top Crypto Ranking", "🔥 Derivatives Trend Scan", "🛠️ Backtest Engine", "🏛️ US Indices", "🎯 Composite Derivative Backtest", "🌌 Volatility Quantum Analysis", "📈 Volatility Dashboard", "🇬🇧 GBP/USD Quantum Backtest", "🛡️ Options Analysis (GEX)"])

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

        # Initialize session state to hold the dataframe from the scan
        if 'df_deriv' not in st.session_state:
            st.session_state.df_deriv = pd.DataFrame()

        if st.button("Scan Top Derivatives"):
            with st.spinner("Scanning top derivative assets..."):
                df_deriv = scan_top_derivative_assets(timeframe=timeframe_deriv, flow_timeframe=flow_timeframe, volume_timeframe=volume_timeframe, top_n=100)
                if df_deriv is not None and not df_deriv.empty:
                    # Store the results in session state to persist them
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
                    st.session_state.df_deriv = df_deriv
                else:
                    st.warning("No derivative asset data returned. Try again in a moment.")

        # --- Display and Analysis Section (runs if data exists in session state) ---
        if not st.session_state.df_deriv.empty:
            df_deriv = st.session_state.df_deriv.copy() # Work with a copy

            # --- Calculate Quant Composite Strength Score (QS) ---
            # Calculate Z-scores for each component across the universe of scanned assets
            df_deriv['z_momentum'] = (df_deriv['momentum'] - df_deriv['momentum'].mean()) / df_deriv['momentum'].std()
            df_deriv['z_flow'] = (df_deriv['money_flow_signal'] - df_deriv['money_flow_signal'].mean()) / df_deriv['money_flow_signal'].std()
            df_deriv['z_volume'] = (df_deriv['vol_ratio'] - df_deriv['vol_ratio'].mean()) / df_deriv['vol_ratio'].std()
            df_deriv['z_volatility'] = (df_deriv['atr14'] - df_deriv['atr14'].mean()) / df_deriv['atr14'].std()
            df_deriv['z_trend'] = (df_deriv['z_score'] - df_deriv['z_score'].mean()) / df_deriv['z_score'].std() # Z-score of the Z-score

            # Apply the QS formula
            df_deriv['qs_score'] = (
                0.35 * df_deriv['z_momentum'] +
                0.25 * df_deriv['z_flow'] +
                0.20 * df_deriv['z_volume'] -
                0.10 * df_deriv['z_volatility'] +
                0.10 * df_deriv['z_trend']
            )

            # Calculate QS relative to BTC
            btc_qs = df_deriv[df_deriv['symbol'] == 'BTC/USDT']['qs_score'].iloc[0] if 'BTC/USDT' in df_deriv['symbol'].values else 0
            df_deriv['qs_rel_btc'] = df_deriv['qs_score'] - btc_qs

            # Sort by the new QS score
            df_deriv.sort_values(by='qs_score', ascending=False, inplace=True)

            # --- Calculate "Pump Potential" Score (User Request) ---
            # Criteria: High Z-Flow, High Z-Volume, and Low RSI.
            # We will use rsi_15m for a faster-reacting momentum component.
            # We create a "low_rsi_score" that is higher when RSI is lower.
            # An RSI of 20 gets a high score, an RSI of 80 gets a low score.
            df_deriv['low_rsi_score'] = (50 - df_deriv['rsi_15m']) / 50.0 # Scale to ~ -0.6 to +0.6

            df_deriv['pump_score'] = (
                (0.4 * df_deriv['z_flow']) +
                (0.4 * df_deriv['z_volume']) +
                (0.2 * df_deriv['low_rsi_score'])
            ).fillna(0)

            df_deriv.sort_values(by='qs_score', ascending=False, inplace=True)

            st.caption("TPS = (0.4 × Z-score Signal) + (0.3 × Money Flow Signal) + (0.3 × Funding Signal)")
            styler = df_deriv.style.format({
                "qs_score": "{:.2f}",
                "qs_rel_btc": "{:+.2f}",
                "quantum_verdict": "{}",
                "pump_score": "{:.2f}",
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

            def color_qs(val):
                if val > 2: return 'background-color: #0a8a0a; color: white' # Very Strong
                if val > 1: return 'background-color: #90ee90' # Strong
                if val < -2: return 'background-color: #a52a2a; color: white' # Very Weak
                if val < -1: return 'background-color: #f08080' # Weak
                return ''

            if hasattr(styler, 'map'):
                styler = styler.map(color_metrics, subset=['momentum', 'z_score', 'money_flow_signal', 'funding_signal', 'tps', 'liquidity_ratio', 'net_flow', 'rsi_15m', 'entry_score', 'quantum_verdict', 'qs_rel_btc', 'pump_score'])
                styler = styler.apply(lambda x: [color_qs(v) for v in x], subset=['qs_score'])
            else:
                styler = styler.applymap(color_metrics, subset=['momentum', 'z_score', 'money_flow_signal', 'funding_signal', 'tps', 'liquidity_ratio', 'net_flow', 'rsi_15m', 'entry_score', 'quantum_verdict', 'qs_rel_btc', 'pump_score'])
                styler = styler.apply(lambda x: [color_qs(v) for v in x], subset=['qs_score'])
            st.dataframe(styler, width='stretch')

            # --- Display Top 10 Pump Score Assets ---
            st.subheader("🚀 Top 10 Pump Score Candidates")
            st.write("These assets have the best combination of high relative money flow (Z-Flow), high relative volume (Z-Volume), and low RSI (room to grow).")
            top_10_pump = df_deriv.sort_values(by='pump_score', ascending=False).head(10)
            st.dataframe(top_10_pump[['symbol', 'pump_score', 'z_flow', 'z_volume', 'rsi_15m']].style.format(
                {'pump_score': '{:.2f}', 'z_flow': '{:.2f}', 'z_volume': '{:.2f}', 'rsi_15m': '{:.1f}'}
            ).background_gradient(subset=['pump_score', 'z_flow', 'z_volume'], cmap='Greens'))

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

            # --- Volatility Quantum Analysis Section ---
            st.divider()
            st.subheader("🌌 Volatility Quantum Analysis for Scanned Asset")
            st.info("Select an asset from the scan results above to perform a deep-dive volatility analysis.")

            col_q1, col_q2 = st.columns([1, 1])
            with col_q1:
                # Create a list of symbols from the scan results
                scanned_symbols = df_deriv['symbol'].tolist()
                # Add other relevant symbols that might not be in the top derivatives list
                for sym in ['SPY', 'QQQ', '^VIX', 'GBPUSD=X']:
                    if sym not in scanned_symbols:
                        scanned_symbols.append(sym)
                
                quantum_symbol_deriv = st.selectbox("Select Asset for Analysis", options=scanned_symbols, key="quantum_sym_deriv")
            with col_q2:
                quantum_timeframe_deriv = st.selectbox("Select Timeframe", ["15m", "1h", "4h"], index=1, key="quantum_tf_deriv")

            if st.button(f"Generate Volatility Surface for {quantum_symbol_deriv}", key="gen_surface_deriv"):
                with st.spinner(f"Performing quantum analysis on {quantum_symbol_deriv}..."):
                    # Fetch maximum available data for the surface instead of a fixed 1-year lookback.
                    # This allows analysis on newer assets with less history.
                    df_quantum = fetch_and_analyze(quantum_symbol_deriv, timeframe=quantum_timeframe_deriv, start_date=None, silent=True)

                    if df_quantum is not None and not df_quantum.empty:
                        fig, prediction_data = plot_volatility_surface(df_quantum, quantum_symbol_deriv)
                        if fig:
                            st.pyplot(fig)
                            
                            # Market Direction Prediction Logic
                            vol_prediction = "NEUTRAL"
                            if prediction_data.get('vol_grad_x', 0) > 0:
                                vol_prediction = "INCREASE (Breakout/Trend Likely)"
                            elif prediction_data.get('vol_grad_x', 0) < 0:
                                vol_prediction = "DECREASE (Consolidation Likely)"

                            st.info(f"**Final Prediction:** The analysis suggests the market is most likely to **{get_quantum_signal_for_candle(df_quantum)}**.")
                    else:
                        st.warning(f"Could not generate volatility surface for {quantum_symbol_deriv}. Not enough data available.")
            
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
        
        if 'index_stats' not in st.session_state:
            st.session_state.index_stats = []
        
        if st.button("Refresh Indices Data"):
            with st.spinner("Fetching US Indices Data..."):
                indices = ['SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB'] # Main indices for this table
                index_stats = []
                missing_indices = []
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
                
                # Save the results to session state
                st.session_state.index_stats = index_stats

        # Display the table if data exists in session state
        if st.session_state.index_stats:
            df_ind = pd.DataFrame(st.session_state.index_stats)
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

        # --- Real-Time Order Flow Table ---
        st.divider()
        st.subheader("📊 Real-Time Order Flow")
        c1, c2 = st.columns(2)
        with c1:
            flow_tf = st.selectbox("Select Order Flow Timeframe", options=['5m', '15m', '1h', '4h'], index=2)
        with c2:
            flow_candles = st.number_input("Number of Candles to Analyze", min_value=50, max_value=1000, value=500, step=50)
        
        if st.button("Refresh Order Flow"):
            with st.spinner(f"Calculating order flow for {flow_tf} timeframe..."):
                indices = ['SPY', 'QQQ', 'DIA', '^VIX', 'DX-Y.NYB']
                order_flow_data = []
                for sym in indices:
                    df_flow_calc = fetch_and_analyze(sym, timeframe=flow_tf, silent=True, limit=flow_candles)
                    if df_flow_calc is not None and not df_flow_calc.empty:
                        df_flow_calc['is_up'] = df_flow_calc['close'] >= df_flow_calc['open']
                        inflow = float(df_flow_calc.loc[df_flow_calc['is_up'], 'volume'].sum())
                        outflow = float(df_flow_calc.loc[~df_flow_calc['is_up'], 'volume'].sum())
                        order_flow_data.append({
                            'Symbol': sym,
                            'Inflow': inflow,
                            'Outflow': outflow,
                            'Net Flow': inflow - outflow,
                            'Money Flow Signal': (inflow - outflow) / (inflow + outflow) if (inflow + outflow) > 0 else 0
                        })
                st.session_state.order_flow_data = (order_flow_data, flow_candles, flow_tf)

        if 'order_flow_data' in st.session_state and st.session_state.order_flow_data and st.session_state.order_flow_data[0]:
            order_flow_data, flow_candles, flow_tf = st.session_state.order_flow_data
            st.caption(f"Shows buying vs. selling pressure based on the last {flow_candles} candles ({flow_tf} timeframe).")
            df_flow = pd.DataFrame(order_flow_data)
            df_flow.sort_values(by='Net Flow', ascending=False, inplace=True)

            styler_flow = df_flow.style.format({
                'Inflow': format_large_number,
                'Outflow': format_large_number,
                'Net Flow': format_large_number,
                'Money Flow Signal': '{:.2f}'
            })
            styler_flow = styler_flow.background_gradient(subset=['Net Flow'], cmap='RdYlGn')
            styler_flow = styler_flow.bar(subset=['Money Flow Signal'], align='zero', color=['#d65f5f', '#5fba7d'])
            st.dataframe(styler_flow, width='stretch')

        # --- QS Score Comparison for SPY, QQQ, DIA ---
        st.divider()
        st.subheader("Macro Quant Strength (QS) Comparison")
        st.caption("This table compares major indices and currency pairs using a relative strength model. A higher score indicates stronger performance versus the group.")

        qs_indices_data = []
        indices_for_qs = ['SPY', 'QQQ', 'DIA', 'DX-Y.NYB', '^FTSE', 'GBPUSD=X']
        for sym in indices_for_qs:
            df_idx_qs = fetch_and_analyze(sym, timeframe=timeframe, silent=True)
            if df_idx_qs is not None and not df_idx_qs.empty:
                current = df_idx_qs.iloc[-1]
                # Calculate money flow signal (inflow vs outflow)
                df_idx_qs['is_up'] = df_idx_qs['close'] >= df_idx_qs['open']
                inflow = float(df_idx_qs.loc[df_idx_qs['is_up'], 'volume'].sum())
                outflow = float(df_idx_qs.loc[~df_idx_qs['is_up'], 'volume'].sum())
                total_flow = inflow + outflow
                money_flow_signal = (inflow - outflow) / total_flow if total_flow > 0 else 0.0

                qs_indices_data.append({
                    'symbol': sym,
                    'momentum': current.get('momentum', 0.0),
                    'money_flow_signal': money_flow_signal,
                    'atr14': current.get('atr14', 0.0),
                    'z_score_raw': current.get('z_score', 0.0) # Price vs BBands
                })
        
        if len(qs_indices_data) >= 3: # Check if we have at least 3 to compare
            df_qs_indices = pd.DataFrame(qs_indices_data)
            
            # Calculate Z-Scores relative to each other
            df_qs_indices['z_momentum'] = (df_qs_indices['momentum'] - df_qs_indices['momentum'].mean()) / df_qs_indices['momentum'].std()
            df_qs_indices['z_flow'] = (df_qs_indices['money_flow_signal'] - df_qs_indices['money_flow_signal'].mean()) / df_qs_indices['money_flow_signal'].std()
            df_qs_indices['z_volatility'] = (df_qs_indices['atr14'] - df_qs_indices['atr14'].mean()) / df_qs_indices['atr14'].std()
            df_qs_indices['z_trend'] = (df_qs_indices['z_score_raw'] - df_qs_indices['z_score_raw'].mean()) / df_qs_indices['z_score_raw'].std()

            # Apply the QS formula (simplified for indices, no z_volume)
            df_qs_indices['qs_score'] = (
                (0.50 * df_qs_indices['z_momentum']) +
                (0.30 * df_qs_indices['z_flow']) -
                (0.10 * df_qs_indices['z_volatility']) +
                (0.10 * df_qs_indices['z_trend'])
            ).fillna(0)

            df_qs_indices.sort_values(by='qs_score', ascending=False, inplace=True)

            # Display the table
            display_cols = ['symbol', 'qs_score', 'z_momentum', 'z_flow', 'z_volatility', 'z_trend']
            styler_qs = df_qs_indices[display_cols].style.format({
                'qs_score': '{:.2f}',
                'z_momentum': '{:.2f}',
                'z_flow': '{:.2f}',
                'z_volatility': '{:.2f}',
                'z_trend': '{:.2f}'
            }).background_gradient(subset=['qs_score'], cmap='viridis')

            st.dataframe(styler_qs, width='stretch')
        else:
            st.warning("Could not fetch enough data for the Macro QS comparison.")

        # --- Key Support Levels (Put Support Proxy) ---
        st.divider()
        st.subheader("Key Support Levels (Put Support Proxy)")
        st.caption("These technical levels often act as strong support, similar to areas with high Put option open interest.")
        
        support_data = []
        for sym in indices_for_qs:
            # We can reuse the data we just fetched
            df_support = fetch_and_analyze(sym, timeframe=timeframe, silent=True)
            if df_support is not None and not df_support.empty:
                current = df_support.iloc[-1]
                support_data.append({
                    'Symbol': sym,
                    'Current Price': current['close'],
                    'Bullish Order Block': current.get('ob_bull', 0.0),
                    'Lower Bollinger Band': current.get('bb_lower', 0.0),
                    '200-Period MA': current.get('sma_200', 0.0)
                })
        
        if support_data:
            df_support_levels = pd.DataFrame(support_data)
            styler_support = df_support_levels.style.format({
                'Current Price': '${:,.2f}',
                'Bullish Order Block': '${:,.2f}',
                'Lower Bollinger Band': '${:,.2f}',
                '200-Period MA': '${:,.2f}'
            }).background_gradient(
                subset=['Bullish Order Block', 'Lower Bollinger Band', '200-Period MA'], 
                cmap='Reds_r' # Use a reverse red map to highlight lower values
            )
            st.dataframe(styler_support, width='stretch')
        else:
            st.info("Click 'Refresh Indices Data' to load the latest metrics for US Markets.")
            
        st.divider()
        st.subheader("🏢 Top 10 US Stocks Overview")
        st.write("Tracking the top 10 US companies by market cap.")
        
        # Initialize session state to hold the dataframe from the scan
        if 'df_stocks' not in st.session_state:
            st.session_state.df_stocks = pd.DataFrame()
            
        top_stocks = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM']
        
        if st.button("Refresh Stocks Data"):
            with st.spinner("Fetching and analyzing Top 10 US Stocks..."):
                qs_stocks_data = []
                
                for sym in top_stocks:
                    df_stock = fetch_and_analyze(sym, timeframe=timeframe, silent=True)
                    if df_stock is not None and not df_stock.empty:
                        current = df_stock.iloc[-1]

                        # Calculate money flow signal
                        df_stock['is_up'] = df_stock['close'] >= df_stock['open']
                        inflow = float(df_stock.loc[df_stock['is_up'], 'volume'].sum())
                        outflow = float(df_stock.loc[~df_stock['is_up'], 'volume'].sum())
                        total_flow = inflow + outflow
                        money_flow_signal = (inflow - outflow) / total_flow if total_flow > 0 else 0.0

                        qs_stocks_data.append({
                            'symbol': sym,
                            'price': current.get('close', 0.0),
                            'momentum': current.get('momentum', 0.0),
                            'money_flow_signal': money_flow_signal,
                            'atr14': current.get('atr14', 0.0),
                            'z_score_raw': current.get('z_score', 0.0),
                            'rsi': current.get('rsi', 50.0),
                            'trend_bullish': current.get('close', 0) > current.get('ema_50', 0)
                        })
                        
                if qs_stocks_data:
                    df_stocks = pd.DataFrame(qs_stocks_data)

                    # Calculate Z-Scores and QS Score
                    df_stocks['z_momentum'] = (df_stocks['momentum'] - df_stocks['momentum'].mean()) / df_stocks['momentum'].std()
                    df_stocks['z_flow'] = (df_stocks['money_flow_signal'] - df_stocks['money_flow_signal'].mean()) / df_stocks['money_flow_signal'].std()
                    df_stocks['z_volatility'] = (df_stocks['atr14'] - df_stocks['atr14'].mean()) / df_stocks['atr14'].std()
                    df_stocks['z_trend'] = (df_stocks['z_score_raw'] - df_stocks['z_score_raw'].mean()) / df_stocks['z_score_raw'].std()

                    df_stocks['qs_score'] = (
                        (0.50 * df_stocks['z_momentum']) +
                        (0.30 * df_stocks['z_flow']) -
                        (0.10 * df_stocks['z_volatility']) +
                        (0.10 * df_stocks['z_trend'])
                    ).fillna(0)

                    df_stocks.sort_values(by='qs_score', ascending=False, inplace=True)
                    st.session_state.df_stocks = df_stocks
                else:
                    st.warning("No stock data was returned from the fetch operation.")

        if not st.session_state.df_stocks.empty:
            df_stocks = st.session_state.df_stocks

            # --- Market Health Score UI ---
            bullish_trends = df_stocks['trend_bullish'].sum()
            positive_momentum = (df_stocks['momentum'] > 0).sum()
            valid_stocks = len(df_stocks)
            health_score = (bullish_trends / valid_stocks) * 100 if valid_stocks > 0 else 0
            health_status = "🟢 STRONG BULL" if health_score >= 70 else "🔴 BEARISH" if health_score <= 30 else "🟡 NEUTRAL"
            
            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("Stocks in Bullish Trend", f"{bullish_trends} / {valid_stocks}")
            hc2.metric("Positive Momentum", f"{positive_momentum} / {valid_stocks}")
            hc3.metric("Overall Health", health_status)

            display_cols = ['symbol', 'qs_score', 'price', 'momentum', 'rsi']
            styler_stocks = df_stocks[display_cols].style.format({
                "price": "${:.2f}",
                "qs_score": "{:.2f}",
                "momentum": "{:.2%}",
                "rsi": "{:.1f}"
            })

            if hasattr(styler_stocks, 'map'):
                styler_stocks = styler_stocks.map(color_metrics, subset=['momentum'])
            else:
                styler_stocks = styler_stocks.applymap(color_metrics, subset=['momentum'])

            st.dataframe(styler_stocks, width='stretch')
        else:
            st.info("Click 'Refresh Stocks Data' to load the latest metrics for Top US Stocks.")

    with tab6:
        st.subheader("🎯 Composite Derivative Backtest")
        st.write("Backtest all composite derivative entry/exit signals: MACD, ATR, MA20, money flow, and volume ratio.")
        
        # Asset selection for backtest
        deriv_symbol = st.text_input("Enter symbol (e.g., BTC/USDT or SPY)", value="BTC/USDT")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            rr_ratios = st.multiselect("Take Profit R:R Ratios", options=[1.0, 1.5, 2.0, 2.5, 3.0], default=[1.5, 2.0])
        with col2:
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
                with st.spinner(f"Fetching data for {deriv_symbol}..."):
                    start_date_str = None
                    if lookback_option == "Last N Days":
                        start_date_str = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
                    elif start_date_param:
                        start_date_str = start_date_param.strftime('%Y-%m-%d')
                
                    # Fetch all necessary data for the backtest period
                    df_backtest = fetch_and_analyze(symbol=deriv_symbol, timeframe=backtest_timeframe, start_date=start_date_str)
                
                if df_backtest is not None and not df_backtest.empty:
                    all_stats = []
                    # Also run a baseline without take-profit
                    ratios_to_run = [None] + rr_ratios
                    with st.spinner(f"Running {len(ratios_to_run)} backtest simulations..."):
                        for ratio in ratios_to_run:
                            stats = backtest_composite_derivative(df_backtest, deriv_symbol, risk_reward_ratio=ratio)
                            if stats:
                                all_stats.append(stats)
                    
                    if all_stats:
                        st.subheader("Backtest Results by R:R Ratio")
                        results_df = pd.DataFrame(all_stats)
                        results_df = results_df.set_index('rr_ratio')
                        display_cols = ['roi', 'total_trades', 'win_rate', 'profit_factor', 'max_drawdown', 'final_balance']
                        styler = results_df[display_cols].style.format({
                            'roi': '{:.2f}%',
                            'win_rate': '{:.1f}%',
                            'profit_factor': '{:.2f}',
                            'max_drawdown': '{:.2f}%',
                            'final_balance': '${:,.2f}'
                        }).background_gradient(subset=['roi', 'profit_factor'], cmap='viridis')
                        st.dataframe(styler, width='stretch')
                    else:
                        st.error("Backtest simulations failed to produce results.")
                else:
                    st.error(f"Could not run backtest for {deriv_symbol}. Check symbol format and data availability.")
            else:
                st.warning("Please enter a valid symbol.")

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
                            final_prediction = "MOVE UP"
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

    with tab8:
        st.subheader("🇬🇧 GBP/USD Volatility Quantum Backtest")
        st.write("This backtest uses the Volatility Quantum Analysis strategy specifically for the GBP/USD forex pair.")
        st.info("A trade is opened on a 'MOVE UP' or 'MOVE DOWN' signal and closed when the signal returns to 'CONSOLIDATE / CHOP'.")

        gbp_symbol = "GBPUSD=X"

        col1, col2, col3 = st.columns(3)
        with col1:
            gbp_backtest_timeframe = st.selectbox("Backtest timeframe", ["15m", "1h", "4h"], index=1, key="gbp_bt_tf")
        with col2:
            gbp_lookback_option = st.radio("Date Range", ["Last N Days", "Custom Range"], index=0, key="gbp_lookback_option")

        if gbp_lookback_option == "Last N Days":
            gbp_lookback_days = st.slider("Lookback days", min_value=7, max_value=730, value=90, step=1, key="gbp_lookback_days")
            gbp_start_date_param = None
        else:
            date_col1, date_col2 = st.columns(2)
            with date_col1:
                gbp_start_date_param = st.date_input("Start date", value=(datetime.now() - timedelta(days=90)), key="gbp_start_date")
            with date_col2:
                end_date_param = st.date_input("End date", value=datetime.now(), key="gbp_end_date")

        if st.button("Run GBP/USD Quantum Backtest"):
            with st.spinner(f"Fetching data and running Quantum Backtest for {gbp_symbol}..."):
                start_date_str = None
                if gbp_lookback_option == "Last N Days":
                    start_date_str = (datetime.now() - timedelta(days=gbp_lookback_days)).strftime('%Y-%m-%d')
                elif gbp_start_date_param:
                    start_date_str = gbp_start_date_param.strftime('%Y-%m-%d')

                # Fetch all necessary data for the backtest period
                df_backtest = fetch_and_analyze(symbol=gbp_symbol, timeframe=gbp_backtest_timeframe, start_date=start_date_str)

                stats = backtest_composite_derivative(df_backtest, gbp_symbol)

                if stats:
                    # Display metrics
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Final Balance", f"${stats['final_balance']:.2f}")
                    m2.metric("ROI", f"{stats['roi']:.2f}%")
                    m3.metric("Total Trades", int(stats['total_trades']))
                    m4.metric("Win Rate", f"{stats['win_rate']:.1f}%")
                    m5.metric("Profit Factor", f"{stats['profit_factor']:.2f}")

                    st.divider()
                    st.subheader("Trade History")
                    if stats['trade_history']:
                        trade_df = pd.DataFrame(stats['trade_history']).sort_values(by='entry_date')
                        trade_df['pnl_pct'] = trade_df['pnl'] * 100
                        styler = trade_df.style.format({
                            'entry_price': '{:.5f}', 'exit_price': '{:.5f}', 'pnl_pct': '{:.2f}%'
                        })
                        styler = styler.map(color_metrics, subset=['pnl_pct'])
                        st.dataframe(styler, width='stretch')
                    else:
                        st.warning("No trades were generated during this backtest period.")
                else:
                    st.error(f"Could not run backtest for {gbp_symbol}. Ensure data is available for the selected period.")

    with tab8:
        st.subheader("📈 Volatility Dashboard (GEX Proxy)")
        st.info("""
        This dashboard visualizes key price levels that often act like high Gamma Exposure zones, influencing volatility.
        - **Order Blocks (Red/Green Lines)**: Institutional support/resistance. Prices often reject from these levels.
        - **Fair Value Gaps (Shaded Zones)**: Price imbalances that act as magnets. The market tends to revisit these zones to 'rebalance' price.
        - **Bollinger Bands (Dashed Lines)**: Dynamic support/resistance. Prices are statistically 'expensive' or 'cheap' at these bands.
        """)

        c1, c2 = st.columns(2)
        with c1:
            # Note: SPX options are cash-settled and have different dynamics. SPY is the ETF equivalent and more suitable for this technical analysis.
            gamma_asset = st.selectbox("Select Asset", options=['SPY', 'QQQ', 'DIA', 'GBPUSD=X'], index=0, key='gamma_asset')
        with c2:
            gamma_tf = st.selectbox("Select Timeframe", options=['15m', '1h', '4h'], index=1, key='gamma_tf')

        if st.button("Generate Volatility Chart", key='gen_gamma_chart'):
            with st.spinner(f"Fetching data and identifying key levels for {gamma_asset}..."):
                df_gamma = fetch_and_analyze(gamma_asset, timeframe=gamma_tf, limit=200) # Fetch last 200 candles

                if df_gamma is not None and not df_gamma.empty:
                    # --- Create Interactive Plotly Chart ---
                    fig = go.Figure()

                    # 1. Candlestick Chart for Price
                    fig.add_trace(go.Candlestick(x=df_gamma.index,
                                                 open=df_gamma['open'],
                                                 high=df_gamma['high'],
                                                 low=df_gamma['low'],
                                                 close=df_gamma['close'],
                                                 name='Price'))

                    # 2. Bollinger Bands
                    fig.add_trace(go.Scatter(x=df_gamma.index, y=df_gamma['bb_upper'], mode='lines',
                                             line=dict(color='gray', width=1, dash='dash'), name='Bollinger Bands'))
                    fig.add_trace(go.Scatter(x=df_gamma.index, y=df_gamma['bb_lower'], mode='lines',
                                             line=dict(color='gray', width=1, dash='dash'), showlegend=False))

                    # 3. Order Blocks (Horizontal Lines)
                    fig.add_hline(y=df_gamma['ob_bear'].iloc[-1], line_width=2, line_dash="solid", line_color="red",
                                  annotation_text="Bearish Order Block", annotation_position="bottom right")
                    fig.add_hline(y=df_gamma['ob_bull'].iloc[-1], line_width=2, line_dash="solid", line_color="green",
                                  annotation_text="Bullish Order Block", annotation_position="top right")

                    # 4. Fair Value Gaps (Shaded Rectangles)
                    # Find the most recent FVG zones to plot
                    bull_fvg_top = df_gamma['last_bull_fvg_top'].iloc[-1]
                    bull_fvg_bottom = df_gamma['last_bull_fvg_bottom'].iloc[-1]
                    bear_fvg_top = df_gamma['last_bear_fvg_top'].iloc[-1]
                    bear_fvg_bottom = df_gamma['last_bear_fvg_bottom'].iloc[-1]

                    if bull_fvg_top > 0:
                        fig.add_hrect(y0=bull_fvg_bottom, y1=bull_fvg_top, line_width=0, fillcolor="green", opacity=0.2,
                                      annotation_text="Bullish FVG", annotation_position="top left")
                    if bear_fvg_top < 10000000:
                         fig.add_hrect(y0=bear_fvg_bottom, y1=bear_fvg_top, line_width=0, fillcolor="red", opacity=0.2,
                                       annotation_text="Bearish FVG", annotation_position="bottom left")

                    # 5. Update Layout for TradingView feel
                    fig.update_layout(
                        title=f"Key Volatility Levels for {gamma_asset} ({gamma_tf})",
                        yaxis_title='Price',
                        xaxis_rangeslider_visible=False, # Hide the range slider
                        template='plotly_dark', # Dark theme
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(f"Could not fetch data for {gamma_asset} on the {gamma_tf} timeframe.")

    with tab10:
        st.subheader("🛡️ Options Gamma Exposure (GEX)")
        st.info("""
        This tool analyzes real options data to calculate the total Gamma Exposure (GEX) of market makers. High GEX can suppress volatility, while certain strike levels can act as 'magnets' or 'pins' for the price.
        - **Total GEX**: The overall gamma imbalance. A large positive value suggests volatility suppression (a 'gamma trap').
        - **Zero Gamma**: The price level where market maker gamma exposure flips from positive to negative. This level can act as a pivot point.
        - **GEX Profile**: The bar chart shows which strike prices hold the most positive (call) and negative (put) gamma.
        """)

        gex_asset = st.selectbox("Select Asset (US Stocks/ETFs)", options=['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'TSLA', 'NVDA', 'AMZN'], key='gex_asset')

        @st.cache_data(ttl=600) # Cache for 10 minutes
        def calculate_gex(symbol):
            """Fetches options data and calculates Gamma Exposure."""
            try:
                ticker = yf.Ticker(symbol)
                current_price = ticker.history(period='1d')['Close'].iloc[-1]
                expirations = ticker.options
                
                # Limit to nearest expirations for performance
                expirations_to_scan = expirations[:min(5, len(expirations))]

                all_options = []
                for exp in expirations_to_scan:
                    opt_chain = ticker.option_chain(exp)
                    # Combine calls and puts, adding a 'type' column
                    opt_chain.calls['type'] = 'call'
                    opt_chain.puts['type'] = 'put'
                    all_options.append(opt_chain.calls)
                    all_options.append(opt_chain.puts)
                
                if not all_options:
                    return {"current_price": current_price}, "No options data found for this asset."

                df = pd.concat(all_options)
                df.fillna(0, inplace=True)

                # --- Enhanced Robustness Check ---
                # yfinance may provide a mix of contracts with and without greeks.
                # We will filter to only use rows where gamma and openInterest are available and non-zero.
                if 'gamma' not in df.columns or 'openInterest' not in df.columns:
                    return {"current_price": current_price}, "GEX calculation failed: The API did not provide 'gamma' or 'openInterest' data for this asset."

                # Filter for usable data
                df_valid = df[(df['gamma'] > 0) & (df['openInterest'] > 0)].copy()
                if df_valid.empty:
                    return {"current_price": current_price}, "GEX calculation failed: No valid options contracts with Gamma and Open Interest were found."

                # GEX = Gamma * Open Interest * 100 shares/contract
                # Puts have a negative impact on dealer gamma as they are short puts (long stock hedge)
                df_valid['gamma_exposure'] = df_valid['gamma'] * df_valid['openInterest'] * 100 * np.where(df_valid['type'] == 'put', -1, 1)
                
                # Group by strike to see the profile
                gex_profile = df_valid.groupby('strike')['gamma_exposure'].sum()

                total_gex = gex_profile.sum()

                # Find Zero Gamma Level (where cumulative GEX flips)
                cumulative_gex = gex_profile.sort_index().cumsum()
                zero_gamma_level = cumulative_gex[cumulative_gex > 0].index.min()

                return {
                    "total_gex": total_gex,
                    "zero_gamma_level": zero_gamma_level,
                    "gex_profile": gex_profile,
                    "current_price": current_price
                }, None
            except Exception as e:
                # Attempt to get price even if options fail
                try:
                    price = yf.Ticker(symbol).history(period='1d')['Close'].iloc[-1]
                    return {"current_price": price}, str(e)
                except:
                    return None, str(e)

        if st.button("Analyze Gamma Exposure", key='run_gex'):
            with st.spinner(f"Fetching options chain for {gex_asset}..."):
                gex_data, error = calculate_gex(gex_asset)
                if error and gex_data and 'current_price' in gex_data:
                    st.metric("Current Price", f"${gex_data.get('current_price', 0):,.2f}")
                    st.error(f"Could not calculate GEX: {error}")
                elif gex_data and 'gex_profile' in gex_data:
                    # --- Display Metrics ---
                    st.metric("Total GEX (Notional)", f"{gex_data['total_gex']:,.0f}")
                    st.metric("Zero Gamma Level", f"${gex_data['zero_gamma_level']:.2f}")
                    st.metric("Current Price", f"${gex_data['current_price']:.2f}")

                    # --- Create Enhanced Plotly Chart ---
                    gex_profile_df = gex_data['gex_profile'].reset_index()
                    gex_profile_df.columns = ['strike', 'gamma_exposure']

                    fig = go.Figure()

                    # Add GEX bars
                    fig.add_trace(go.Bar(x=gex_profile_df['strike'], y=gex_profile_df['gamma_exposure'], name='Gamma Exposure'))

                    # Add Zero Gamma line
                    fig.add_vline(x=gex_data['zero_gamma_level'], line_width=2, line_dash="dash", line_color="yellow",
                                  annotation_text="Zero Gamma", annotation_position="top right")

                    # Add Current Price line
                    fig.add_vline(x=gex_data['current_price'], line_width=2, line_dash="solid", line_color="cyan",
                                  annotation_text="Current Price", annotation_position="top left")

                    fig.update_layout(title=f'Gamma Exposure Profile for {gex_asset}', xaxis_title='Strike Price', yaxis_title='Gamma Exposure (Notional)', template='plotly_dark')
                    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
     try:
         main()
     except Exception:
         import traceback
         traceback.print_exc()
         input("Press Enter to exit...")

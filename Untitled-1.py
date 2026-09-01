import streamlit as st
import ccxt
import requests
import pandas as pd
import yfinance as yf
import numpy as np
import os
import time
import concurrent.futures
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg') # Fix for Streamlit/Matplotlib GUI errors
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import plotly.graph_objects as go
import streamlit.components.v1 as components

# Keep yfinance's SQLite caches in the writable project directory. This avoids
# "unable to open database file" failures when the dashboard runs sandboxed.
YFINANCE_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.yfinance-cache')
os.makedirs(YFINANCE_CACHE_DIR, exist_ok=True)
yf.set_tz_cache_location(YFINANCE_CACHE_DIR)

polygon_available = True
try:
    from polygon import RESTClient
except ImportError:
    polygon_available = False


# Set page config at the top level to avoid errors and define layout
st.set_page_config(page_title="Quant Scalper 1h", layout="wide")


def build_binance_gex_proxy(current_price, recent_volatility=None, strike_count=21):
    """Build a simple GEX-style proxy profile when Binance options data is unavailable."""
    if current_price is None or current_price <= 0:
        return None

    if recent_volatility is None or recent_volatility <= 0:
        recent_volatility = 0.03

    half_width = max(current_price * 0.2, current_price * recent_volatility * 4)
    strikes = np.linspace(current_price - half_width, current_price + half_width, strike_count)

    distance = strikes - current_price
    scale = max(current_price * recent_volatility * 0.75, 1.0)
    weights = np.exp(-(distance / scale) ** 2)
    signed_weights = np.where(distance >= 0, -weights, weights)

    profile = pd.Series(signed_weights * (current_price * 0.8), index=strikes)
    profile.index.name = 'strike'
    return profile


def summarize_gex_profile(gex_profile, current_price):
    """Summarize a GEX-style profile into total GEX and a zero-Gamma level."""
    if gex_profile is None:
        return None

    cumulative_gex = gex_profile.sort_index().cumsum()
    positive_series = cumulative_gex[cumulative_gex > 0]
    zero_gamma_level = float(positive_series.index.min()) if not positive_series.empty else float(current_price or 0)

    return {
        'total_gex': float(gex_profile.sum()),
        'zero_gamma_level': zero_gamma_level,
        'gex_profile': gex_profile,
        'current_price': float(current_price or 0),
    }


def get_intraday_money_flow(symbol, interval='5m', period='1d'):
    """Get today's intraday inflow/outflow notional for an ETF based on up vs down candles."""
    try:
        yf_symbol = 'BTC-USD' if symbol == 'BTC/USD' else symbol
        data = yf.download(yf_symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        if data is None or data.empty:
            return None

        df = data.reset_index()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df.columns = [str(col).strip().lower() for col in df.columns]

        if 'datetime' in df.columns:
            df = df.rename(columns={'datetime': 'date'})
        elif 'date' in df.columns:
            df = df.rename(columns={'date': 'date'})
        elif 'index' in df.columns:
            df = df.rename(columns={'index': 'date'})

        if 'close' not in df.columns or 'open' not in df.columns or 'volume' not in df.columns:
            return None

        df['date'] = pd.to_datetime(df['date'], utc=True, errors='coerce')
        df = df.dropna(subset=['date']).sort_values('date')
        df['is_up'] = df['close'] >= df['open']
        df['money_flow'] = df['close'] * df['volume']

        df['cumulative_inflow'] = np.where(df['is_up'], df['money_flow'], np.nan)
        df['cumulative_outflow'] = np.where(~df['is_up'], df['money_flow'], np.nan)
        df['cumulative_inflow'] = pd.Series(df['cumulative_inflow']).cumsum().fillna(0)
        df['cumulative_outflow'] = pd.Series(df['cumulative_outflow']).cumsum().fillna(0)
        df['cumulative_net_flow'] = df['cumulative_inflow'] - df['cumulative_outflow']

        inflow = float(df.loc[df['is_up'], 'money_flow'].sum())
        outflow = float(df.loc[~df['is_up'], 'money_flow'].sum())
        net_flow = inflow - outflow
        return {
            'symbol': symbol,
            'inflow': inflow,
            'outflow': outflow,
            'net_flow': net_flow,
            'flow_ratio': net_flow / (inflow + outflow) if (inflow + outflow) > 0 else 0.0,
            'bars': int(len(df)),
            'history': df[['date', 'open', 'close', 'volume', 'is_up', 'money_flow', 'cumulative_net_flow']].copy()
        }
    except Exception:
        return None


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
        raw_symbol = str(symbol).strip()
        symbol_aliases = {
            'XAUUSD': 'GC=F',
            'XAUUSD=X': 'GC=F',
            'XAGUSD': 'SI=F',
            'XAGUSD=X': 'SI=F',
            'BTC/USD': 'BTC/USDT',
        }
        symbol = symbol_aliases.get(raw_symbol.upper(), raw_symbol)

        stock_index_symbols = ['SPY', 'QQQ', 'DIA', 'NQ=F', 'YM=F', '^VIX', 'DX-Y.NYB', 'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 'LLY', 'JPM', 'GBPUSD=X', '^FTSE', 'XAUUSD', 'XAUUSD=X', 'GC=F', 'GLD']
        is_stock_index = symbol in stock_index_symbols or raw_symbol in stock_index_symbols or raw_symbol.upper() in stock_index_symbols

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
                    df = yf.download(
                        symbol,
                        start=effective_start_date,
                        end=end_date,
                        interval=yf_interval,
                        progress=False,
                        prepost=True,
                    )
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

    emoji = "🟢" if ("BUY" in signal or "POSITIVE" in signal) else "🔴"
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
        
        # Add a check to ensure the next candle's data is valid before proceeding
        if pd.isna(next_open) or pd.isna(candle_high):
            continue

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

            # Estimate a simple GEX-notional proxy from price and recent volatility.
            total_gex = 0.0
            try:
                current_price = float(current['close'])
                returns = df['close'].pct_change().dropna()
                recent_volatility = float(returns.std() * np.sqrt(24)) if len(returns) > 1 else 0.03
                proxy_profile = build_binance_gex_proxy(current_price, recent_volatility=recent_volatility)
                if proxy_profile is not None:
                    total_gex = float(summarize_gex_profile(proxy_profile, current_price).get('total_gex', 0.0))
            except Exception:
                total_gex = 0.0

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
                'total_gex': total_gex,
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


@st.cache_data(ttl=900)
def get_implied_volatility_analysis(symbol, current_price, fallback_daily_vol=None):
    """Return an ATM implied-volatility estimate and its one-day expected move."""
    iv = None
    expiration = None
    source = "Near-term ATM options"

    try:
        ticker = yf.Ticker(symbol)
        expirations = ticker.options
        if expirations:
            today = pd.Timestamp.now(tz='UTC').date()
            valid_expirations = [
                expiry for expiry in expirations
                if pd.Timestamp(expiry).date() >= today
            ]
            if valid_expirations:
                expiration = valid_expirations[0]
                chain = ticker.option_chain(expiration)
                atm_values = []
                for option_side in (chain.calls, chain.puts):
                    if option_side is None or option_side.empty:
                        continue
                    valid = option_side[
                        option_side['impliedVolatility'].notna()
                        & (option_side['impliedVolatility'] > 0)
                    ].copy()
                    if not valid.empty:
                        nearest = valid.loc[(valid['strike'] - current_price).abs().idxmin()]
                        atm_values.append(float(nearest['impliedVolatility']))
                if atm_values:
                    iv = float(np.mean(atm_values))
    except Exception:
        pass

    is_implied = iv is not None
    if iv is None and fallback_daily_vol is not None and np.isfinite(fallback_daily_vol):
        # Historical returns are daily, so multiply by sqrt(252) to annualize.
        iv = float(fallback_daily_vol) * np.sqrt(252)
        source = "Historical volatility proxy (options IV unavailable)"

    if iv is None or iv <= 0 or current_price is None or current_price <= 0:
        return None

    daily_move_pct = iv / np.sqrt(252)
    daily_move_amount = float(current_price) * daily_move_pct
    return {
        'annualized_vol': iv,
        'is_implied': is_implied,
        'daily_move_pct': daily_move_pct,
        'daily_move_amount': daily_move_amount,
        'lower_price': float(current_price) - daily_move_amount,
        'upper_price': float(current_price) + daily_move_amount,
        'expiration': expiration,
        'source': source,
    }


def backtest_flow_momentum_z_strategy(
    df,
    symbol,
    reward_risk=1.5,
    atr_risk=1.0,
    trade_start=None,
    trade_end=None,
):
    """Backtest confirmed one-hour Flow Z and Momentum Z expansion signals."""
    if df is None or df.empty or len(df) < 75:
        return None, pd.DataFrame(), pd.DataFrame()

    data = df.copy().sort_index()
    data['is_up'] = data['close'] >= data['open']
    inflow = data['volume'].where(data['is_up'], 0.0)
    outflow = data['volume'].where(~data['is_up'], 0.0)
    rolling_inflow = inflow.rolling(window=20).sum()
    rolling_outflow = outflow.rolling(window=20).sum()
    flow_total = (rolling_inflow + rolling_outflow).replace(0, np.nan)
    data['money_flow_signal'] = (rolling_inflow - rolling_outflow) / flow_total

    def rolling_zscore(series, window=50):
        rolling_std = series.rolling(window).std().replace(0, np.nan)
        return (series - series.rolling(window).mean()) / rolling_std

    data['Momentum Z'] = rolling_zscore(data['momentum'])
    data['Flow Z'] = rolling_zscore(data['money_flow_signal'])
    data['Flow Z Change'] = data['Flow Z'].diff()
    data['Momentum Z Change'] = data['Momentum Z'].diff()
    data['Flow Move Size'] = data['Flow Z Change'].abs()
    data['Large Flow Threshold'] = data['Flow Move Size'].rolling(20).mean().shift(1)
    data['Flow Move Size Change'] = data['Flow Move Size'].diff()

    # Direction must agree across Flow Z and Momentum Z, and the current Flow Z
    # move must be larger than the recent average move.
    large_flow_move = data['Flow Move Size'] > data['Large Flow Threshold']
    data['Long Signal'] = (
        large_flow_move
        & (data['Flow Z Change'] > 0)
        & (data['Momentum Z Change'] > 0)
    )
    data['Short Signal'] = (
        large_flow_move
        & (data['Flow Z Change'] < 0)
        & (data['Momentum Z Change'] < 0)
    )

    # Extra candles may be fetched before the requested period solely to warm up
    # rolling indicators. Entries are restricted to the user's selected dates.
    trade_window = pd.Series(True, index=data.index)
    if trade_start is not None:
        trade_window &= data.index >= pd.Timestamp(trade_start)
    if trade_end is not None:
        trade_window &= data.index < pd.Timestamp(trade_end)
    data['Long Signal'] &= trade_window
    data['Short Signal'] &= trade_window

    trades = []
    position = None

    # A signal is known at candle i; entry occurs at candle i+1 open.
    for i in range(len(data) - 1):
        signal_bar = data.iloc[i]
        next_bar = data.iloc[i + 1]
        exited_this_bar = False

        if position is not None:
            exit_price = None
            exit_reason = None
            if position['Direction'] == 'LONG':
                stop_hit = float(next_bar['low']) <= position['Stop Price']
                target_hit = float(next_bar['high']) >= position['Target Price']
                if stop_hit:
                    exit_price, exit_reason = position['Stop Price'], 'Stop Loss'
                elif target_hit:
                    exit_price, exit_reason = position['Target Price'], 'Take Profit'
            else:
                stop_hit = float(next_bar['high']) >= position['Stop Price']
                target_hit = float(next_bar['low']) <= position['Target Price']
                if stop_hit:
                    exit_price, exit_reason = position['Stop Price'], 'Stop Loss'
                elif target_hit:
                    exit_price, exit_reason = position['Target Price'], 'Take Profit'

            if exit_price is not None:
                risk_amount = abs(position['Entry Price'] - position['Stop Price'])
                pnl_amount = (
                    exit_price - position['Entry Price']
                    if position['Direction'] == 'LONG'
                    else position['Entry Price'] - exit_price
                )
                position.update({
                    'Exit Time': data.index[i + 1],
                    'Exit Price': exit_price,
                    'Exit Reason': exit_reason,
                    'R Multiple': pnl_amount / risk_amount if risk_amount > 0 else 0.0,
                    'Return %': (
                        pnl_amount / position['Entry Price'] * 100
                        if position['Entry Price'] > 0 else 0.0
                    ),
                })
                trades.append(position)
                position = None
                exited_this_bar = True

        if (
            position is None
            and not exited_this_bar
            and (bool(signal_bar['Long Signal']) or bool(signal_bar['Short Signal']))
        ):
            atr = float(signal_bar['atr14']) if pd.notna(signal_bar['atr14']) else 0.0
            entry_price = float(next_bar['open'])
            risk_distance = atr * atr_risk
            if entry_price <= 0 or risk_distance <= 0:
                continue

            direction = 'LONG' if bool(signal_bar['Long Signal']) else 'SHORT'
            if direction == 'LONG':
                stop_price = entry_price - risk_distance
                target_price = entry_price + (risk_distance * reward_risk)
            else:
                stop_price = entry_price + risk_distance
                target_price = entry_price - (risk_distance * reward_risk)

            position = {
                'Symbol': symbol,
                'Direction': direction,
                'Signal Time': data.index[i],
                'Entry Time': data.index[i + 1],
                'Entry Price': entry_price,
                'Stop Price': stop_price,
                'Target Price': target_price,
                'Flow Z': float(signal_bar['Flow Z']),
                'Flow Z Change': float(signal_bar['Flow Z Change']),
                'Flow Move Size': float(signal_bar['Flow Move Size']),
                'Flow Move Size Change': float(signal_bar['Flow Move Size Change']),
                'Momentum Z': float(signal_bar['Momentum Z']),
                'Momentum Z Change': float(signal_bar['Momentum Z Change']),
            }

            # The entry is at the next candle's open, so its high/low can hit the
            # stop or target during that same candle.
            if direction == 'LONG':
                entry_stop_hit = float(next_bar['low']) <= stop_price
                entry_target_hit = float(next_bar['high']) >= target_price
            else:
                entry_stop_hit = float(next_bar['high']) >= stop_price
                entry_target_hit = float(next_bar['low']) <= target_price

            if entry_stop_hit or entry_target_hit:
                exit_price = stop_price if entry_stop_hit else target_price
                exit_reason = 'Stop Loss' if entry_stop_hit else 'Take Profit'
                pnl_amount = (
                    exit_price - entry_price
                    if direction == 'LONG'
                    else entry_price - exit_price
                )
                position.update({
                    'Exit Time': data.index[i + 1],
                    'Exit Price': exit_price,
                    'Exit Reason': exit_reason,
                    'R Multiple': pnl_amount / risk_distance,
                    'Return %': pnl_amount / entry_price * 100,
                })
                trades.append(position)
                position = None

    if position is not None:
        final_price = float(data['close'].iloc[-1])
        risk_amount = abs(position['Entry Price'] - position['Stop Price'])
        pnl_amount = (
            final_price - position['Entry Price']
            if position['Direction'] == 'LONG'
            else position['Entry Price'] - final_price
        )
        position.update({
            'Exit Time': data.index[-1],
            'Exit Price': final_price,
            'Exit Reason': 'End of Data',
            'R Multiple': pnl_amount / risk_amount if risk_amount > 0 else 0.0,
            'Return %': pnl_amount / position['Entry Price'] * 100,
        })
        trades.append(position)

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        stats = {
            'Symbol': symbol, 'Trades': 0, 'Win Rate %': 0.0,
            'Net R': 0.0, 'Average R': 0.0, 'Profit Factor': 0.0,
        }
    else:
        wins = trades_df['R Multiple'] > 0
        gross_profit = trades_df.loc[trades_df['R Multiple'] > 0, 'R Multiple'].sum()
        gross_loss = abs(trades_df.loc[trades_df['R Multiple'] < 0, 'R Multiple'].sum())
        stats = {
            'Symbol': symbol,
            'Trades': int(len(trades_df)),
            'Win Rate %': float(wins.mean() * 100),
            'Net R': float(trades_df['R Multiple'].sum()),
            'Average R': float(trades_df['R Multiple'].mean()),
            'Profit Factor': float(gross_profit / gross_loss) if gross_loss > 0 else np.inf,
        }

    return stats, trades_df, data


def calculate_zone_battle_scores(df, observation_bars=10, max_zones=30):
    """Score completed bullish and bearish FVG reactions using the Zone Battle model."""
    if df is None or df.empty or len(df) < 75:
        return pd.DataFrame(), pd.DataFrame()

    data = df.copy().sort_index()

    def rolling_zscore(series, window=50):
        rolling_std = series.rolling(window).std().replace(0, np.nan)
        return (series - series.rolling(window).mean()) / rolling_std

    # ATR(15), matching the reaction normalization requested for Rz.
    previous_close = data['close'].shift(1)
    true_range = pd.concat([
        data['high'] - data['low'],
        (data['high'] - previous_close).abs(),
        (data['low'] - previous_close).abs(),
    ], axis=1).max(axis=1)
    data['ATR15'] = true_range.ewm(alpha=1 / 15, adjust=False).mean()

    # Use the same historical Flow Z and Momentum Z construction as the US Indices chart.
    is_up = data['close'] >= data['open']
    inflow = data['volume'].where(is_up, 0.0)
    outflow = data['volume'].where(~is_up, 0.0)
    rolling_inflow = inflow.rolling(20).sum()
    rolling_outflow = outflow.rolling(20).sum()
    flow_total = (rolling_inflow + rolling_outflow).replace(0, np.nan)
    data['Money Flow Signal'] = (rolling_inflow - rolling_outflow) / flow_total
    data['Flow Z'] = rolling_zscore(data['Money Flow Signal'])
    data['Momentum Z'] = rolling_zscore(data['momentum'])

    bullish_fvg = (data['low'] > data['high'].shift(2)) & (data['close'] > data['open'])
    bearish_fvg = (data['high'] < data['low'].shift(2)) & (data['close'] < data['open'])
    zone_candidates = []

    for position in np.flatnonzero((bullish_fvg | bearish_fvg).to_numpy()):
        if position < 2:
            continue
        if bool(bullish_fvg.iloc[position]):
            zone_type = 'Bullish'
            zone_bottom = float(data['high'].iloc[position - 2])
            zone_top = float(data['low'].iloc[position])
        else:
            zone_type = 'Bearish'
            zone_bottom = float(data['high'].iloc[position])
            zone_top = float(data['low'].iloc[position - 2])

        if zone_top <= zone_bottom:
            continue
        zone_candidates.append((position, zone_type, zone_bottom, zone_top))

    results = []
    for formed_at, zone_type, zone_bottom, zone_top in zone_candidates:
        entry_position = None
        for candidate in range(formed_at + 1, len(data)):
            candle = data.iloc[candidate]
            if float(candle['low']) <= zone_top and float(candle['high']) >= zone_bottom:
                entry_position = candidate
                break
        if entry_position is None:
            continue

        reaction_end = entry_position + int(observation_bars) - 1
        if reaction_end >= len(data):
            continue
        reaction = data.iloc[entry_position:reaction_end + 1]
        atr15 = float(data['ATR15'].iloc[entry_position])
        if not np.isfinite(atr15) or atr15 <= 0:
            continue

        entry_price = (zone_bottom + zone_top) / 2
        reaction_close = float(reaction['close'].iloc[-1])
        reaction_score = (reaction_close - entry_price) / atr15
        flow_z = float(reaction['Flow Z'].iloc[-1]) if pd.notna(reaction['Flow Z'].iloc[-1]) else 0.0
        momentum_z = (
            float(reaction['Momentum Z'].iloc[-1])
            if pd.notna(reaction['Momentum Z'].iloc[-1]) else 0.0
        )

        # +1 means price was above the zone, -1 below it, and 0 inside it.
        location = np.where(
            reaction['close'] > zone_top,
            1.0,
            np.where(reaction['close'] < zone_bottom, -1.0, 0.0),
        )
        acceptance_score = float(np.mean(location))
        battle_score = (
            (0.40 * reaction_score)
            + (0.30 * flow_z)
            + (0.20 * momentum_z)
            + (0.10 * acceptance_score)
        )
        winner = (
            'Strong Buyer Victory' if battle_score > 1.0
            else 'Strong Seller Victory' if battle_score < -1.0
            else 'Battle Unresolved'
        )

        results.append({
            'Zone Formed': data.index[formed_at],
            'Zone Entry': data.index[entry_position],
            'Zone Type': zone_type,
            'Zone Bottom': zone_bottom,
            'Zone Top': zone_top,
            'Entry Price': entry_price,
            'Reaction Close': reaction_close,
            'Bars Observed': int(len(reaction)),
            'Reaction Rz': reaction_score,
            'Flow Fz': flow_z,
            'Momentum Mz': momentum_z,
            'Acceptance Az': acceptance_score,
            'Battle Score': battle_score,
            'Result': winner,
        })

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('Zone Entry', ascending=False).head(max_zones)
    return results_df, data


def calculate_liquidity_zones(df, price_bins=24, top_zones=8):
    """Rank price bands by participation, flow, compression, and price acceptance."""
    if df is None or df.empty or len(df) < 50:
        return pd.DataFrame(), pd.DataFrame()

    data = df.copy().sort_index().tail(750)
    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(data.columns):
        return pd.DataFrame(), pd.DataFrame()

    for column in required:
        data[column] = pd.to_numeric(data[column], errors='coerce')
    data = data.dropna(subset=list(required))
    if len(data) < 50 or float(data['high'].max()) <= float(data['low'].min()):
        return pd.DataFrame(), pd.DataFrame()

    previous_close = data['close'].shift(1)
    true_range = pd.concat([
        data['high'] - data['low'],
        (data['high'] - previous_close).abs(),
        (data['low'] - previous_close).abs(),
    ], axis=1).max(axis=1)
    data['ATR14'] = true_range.ewm(alpha=1 / 14, adjust=False).mean()
    data['Typical Price'] = (data['high'] + data['low'] + data['close']) / 3
    cumulative_volume = data['volume'].cumsum().replace(0, np.nan)
    data['VWAP'] = (data['Typical Price'] * data['volume']).cumsum() / cumulative_volume
    data['Signed Volume'] = np.where(
        data['close'] >= data['open'], data['volume'], -data['volume']
    )
    data['ATR Compression'] = -(
        (data['ATR14'] - data['ATR14'].rolling(50, min_periods=20).mean())
        / data['ATR14'].rolling(50, min_periods=20).std().replace(0, np.nan)
    )
    data['VWAP Distance'] = (
        (data['Typical Price'] - data['VWAP']).abs()
        / data['ATR14'].replace(0, np.nan)
    )
    swing_high = (
        (data['high'] >= data['high'].shift(1))
        & (data['high'] >= data['high'].shift(-1))
    )
    swing_low = (
        (data['low'] <= data['low'].shift(1))
        & (data['low'] <= data['low'].shift(-1))
    )
    data['Swing'] = (swing_high | swing_low).astype(int)

    edges = np.linspace(
        float(data['low'].min()), float(data['high'].max()), int(price_bins) + 1
    )
    data['Price Band'] = pd.cut(
        data['Typical Price'], bins=edges, include_lowest=True, labels=False
    )
    grouped = data.dropna(subset=['Price Band']).groupby('Price Band', observed=True)
    zones = grouped.agg(
        Volume=('volume', 'sum'),
        Signed_Volume=('Signed Volume', 'sum'),
        ATR_Compression=('ATR Compression', 'mean'),
        Swing_Density=('Swing', 'sum'),
        Time_at_Price=('close', 'size'),
        VWAP_Distance=('VWAP Distance', 'mean'),
    ).reset_index()
    if zones.empty:
        return pd.DataFrame(), data

    zones['Price Band'] = zones['Price Band'].astype(int)
    zones['Zone Bottom'] = zones['Price Band'].map(lambda value: edges[value])
    zones['Zone Top'] = zones['Price Band'].map(lambda value: edges[value + 1])
    zones['Zone Midpoint'] = (zones['Zone Bottom'] + zones['Zone Top']) / 2
    zones['Order Flow Imbalance'] = (
        zones['Signed_Volume'] / zones['Volume'].replace(0, np.nan)
    ).fillna(0.0)

    def cross_section_zscore(series):
        clean = pd.to_numeric(series, errors='coerce').fillna(0.0)
        standard_deviation = clean.std(ddof=0)
        if not np.isfinite(standard_deviation) or standard_deviation == 0:
            return pd.Series(0.0, index=clean.index)
        return (clean - clean.mean()) / standard_deviation

    zones['Volume Z-Score'] = cross_section_zscore(np.log1p(zones['Volume']))
    zones['Flow Strength Z'] = cross_section_zscore(zones['Order Flow Imbalance'].abs())
    zones['ATR Compression Z'] = cross_section_zscore(zones['ATR_Compression'])
    zones['Swing Density Z'] = cross_section_zscore(zones['Swing_Density'])
    zones['Time at Price Z'] = cross_section_zscore(zones['Time_at_Price'])
    zones['VWAP Proximity Z'] = cross_section_zscore(-zones['VWAP_Distance'])
    zones['Raw Liquidity Score'] = (
        0.25 * zones['Volume Z-Score']
        + 0.20 * zones['Flow Strength Z']
        + 0.15 * zones['ATR Compression Z']
        + 0.15 * zones['Swing Density Z']
        + 0.15 * zones['Time at Price Z']
        + 0.10 * zones['VWAP Proximity Z']
    )
    score_min = zones['Raw Liquidity Score'].min()
    score_range = zones['Raw Liquidity Score'].max() - score_min
    zones['Liquidity Score'] = (
        100 * (zones['Raw Liquidity Score'] - score_min) / score_range
        if score_range > 0 else 50.0
    )
    latest_price = float(data['close'].iloc[-1])
    zones['Location vs Price'] = np.where(
        zones['Zone Top'] < latest_price,
        'Below price',
        np.where(zones['Zone Bottom'] > latest_price, 'Above price', 'At price'),
    )
    zones['Flow Bias'] = np.where(
        zones['Order Flow Imbalance'] > 0.05,
        'Buying',
        np.where(zones['Order Flow Imbalance'] < -0.05, 'Selling', 'Balanced'),
    )
    zones = zones.sort_values('Liquidity Score', ascending=False).head(int(top_zones))
    return zones.reset_index(drop=True), data


def build_daily_zscore_alert_history(
    df,
    z_score_series,
    symbol,
    timeframe,
    threshold=1.0,
    momentum_z_series=None,
):
    """Build first daily Z alerts, optionally requiring aligned Momentum Z confirmation."""
    if df is None or df.empty:
        return []

    scores = pd.to_numeric(z_score_series, errors='coerce')
    prices = pd.to_numeric(df['close'], errors='coerce')
    history = pd.DataFrame({'Z-Score': scores, 'Price': prices}).dropna()
    if momentum_z_series is not None:
        history['Momentum Z'] = pd.to_numeric(
            momentum_z_series, errors='coerce'
        )
        history = history.dropna(subset=['Momentum Z'])
    if history.empty:
        return []

    timestamps = pd.DatetimeIndex(history.index)
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize('UTC')
    ghana_times = timestamps.tz_convert('Africa/Accra')
    history['Ghana Time'] = ghana_times
    history['Ghana Date'] = ghana_times.date
    positive_condition = history['Z-Score'] >= float(threshold)
    negative_condition = history['Z-Score'] <= -float(threshold)
    if momentum_z_series is not None:
        positive_condition &= history['Momentum Z'] >= float(threshold)
        negative_condition &= history['Momentum Z'] <= -float(threshold)
    history['Alert'] = np.where(
        positive_condition,
        'POSITIVE Z-SCORE',
        np.where(negative_condition, 'NEGATIVE Z-SCORE', None),
    )
    qualifying = history.dropna(subset=['Alert']).copy()
    if qualifying.empty:
        return []

    # Keep the first threshold event of each direction for every Ghana day.
    qualifying = qualifying.sort_values('Ghana Time').drop_duplicates(
        subset=['Ghana Date', 'Alert'], keep='first'
    )
    rows = []
    for _, event in qualifying.iterrows():
        row = {
            'Ghana Date': event['Ghana Date'],
            'Ghana Time': event['Ghana Time'],
            'Symbol': symbol,
            'Alert': event['Alert'],
            'Price': float(event['Price']),
            'Z-Score': float(event['Z-Score']),
            'Timeframe': timeframe,
        }
        if momentum_z_series is not None:
            row['Momentum Z'] = float(event['Momentum Z'])
        rows.append(row)
    return rows


def backtest_nas100_from_vix_daily_alerts(
    nas100_df,
    vix_df,
    nas100_alert_df=None,
    alert_timeframe='1h',
    nas100_alert_timeframe='1h',
    entry_timeframe='5m',
    z_threshold=1.0,
    atr_risk=1.0,
    start_date=None,
    end_date=None,
    strategy_logic='logic_1',
    stored_daily_alerts=None,
    component_drop_threshold=0.5,
):
    """Trade NQ futures in the opposite direction of daily VIX Z-score alerts."""
    using_stored_alerts = stored_daily_alerts is not None and len(stored_daily_alerts) > 0
    if nas100_df is None or nas100_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    if not using_stored_alerts and (vix_df is None or vix_df.empty):
        return {}, pd.DataFrame(), pd.DataFrame()

    nas = nas100_df.copy().sort_index()
    vix = vix_df.copy().sort_index() if vix_df is not None else pd.DataFrame()
    nas_alert_source = (
        nas100_alert_df.copy().sort_index()
        if nas100_alert_df is not None and not nas100_alert_df.empty
        else nas.copy()
    )
    timeframe_durations = {
        '5m': pd.Timedelta(minutes=5),
        '15m': pd.Timedelta(minutes=15),
        '1h': pd.Timedelta(hours=1),
        '4h': pd.Timedelta(hours=4),
    }
    alert_duration = timeframe_durations.get(
        alert_timeframe, pd.Timedelta(hours=1)
    )
    nas_alert_duration = timeframe_durations.get(
        nas100_alert_timeframe, pd.Timedelta(hours=1)
    )
    entry_duration = timeframe_durations.get(
        entry_timeframe, pd.Timedelta(minutes=5)
    )
    previous_close = nas['close'].shift(1)
    true_range = pd.concat([
        nas['high'] - nas['low'],
        (nas['high'] - previous_close).abs(),
        (nas['low'] - previous_close).abs(),
    ], axis=1).max(axis=1)
    nas['ATR14'] = true_range.rolling(14).mean()

    if using_stored_alerts:
        stored_alerts_df = pd.DataFrame(stored_daily_alerts).copy()
        stored_alerts_df['Ghana Time'] = pd.to_datetime(
            stored_alerts_df['Ghana Time'], utc=True
        )
        anchor_symbol = 'NQ=F' if strategy_logic == 'logic_3' else '^VIX'
        signals = stored_alerts_df.loc[
            (stored_alerts_df['Symbol'] == anchor_symbol)
            & (stored_alerts_df['Timeframe'] == alert_timeframe)
        ].sort_values('Ghana Time')
    else:
        vix_scores = pd.to_numeric(vix['z_score'], errors='coerce')
        vix_momentum = pd.to_numeric(vix['momentum'], errors='coerce')
        momentum_mean = vix_momentum.rolling(50).mean()
        momentum_std = vix_momentum.rolling(50).std().replace(0, np.nan)
        vix_momentum_z = (vix_momentum - momentum_mean) / momentum_std
        vix_at_close = vix.copy()
        vix_at_close.index = pd.DatetimeIndex(vix_at_close.index) + alert_duration
        vix_scores.index = vix_at_close.index
        vix_momentum_z.index = vix_at_close.index
        alert_rows = build_daily_zscore_alert_history(
            vix_at_close,
            vix_scores,
            '^VIX',
            alert_timeframe,
            threshold=float(z_threshold),
            momentum_z_series=vix_momentum_z,
        )
        if not alert_rows:
            return {}, pd.DataFrame(), pd.DataFrame()
        signals = pd.DataFrame(alert_rows).sort_values('Ghana Time')
    if start_date is not None:
        signals = signals.loc[
            pd.to_datetime(signals['Ghana Date']).dt.date >= start_date
        ]
    if end_date is not None:
        signals = signals.loc[
            pd.to_datetime(signals['Ghana Date']).dt.date <= end_date
        ]
    if signals.empty:
        return {}, pd.DataFrame(), signals

    signals = signals.copy()
    signals['Execution Time'] = signals['Ghana Time']
    if strategy_logic == 'logic_2':
        if using_stored_alerts:
            nas_alerts = stored_alerts_df.loc[
                (stored_alerts_df['Symbol'] == 'NQ=F')
                & (stored_alerts_df['Timeframe'] == nas100_alert_timeframe)
            ].copy()
            nas_alerts = nas_alerts.rename(columns={
                'Ghana Time': 'NAS100 Alert Time',
                'Alert': 'NAS100 Alert',
                'Z-Score': 'NAS100 Z-Score',
                'Momentum Z': 'NAS100 Momentum Z',
            }).sort_values('NAS100 Alert Time')
        else:
            nas_z = pd.to_numeric(nas_alert_source['z_score'], errors='coerce')
            nas_momentum = pd.to_numeric(
                nas_alert_source['momentum'], errors='coerce'
            )
            nas_momentum_z = (
                (nas_momentum - nas_momentum.rolling(50).mean())
                / nas_momentum.rolling(50).std().replace(0, np.nan)
            )
            positive_aligned = (
                (nas_z >= float(z_threshold))
                & (nas_momentum_z >= float(z_threshold))
            )
            negative_aligned = (
                (nas_z <= -float(z_threshold))
                & (nas_momentum_z <= -float(z_threshold))
            )
            positive_trigger = positive_aligned & ~positive_aligned.shift(1).fillna(False)
            negative_trigger = negative_aligned & ~negative_aligned.shift(1).fillna(False)
            trigger_mask = positive_trigger | negative_trigger

            nas_alert_times = pd.DatetimeIndex(nas_alert_source.index)
            if nas_alert_times.tz is None:
                nas_alert_times = nas_alert_times.tz_localize('UTC')
            else:
                nas_alert_times = nas_alert_times.tz_convert('UTC')
            nas_alert_times = nas_alert_times + nas_alert_duration
            nas_alerts = pd.DataFrame({
                'NAS100 Alert Time': nas_alert_times[trigger_mask.to_numpy()],
                'NAS100 Alert': np.where(
                    positive_trigger[trigger_mask],
                    'POSITIVE Z-SCORE',
                    'NEGATIVE Z-SCORE',
                ),
                'NAS100 Z-Score': nas_z[trigger_mask].to_numpy(),
                'NAS100 Momentum Z': nas_momentum_z[trigger_mask].to_numpy(),
            }).sort_values('NAS100 Alert Time')

        confirmed_rows = []
        ordered_vix = signals.sort_values('Ghana Time').reset_index(drop=True)
        for signal_position, vix_signal in ordered_vix.iterrows():
            vix_time = pd.Timestamp(vix_signal['Ghana Time'])
            next_vix_time = (
                pd.Timestamp(ordered_vix.iloc[signal_position + 1]['Ghana Time'])
                if signal_position + 1 < len(ordered_vix)
                else None
            )
            candidates = nas_alerts.loc[
                (nas_alerts['NAS100 Alert Time'] > vix_time)
                & (nas_alerts['NAS100 Alert'] == vix_signal['Alert'])
            ]
            if next_vix_time is not None:
                candidates = candidates.loc[
                    candidates['NAS100 Alert Time'] < next_vix_time
                ]
            if candidates.empty:
                continue
            confirmation = candidates.iloc[0]
            confirmed = vix_signal.to_dict()
            confirmed.update(confirmation.to_dict())
            confirmed['Execution Time'] = confirmation['NAS100 Alert Time']
            confirmed_rows.append(confirmed)

        signals = pd.DataFrame(confirmed_rows)
        if signals.empty:
            return {}, pd.DataFrame(), signals
    elif strategy_logic == 'logic_3':
        component_data = nas_alert_source.copy().sort_index()

        def component_zscore(series, window=50):
            numeric = pd.to_numeric(series, errors='coerce')
            return (
                (numeric - numeric.rolling(window).mean())
                / numeric.rolling(window).std().replace(0, np.nan)
            )

        component_data['Momentum Z'] = component_zscore(
            component_data['momentum']
        )
        is_up_candle = component_data['close'] >= component_data['open']
        component_inflow = component_data['volume'].where(is_up_candle, 0.0)
        component_outflow = component_data['volume'].where(~is_up_candle, 0.0)
        rolling_inflow = component_inflow.rolling(20).sum()
        rolling_outflow = component_outflow.rolling(20).sum()
        component_flow = (
            (rolling_inflow - rolling_outflow)
            / (rolling_inflow + rolling_outflow).replace(0, np.nan)
        )
        component_data['Flow Z'] = component_zscore(component_flow)
        component_atr = pd.to_numeric(
            component_data.get('atr14'), errors='coerce'
        )
        component_data['Volatility Z'] = component_zscore(component_atr)
        component_data['Trend Z'] = component_zscore(component_data['z_score'])
        component_columns = [
            'Momentum Z', 'Flow Z', 'Volatility Z', 'Trend Z'
        ]
        component_data['Highest Component Z'] = component_data[
            component_columns
        ].max(axis=1)
        component_data['Lowest Component Z'] = component_data[
            component_columns
        ].min(axis=1)
        component_data['Component Dispersion'] = (
            component_data['Highest Component Z']
            - component_data['Lowest Component Z']
        )
        component_data['Highest-Z Drop'] = (
            component_data['Highest Component Z'].shift(1)
            - component_data['Highest Component Z']
        )
        component_data['Lowest-Z Recovery'] = (
            component_data['Lowest Component Z']
            - component_data['Lowest Component Z'].shift(1)
        )
        component_data['Momentum Z Change'] = component_data['Momentum Z'].diff()
        component_data['Flow Z Change'] = component_data['Flow Z'].diff()
        component_data['Dispersion Change'] = component_data[
            'Component Dispersion'
        ].diff()

        component_times = pd.DatetimeIndex(component_data.index)
        if component_times.tz is None:
            component_times = component_times.tz_localize('UTC')
        else:
            component_times = component_times.tz_convert('UTC')
        component_data['Component Time'] = component_times + nas_alert_duration

        confirmed_rows = []
        ordered_alerts = signals.sort_values('Ghana Time').reset_index(drop=True)
        for alert_position, historical_alert in ordered_alerts.iterrows():
            alert_time = pd.Timestamp(historical_alert['Ghana Time'])
            next_alert_time = (
                pd.Timestamp(ordered_alerts.iloc[alert_position + 1]['Ghana Time'])
                if alert_position + 1 < len(ordered_alerts)
                else None
            )
            candidates = component_data.loc[
                component_data['Component Time'] > alert_time
            ].copy()
            if next_alert_time is not None:
                candidates = candidates.loc[
                    candidates['Component Time'] < next_alert_time
                ]

            if historical_alert['Alert'] == 'POSITIVE Z-SCORE':
                confirmation_mask = (
                    (candidates['Highest Component Z'] >= float(z_threshold))
                    & (candidates['Highest-Z Drop'] >= float(component_drop_threshold))
                    & (candidates['Momentum Z Change'] < 0)
                    & (candidates['Flow Z Change'] < 0)
                    & (candidates['Dispersion Change'] < 0)
                )
                trade_direction = 'SHORT'
            else:
                confirmation_mask = (
                    (candidates['Lowest Component Z'] <= -float(z_threshold))
                    & (candidates['Lowest-Z Recovery'] >= float(component_drop_threshold))
                    & (candidates['Momentum Z Change'] > 0)
                    & (candidates['Flow Z Change'] > 0)
                    & (candidates['Dispersion Change'] < 0)
                )
                trade_direction = 'LONG'

            confirmations = candidates.loc[confirmation_mask]
            if confirmations.empty:
                continue
            confirmation = confirmations.iloc[0]
            confirmed = historical_alert.to_dict()
            confirmed.update({
                'Execution Time': confirmation['Component Time'],
                'Trade Direction': trade_direction,
                'Highest Component Z': float(confirmation['Highest Component Z']),
                'Lowest Component Z': float(confirmation['Lowest Component Z']),
                'Highest-Z Drop': float(confirmation['Highest-Z Drop']),
                'Lowest-Z Recovery': float(confirmation['Lowest-Z Recovery']),
                'Momentum Z Change': float(confirmation['Momentum Z Change']),
                'Flow Z Change': float(confirmation['Flow Z Change']),
                'Component Dispersion': float(confirmation['Component Dispersion']),
                'Dispersion Change': float(confirmation['Dispersion Change']),
            })
            confirmed_rows.append(confirmed)

        signals = pd.DataFrame(confirmed_rows)
        if signals.empty:
            return {}, pd.DataFrame(), signals

    nas_index = pd.DatetimeIndex(nas.index)
    signal_index = pd.DatetimeIndex(signals['Execution Time'])
    if nas_index.tz is None:
        nas_comparison_index = nas_index.tz_localize('UTC')
    else:
        nas_comparison_index = nas_index.tz_convert('UTC')
    # Compare candle close times so the selected hourly NAS100 price is the
    # close available exactly when the completed hourly VIX alert triggers.
    nas_comparison_index = nas_comparison_index + entry_duration
    signal_index = signal_index.tz_convert('UTC')

    trades = []
    next_available_position = 0
    for signal_number, (_, signal) in enumerate(signals.iterrows()):
        signal_time = signal_index[signal_number]
        # The VIX alert is confirmed at its candle close. Enter using the NAS100
        # close at that same timestamp (or the latest NAS100 close available at it).
        entry_position = int(
            nas_comparison_index.searchsorted(signal_time, side='right') - 1
        )
        # Ignore alerts that occur while an earlier trade is still open; do not
        # defer a stale signal until that position closes.
        if entry_position < 0 or entry_position < next_available_position:
            continue
        if entry_position >= len(nas):
            continue

        entry_bar = nas.iloc[entry_position]
        entry_price = float(entry_bar['close'])
        atr = float(entry_bar['ATR14'])
        if not np.isfinite(entry_price) or not np.isfinite(atr) or atr <= 0:
            continue

        if strategy_logic == 'logic_3':
            direction = signal['Trade Direction']
        elif strategy_logic == 'logic_2':
            direction = 'LONG' if signal['Alert'] == 'POSITIVE Z-SCORE' else 'SHORT'
        else:
            direction = 'SHORT' if signal['Alert'] == 'POSITIVE Z-SCORE' else 'LONG'
        risk_distance = atr * float(atr_risk)

        # Build the 1-hour liquidity map only from candles that had closed by
        # the entry time. This prevents future zones leaking into the backtest.
        liquidity_index = pd.DatetimeIndex(nas_alert_source.index)
        if liquidity_index.tz is None:
            liquidity_close_index = liquidity_index.tz_localize('UTC')
        else:
            liquidity_close_index = liquidity_index.tz_convert('UTC')
        liquidity_close_index = liquidity_close_index + nas_alert_duration
        available_liquidity_positions = np.flatnonzero(
            liquidity_close_index <= signal_time
        )
        if len(available_liquidity_positions) == 0:
            continue
        liquidity_history = nas_alert_source.iloc[
            :int(available_liquidity_positions[-1]) + 1
        ]
        liquidity_zones, _ = calculate_liquidity_zones(
            liquidity_history,
            price_bins=24,
            top_zones=24,
        )
        if liquidity_zones.empty:
            continue

        if direction == 'LONG':
            stop_price = entry_price - risk_distance
            target_candidates = liquidity_zones.loc[
                liquidity_zones['Zone Bottom'] > entry_price
            ].sort_values('Zone Bottom')
            if target_candidates.empty:
                continue
            target_zone = target_candidates.iloc[0]
            target_price = float(target_zone['Zone Bottom'])
        else:
            stop_price = entry_price + risk_distance
            target_candidates = liquidity_zones.loc[
                liquidity_zones['Zone Top'] < entry_price
            ].sort_values('Zone Top', ascending=False)
            if target_candidates.empty:
                continue
            target_zone = target_candidates.iloc[0]
            target_price = float(target_zone['Zone Top'])
        planned_reward_risk = abs(target_price - entry_price) / risk_distance

        exit_position = len(nas) - 1
        exit_price = float(nas['close'].iloc[-1])
        exit_reason = 'End of Data'
        # Entry occurs at the signal candle close, so only subsequent candles
        # can touch the stop or target.
        for position in range(entry_position + 1, len(nas)):
            candle = nas.iloc[position]
            if direction == 'LONG':
                stop_hit = float(candle['low']) <= stop_price
                target_hit = float(candle['high']) >= target_price
            else:
                stop_hit = float(candle['high']) >= stop_price
                target_hit = float(candle['low']) <= target_price

            # Conservative rule: count the stop first if both are hit in one candle.
            if stop_hit or target_hit:
                exit_position = position
                exit_price = stop_price if stop_hit else target_price
                exit_reason = 'Stop Loss' if stop_hit else 'Take Profit'
                break

        pnl_points = (
            exit_price - entry_price
            if direction == 'LONG'
            else entry_price - exit_price
        )
        r_multiple = pnl_points / risk_distance
        trades.append({
            'Ghana Alert Date': signal['Ghana Date'],
            'Alert Source': signal.get('Symbol', '^VIX'),
            'Historical Alert Time': signal['Ghana Time'],
            'Historical Alert Z-Score': float(signal['Z-Score']),
            'Historical Alert Momentum Z': float(signal['Momentum Z']),
            'VIX Alert Time': signal['Ghana Time'],
            'VIX Z-Score': float(signal['Z-Score']),
            'VIX Momentum Z': float(signal['Momentum Z']),
            'NAS100 Alert Time': signal.get('NAS100 Alert Time', pd.NaT),
            'NAS100 Z-Score': float(signal.get('NAS100 Z-Score', np.nan)),
            'NAS100 Momentum Z': float(signal.get('NAS100 Momentum Z', np.nan)),
            'Highest Component Z': float(signal.get('Highest Component Z', np.nan)),
            'Lowest Component Z': float(signal.get('Lowest Component Z', np.nan)),
            'Highest-Z Drop': float(signal.get('Highest-Z Drop', np.nan)),
            'Lowest-Z Recovery': float(signal.get('Lowest-Z Recovery', np.nan)),
            'Momentum Z Change': float(signal.get('Momentum Z Change', np.nan)),
            'Flow Z Change': float(signal.get('Flow Z Change', np.nan)),
            'Component Dispersion': float(signal.get('Component Dispersion', np.nan)),
            'Dispersion Change': float(signal.get('Dispersion Change', np.nan)),
            'Direction': direction,
            'Entry Time': nas_comparison_index[entry_position],
            'Entry Price': entry_price,
            'ATR14': atr,
            'Stop Price': stop_price,
            'Target Price': target_price,
            'Target Zone Bottom': float(target_zone['Zone Bottom']),
            'Target Zone Top': float(target_zone['Zone Top']),
            'Target Liquidity Score': float(target_zone['Liquidity Score']),
            'Planned Reward/Risk': planned_reward_risk,
            'Exit Time': nas_comparison_index[exit_position],
            'Exit Price': exit_price,
            'Exit Reason': exit_reason,
            'R Multiple': r_multiple,
            'PnL Points': pnl_points,
        })
        next_available_position = exit_position + 1

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return {}, trades_df, signals

    winning_r = trades_df.loc[trades_df['R Multiple'] > 0, 'R Multiple'].sum()
    losing_r = abs(trades_df.loc[trades_df['R Multiple'] < 0, 'R Multiple'].sum())
    summary = {
        'Trades': int(len(trades_df)),
        'Win Rate %': float((trades_df['R Multiple'] > 0).mean() * 100),
        'Net R': float(trades_df['R Multiple'].sum()),
        'Average R': float(trades_df['R Multiple'].mean()),
        'Profit Factor': float(winning_r / losing_r) if losing_r > 0 else np.inf,
        'Net Points': float(trades_df['PnL Points'].sum()),
    }
    trades_df['Cumulative R'] = trades_df['R Multiple'].cumsum()
    return summary, trades_df, signals


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

@st.cache_data(ttl=300)
def build_derivative_factor_history(symbol, timeframe='1h', flow_timeframe='1h', lookback=200):
    """Build historical factor series for a selected derivative asset."""
    try:
        exchange = ccxt.bybit({'enableRateLimit': True, 'options': {'defaultType': 'swap'}})
        candidates = []
        raw_symbol = str(symbol).strip()
        if raw_symbol:
            candidates.append(raw_symbol)
            if raw_symbol.endswith('/USDT'):
                candidates.append(raw_symbol.replace('/USDT', ''))
                candidates.append(raw_symbol.replace('/USDT', ':USDT'))
                candidates.append(raw_symbol.replace('/USDT', 'USDT'))
            elif raw_symbol.endswith('USDT'):
                candidates.append(f'{raw_symbol}/USDT')
            else:
                candidates.append(f'{raw_symbol}/USDT')
                candidates.append(f'{raw_symbol}USDT')

        # Try a few common symbol forms and fall back to Binance if Bybit rejects the symbol.
        ohlcv = None
        for candidate in candidates:
            try:
                ohlcv = exchange.fetch_ohlcv(candidate, timeframe=timeframe, limit=lookback)
                if ohlcv:
                    market_symbol = candidate
                    break
            except Exception:
                continue

        if not ohlcv:
            try:
                binance_exchange = ccxt.binance({'enableRateLimit': True})
                for candidate in candidates:
                    try:
                        ohlcv = binance_exchange.fetch_ohlcv(candidate, timeframe=timeframe, limit=lookback)
                        if ohlcv:
                            market_symbol = candidate
                            break
                    except Exception:
                        continue
            except Exception:
                ohlcv = None

        if not ohlcv:
            return None

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('date').sort_index()
        df['close'] = pd.to_numeric(df['close'], errors='coerce')
        df['open'] = pd.to_numeric(df['open'], errors='coerce')
        df['high'] = pd.to_numeric(df['high'], errors='coerce')
        df['low'] = pd.to_numeric(df['low'], errors='coerce')
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        df = df.dropna(subset=['close', 'open', 'high', 'low', 'volume'])

        df['returns'] = df['close'].pct_change()
        df['momentum_z'] = (df['returns'] - df['returns'].rolling(20).mean()) / (df['returns'].rolling(20).std() + 1e-6)

        df['up_volume'] = np.where(df['close'] >= df['open'], df['volume'], 0)
        df['down_volume'] = np.where(df['close'] < df['open'], df['volume'], 0)
        df['flow_ratio'] = (df['up_volume'] - df['down_volume']) / (df['up_volume'] + df['down_volume'] + 1e-6)
        df['flow_z'] = (df['flow_ratio'] - df['flow_ratio'].rolling(20).mean()) / (df['flow_ratio'].rolling(20).std() + 1e-6)

        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        atr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
        df['volatility_z'] = (atr - atr.rolling(20).mean()) / (atr.rolling(20).std() + 1e-6)

        ma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['trend_z'] = (df['close'] - ma20) / (std20 + 1e-6)

        df[['momentum_z', 'flow_z', 'volatility_z', 'trend_z']] = df[['momentum_z', 'flow_z', 'volatility_z', 'trend_z']].fillna(0)
        return df[['momentum_z', 'flow_z', 'volatility_z', 'trend_z', 'close', 'volume']]
    except Exception:
        return None


DEXSCREENER_HISTORY_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'dexscreener_meme_history.csv'
)


def _safe_number(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _cross_section_z(series):
    values = pd.to_numeric(series, errors='coerce')
    std = values.std(ddof=1)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / std


@st.cache_data(ttl=60, show_spinner=False)
def fetch_dexscreener_pairs(discovery_mode='Top boosted', query='meme', limit=20):
    """Fetch liquid token pairs from Dexscreener's documented public API."""
    headers = {'Accept': 'application/json', 'User-Agent': 'Hashem-Quant/1.0'}
    pairs = []
    if discovery_mode == 'Search':
        response = requests.get(
            'https://api.dexscreener.com/latest/dex/search',
            params={'q': query.strip() or 'meme'}, headers=headers, timeout=15,
        )
        response.raise_for_status()
        pairs = response.json().get('pairs') or []
    else:
        response = requests.get(
            'https://api.dexscreener.com/token-boosts/top/v1',
            headers=headers, timeout=15,
        )
        response.raise_for_status()
        boosted = response.json() or []
        seen_tokens = set()
        for token in boosted:
            chain = token.get('chainId')
            address = token.get('tokenAddress')
            token_key = (chain, address)
            if not chain or not address or token_key in seen_tokens:
                continue
            seen_tokens.add(token_key)
            pair_response = requests.get(
                f'https://api.dexscreener.com/token-pairs/v1/{chain}/{address}',
                headers=headers, timeout=15,
            )
            if not pair_response.ok:
                continue
            token_pairs = pair_response.json() or []
            if token_pairs:
                pairs.append(max(
                    token_pairs,
                    key=lambda item: _safe_number((item.get('liquidity') or {}).get('usd'), 0),
                ))
            if len(pairs) >= int(limit):
                break

    unique_pairs = {}
    for pair in pairs:
        key = (pair.get('chainId'), pair.get('pairAddress'))
        if key[0] and key[1]:
            current_liquidity = _safe_number((pair.get('liquidity') or {}).get('usd'), 0)
            old_liquidity = _safe_number(
                (unique_pairs.get(key, {}).get('liquidity') or {}).get('usd'), -1
            )
            if current_liquidity > old_liquidity:
                unique_pairs[key] = pair
    return sorted(
        unique_pairs.values(),
        key=lambda item: _safe_number((item.get('liquidity') or {}).get('usd'), 0),
        reverse=True,
    )[:int(limit)]


def build_dexscreener_overview(pairs, horizon='h1'):
    """Adapt the US-indices overview factors to Dexscreener pair statistics."""
    expected_h24_fraction = {'m5': 1 / 288, 'h1': 1 / 24, 'h6': 1 / 4, 'h24': 1}
    rows = []
    for pair in pairs:
        base = pair.get('baseToken') or {}
        quote = pair.get('quoteToken') or {}
        txns = (pair.get('txns') or {}).get(horizon) or {}
        buys = _safe_number(txns.get('buys'), 0)
        sells = _safe_number(txns.get('sells'), 0)
        transaction_count = buys + sells
        flow = (buys - sells) / transaction_count if transaction_count else 0.0
        volume = _safe_number((pair.get('volume') or {}).get(horizon), 0)
        h24_volume = _safe_number((pair.get('volume') or {}).get('h24'), 0)
        expected_volume = h24_volume * expected_h24_fraction[horizon]
        volume_ratio = volume / expected_volume if expected_volume > 0 else 0.0
        rows.append({
            'Asset': f"{base.get('symbol', '?')}/{quote.get('symbol', '?')}",
            'Name': base.get('name') or base.get('symbol') or 'Unknown',
            'Chain': pair.get('chainId') or '',
            'DEX': pair.get('dexId') or '',
            'Pair Address': pair.get('pairAddress') or '',
            'Token Address': base.get('address') or '',
            'Dexscreener URL': pair.get('url') or '',
            'Price USD': _safe_number(pair.get('priceUsd')),
            'Momentum %': _safe_number((pair.get('priceChange') or {}).get(horizon), 0),
            'Buys': buys,
            'Sells': sells,
            'Buy Pressure': flow,
            'Buy RSI': 100 * buys / transaction_count if transaction_count else 50.0,
            'Volume USD': volume,
            'Volume Ratio': volume_ratio,
            'Liquidity USD': _safe_number((pair.get('liquidity') or {}).get('usd'), 0),
            'Market Cap': _safe_number(pair.get('marketCap')),
            'FDV': _safe_number(pair.get('fdv')),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result['Z-Score'] = _cross_section_z(result['Momentum %'])
    result['Momentum Z'] = _cross_section_z(result['Momentum %'])
    result['Flow Z-Score'] = _cross_section_z(result['Buy Pressure'])
    result['Signal Score'] = (
        -result['Z-Score']
        + np.log1p(result['Volume Ratio'].clip(lower=0))
        + result['Flow Z-Score']
    )
    result['Trend'] = np.where(result['Momentum %'] >= 0, 'Bullish', 'Bearish')
    result['Alert'] = np.select(
        [
            (result['Z-Score'] >= 1) & (result['Momentum Z'] >= 1),
            (result['Z-Score'] <= -1) & (result['Momentum Z'] <= -1),
        ],
        ['Positive aligned', 'Negative aligned'],
        default='None',
    )
    return result.sort_values('Signal Score', ascending=False).reset_index(drop=True)


def store_dexscreener_snapshot(overview, horizon):
    """Persist at most one snapshot per pair/horizon/minute for rolling history."""
    if overview is None or overview.empty:
        return
    timestamp = pd.Timestamp.now(tz='UTC').floor('min')
    snapshot = overview[[
        'Asset', 'Name', 'Chain', 'DEX', 'Pair Address', 'Token Address',
        'Price USD', 'Momentum %', 'Buy Pressure', 'Volume USD',
        'Liquidity USD',
    ]].copy()
    snapshot.insert(0, 'Timestamp', timestamp.isoformat())
    snapshot['Horizon'] = horizon
    if os.path.exists(DEXSCREENER_HISTORY_PATH):
        try:
            history = pd.read_csv(DEXSCREENER_HISTORY_PATH)
        except Exception:
            history = pd.DataFrame()
        history = pd.concat([history, snapshot], ignore_index=True)
    else:
        history = snapshot
    history = history.drop_duplicates(
        subset=['Timestamp', 'Chain', 'Pair Address', 'Horizon'], keep='last'
    ).tail(100000)
    history.to_csv(DEXSCREENER_HISTORY_PATH, index=False)


def load_dexscreener_component_history(chain, pair_address, horizon):
    if not os.path.exists(DEXSCREENER_HISTORY_PATH):
        return pd.DataFrame()
    try:
        history = pd.read_csv(DEXSCREENER_HISTORY_PATH)
    except Exception:
        return pd.DataFrame()
    required = {'Timestamp', 'Chain', 'Pair Address', 'Horizon'}
    if not required.issubset(history.columns):
        return pd.DataFrame()
    history = history[
        (history['Chain'] == chain)
        & (history['Pair Address'] == pair_address)
        & (history['Horizon'] == horizon)
    ].copy()
    if history.empty:
        return history
    history['Timestamp'] = pd.to_datetime(history['Timestamp'], utc=True, errors='coerce')
    history = history.dropna(subset=['Timestamp']).sort_values('Timestamp').set_index('Timestamp')
    for column in ['Price USD', 'Momentum %', 'Buy Pressure', 'Volume USD', 'Liquidity USD']:
        history[column] = pd.to_numeric(history[column], errors='coerce')

    def rolling_z(values, window=50):
        mean = values.rolling(window, min_periods=3).mean()
        std = values.rolling(window, min_periods=3).std()
        return (values - mean) / std.replace(0, np.nan)

    returns = history['Price USD'].pct_change()
    price_mid = history['Price USD'].rolling(20, min_periods=3).mean()
    price_std = history['Price USD'].rolling(20, min_periods=3).std()
    price_position = (history['Price USD'] - price_mid) / price_std.replace(0, np.nan)
    realized_volatility = returns.rolling(14, min_periods=3).std()
    history['Momentum Z'] = rolling_z(history['Momentum %'])
    history['Flow Z'] = rolling_z(history['Buy Pressure'])
    history['Volatility Z'] = rolling_z(realized_volatility)
    history['Trend Z'] = rolling_z(price_position)
    return history


def build_dexscreener_flow_event_analysis(overview, horizon):
    """Return the largest historical Flow-Z rise and drop for every scanned pair."""
    event_rows = []
    coverage_rows = []
    if overview is None or overview.empty:
        return pd.DataFrame(), pd.DataFrame()

    for _, asset_row in overview.iterrows():
        asset = asset_row['Asset']
        chain = asset_row['Chain']
        pair_address = asset_row['Pair Address']
        history = load_dexscreener_component_history(chain, pair_address, horizon)
        coverage_rows.append({
            'Asset': asset,
            'Chain': chain,
            'Pair Address': pair_address,
            'Stored Snapshots': len(history),
            'Ready': len(history) >= 4,
        })
        if len(history) < 4:
            continue

        components = history[
            ['Momentum Z', 'Flow Z', 'Volatility Z', 'Trend Z']
        ].copy()
        component_changes = components.diff()
        flow_changes = component_changes['Flow Z'].dropna()
        if flow_changes.empty:
            continue

        price_changes = history['Price USD'].pct_change() * 100
        event_keys = []
        if flow_changes.min() < 0:
            event_keys.append(('Highest Flow Z Drop', flow_changes.idxmin()))
        if flow_changes.max() > 0:
            event_keys.append(('Highest Flow Z Rise', flow_changes.idxmax()))
        for event_name, event_time in event_keys:
            flow_change = component_changes.at[event_time, 'Flow Z']
            momentum_change = component_changes.at[event_time, 'Momentum Z']
            volatility_change = component_changes.at[event_time, 'Volatility Z']
            trend_change = component_changes.at[event_time, 'Trend Z']
            price_change = price_changes.get(event_time, np.nan)
            volatility_z = components.at[event_time, 'Volatility Z']

            demand_score = (
                (60 if pd.notna(flow_change) and flow_change > 0 else 0)
                + (25 if pd.notna(momentum_change) and momentum_change > 0 else 0)
                + (15 if pd.notna(volatility_change) and volatility_change > 0
                   and pd.notna(price_change) and price_change > 0 else 0)
            )
            supply_score = (
                (60 if pd.notna(flow_change) and flow_change < 0 else 0)
                + (25 if pd.notna(momentum_change) and momentum_change < 0 else 0)
                + (15 if pd.notna(volatility_change) and volatility_change > 0
                   and pd.notna(price_change) and price_change < 0 else 0)
            )
            if demand_score >= 60 and demand_score > supply_score:
                verdict = 'Strong demand' if demand_score >= 85 else 'Moderate demand'
            elif supply_score >= 60 and supply_score > demand_score:
                verdict = 'Strong supply' if supply_score >= 85 else 'Moderate supply'
            else:
                verdict = 'Mixed / unconfirmed'

            if pd.isna(volatility_z):
                volatility_regime = 'Unavailable'
            elif volatility_z >= 1:
                volatility_regime = 'Expansion'
            elif volatility_z <= -1:
                volatility_regime = 'Compression'
            else:
                volatility_regime = 'Normal'

            event_rows.append({
                'Asset': asset,
                'Chain': chain,
                'Timeframe': horizon,
                'Event': event_name,
                'Event Time (Ghana)': event_time,
                'Flow Z': components.at[event_time, 'Flow Z'],
                'Flow Z Change': flow_change,
                'Momentum Z Change': momentum_change,
                'Trend Z Change': trend_change,
                'Volatility Z Change': volatility_change,
                'Price Change %': price_change,
                'Volatility Regime': volatility_regime,
                'Demand Score': demand_score,
                'Supply Score': supply_score,
                'Demand/Supply Verdict': verdict,
                'History Confidence': (
                    'Full lookback' if len(history) >= 50
                    else 'Developing' if len(history) >= 10
                    else 'Provisional'
                ),
                'Stored Snapshots': len(history),
                'Pair Address': pair_address,
            })
    return pd.DataFrame(event_rows), pd.DataFrame(coverage_rows)


def build_us_index_flow_event_analysis(symbol, data, timeframe):
    """Calculate the strongest historical Flow-Z rise and drop for one index."""
    if data is None or data.empty or len(data) < 52:
        return []
    history = data.copy().sort_index()
    required = {'open', 'high', 'low', 'close', 'volume', 'momentum', 'atr14', 'z_score'}
    if not required.issubset(history.columns):
        return []

    history['is_up'] = history['close'] >= history['open']
    inflow = pd.to_numeric(history['volume'], errors='coerce').where(history['is_up'], 0)
    outflow = pd.to_numeric(history['volume'], errors='coerce').where(~history['is_up'], 0)
    rolling_inflow = inflow.rolling(20).sum()
    rolling_outflow = outflow.rolling(20).sum()
    flow_total = (rolling_inflow + rolling_outflow).replace(0, np.nan)
    money_flow_signal = (rolling_inflow - rolling_outflow) / flow_total

    def rolling_z(series, window=50):
        numeric = pd.to_numeric(series, errors='coerce')
        mean = numeric.rolling(window).mean()
        std = numeric.rolling(window).std().replace(0, np.nan)
        return (numeric - mean) / std

    components = pd.DataFrame(index=history.index)
    components['Momentum Z'] = rolling_z(history['momentum'])
    components['Flow Z'] = rolling_z(money_flow_signal)
    components['Volatility Z'] = rolling_z(history['atr14'])
    components['Trend Z'] = rolling_z(history['z_score'])
    changes = components.diff()
    flow_changes = changes['Flow Z'].dropna()
    if flow_changes.empty:
        return []

    event_keys = []
    if flow_changes.min() < 0:
        event_keys.append(('Highest Flow Z Drop', flow_changes.idxmin()))
    if flow_changes.max() > 0:
        event_keys.append(('Highest Flow Z Rise', flow_changes.idxmax()))

    results = []
    for event_name, event_time in event_keys:
        position = history.index.get_indexer([event_time])[0]
        if position <= 0:
            continue
        current_bar = history.iloc[position]
        previous_close = float(history['close'].iloc[position - 1])
        price_change = float(current_bar['close']) - previous_close
        flow_change = float(changes.at[event_time, 'Flow Z'])
        momentum_change = float(changes.at[event_time, 'Momentum Z'])
        volatility_change = float(changes.at[event_time, 'Volatility Z'])
        trend_change = float(changes.at[event_time, 'Trend Z'])

        flow_contribution = 0.60 * np.tanh(flow_change)
        momentum_contribution = 0.25 * np.tanh(momentum_change)
        volatility_direction = (
            np.sign(price_change) * np.tanh(abs(volatility_change))
            if volatility_change > 0 else 0.0
        )
        volatility_contribution = 0.15 * volatility_direction
        direction_score = 100 * (
            flow_contribution + momentum_contribution + volatility_contribution
        )
        if direction_score >= 10:
            classification = 'Candidate Demand Zone'
        elif direction_score <= -10:
            classification = 'Candidate Supply Zone'
        else:
            classification = 'Unresolved / Balanced Zone'

        event_timestamp = pd.Timestamp(event_time)
        if event_timestamp.tzinfo is None:
            ghana_time = event_timestamp.tz_localize('UTC')
        else:
            ghana_time = event_timestamp.tz_convert('UTC')
        results.append({
            'Symbol': symbol,
            'Timeframe': timeframe,
            'Event': event_name,
            'Event Time (Ghana)': ghana_time,
            'Flow Z': float(components.at[event_time, 'Flow Z']),
            'Flow Z Change': flow_change,
            'Momentum Z Change': momentum_change,
            'Trend Z Change': trend_change,
            'Volatility Z Change': volatility_change,
            'Price Change': price_change,
            'Zone Bottom': float(current_bar['low']),
            'Zone Top': float(current_bar['high']),
            'Close': float(current_bar['close']),
            'Volatility Regime': (
                'Expansion' if volatility_change > 0
                else 'Compression' if volatility_change < 0
                else 'Unchanged'
            ),
            'Direction Score': direction_score,
            'Demand/Supply Classification': classification,
            'Historical Candles': len(history),
        })
    return results


def main():
    st.title("Quantitative Scalping Dashboard (1h) 📈")

    @st.cache_data(ttl=600)
    def calculate_gamma_exposure_binance(symbol, timeframe='1h'):
        """Fetches crypto options data when available and falls back to a GEX-style proxy profile otherwise."""
        try:
            exchange = ccxt.binance({'enableRateLimit': True})

            raw_symbol = str(symbol).strip()
            if raw_symbol.endswith('/USDT'):
                base_asset = raw_symbol.split('/')[0]
            elif raw_symbol.endswith('USDT'):
                base_asset = raw_symbol.replace('USDT', '')
            else:
                base_asset = raw_symbol

            # Try multiple binance symbol formats for tickers and OHLCV
            symbol_candidates = []
            if raw_symbol.endswith('/USDT'):
                symbol_candidates = [raw_symbol, raw_symbol.replace('/USDT', 'USDT')]
            elif raw_symbol.endswith('USDT'):
                symbol_candidates = [raw_symbol, raw_symbol.replace('USDT', '/USDT')]
            else:
                symbol_candidates = [f'{base_asset}/USDT', f'{base_asset}USDT']

            ticker = None
            for sym in symbol_candidates:
                try:
                    ticker = exchange.fetch_ticker(sym)
                    market_symbol = sym
                    break
                except Exception:
                    continue

            if ticker is None:
                return None, f"Binance does not have market symbol {raw_symbol}"

            current_price = float(ticker['last'])

            recent_volatility = None
            try:
                for sym in symbol_candidates:
                    try:
                        ohlcv = exchange.fetch_ohlcv(sym, timeframe, limit=50)
                        if ohlcv:
                            market_symbol = sym
                            df_candles = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            closes = pd.to_numeric(df_candles['close'], errors='coerce').dropna()
                            if len(closes) > 1:
                                returns = closes.pct_change().dropna()
                                recent_volatility = float(returns.std() * np.sqrt(24))
                            break
                    except Exception:
                        continue
            except Exception:
                recent_volatility = None

            try:
                markets = exchange.load_markets()
                options_for_asset = {
                    sym: market for sym, market in markets.items()
                    if market.get('option') and market.get('base') == base_asset
                }
            except Exception:
                options_for_asset = {}

            options_data = []
            if options_for_asset:
                try:
                    option_tickers = exchange.fetch_tickers(list(options_for_asset.keys()))
                    for sym, data in option_tickers.items():
                        info = data.get('info', {})
                        if all(k in info for k in ['g', 'o', 's', 'c']):
                            options_data.append({
                                'strike': float(info['s']),
                                'gamma': float(info['g']),
                                'openInterest': float(info['o']),
                                'type': 'call' if info['c'] else 'put'
                            })
                except Exception:
                    options_data = []

            if options_data:
                df_valid = pd.DataFrame(options_data)
                df_valid['gamma_exposure'] = df_valid['gamma'] * df_valid['openInterest'] * np.where(df_valid['type'] == 'put', -1, 1)
                gex_profile = df_valid.groupby('strike')['gamma_exposure'].sum()
                summary = summarize_gex_profile(gex_profile, current_price)
                summary['proxy_used'] = False
                return summary, None

            proxy_profile = build_binance_gex_proxy(current_price, recent_volatility=recent_volatility)
            if proxy_profile is None:
                return {"current_price": current_price}, "Could not retrieve a market price for the selected crypto asset from Binance."

            summary = summarize_gex_profile(proxy_profile, current_price)
            summary['proxy_used'] = True
            summary['proxy_reason'] = 'Binance does not expose crypto options chains through the available API endpoint, so a volatility-based proxy profile is being shown.'
            return summary, None
        except Exception as e:
            return None, f"An unexpected error occurred with the Binance API: {e}"
    
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
    polygon_api_key = st.sidebar.text_input("Polygon.io API Key", type="password")

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
        "🔥 Derivatives Trend Scan",
        "🛠️ Backtest Engine",
        "🏛️ US Indices",
        "🌌 Volatility Quantum Analysis",
        "📈 Volatility Dashboard",
        "🆕 Derivative Crypto",
        "🧪 Index & Gold Z Backtest",
        "📉 NAS100 VIX Alert Backtest",
        "Meme Coins (Dexscreener)",
    ])

    if False:  # Removed: Market Overview
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

    if False:  # Removed: Top Crypto Ranking
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
                
                styler = styler.map(color_metrics, subset=['momentum', 'funding_rate'])
                    
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

    with tab1:
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
            columns_to_display = [c for c in df_deriv.columns if c not in {'open_interest', 'vol_4h', 'liquidity_ratio', 'entry_score', 'atr14'}]
            df_display = df_deriv[columns_to_display].copy()

            styler = df_display.style.format({
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
                "rsi_15m": "{:.2f}",
                "entry_signal": "{}",
                "total_gex": "{:.0f}",
                "24h_volume": format_large_number,
                "inflow": format_large_number,
                "outflow": format_large_number,
                "net_flow": format_large_number
            })

            # This check is now redundant as we are enforcing .map, but left for context
            # on how it was handled previously.

            if hasattr(styler, 'map'):
                styler = styler.map(color_metrics, subset=['momentum', 'z_score', 'money_flow_signal', 'funding_signal', 'tps', 'net_flow', 'rsi_15m', 'quantum_verdict', 'qs_rel_btc', 'pump_score'])
                styler = styler.apply(lambda x: [color_qs(v) for v in x], subset=['qs_score'])
            else:
                styler = styler.applymap(color_metrics, subset=['momentum', 'z_score', 'money_flow_signal', 'funding_signal', 'tps', 'net_flow', 'rsi_15m', 'quantum_verdict', 'qs_rel_btc', 'pump_score'])
                styler = styler.apply(lambda x: [color_qs(v) for v in x], subset=['qs_score'])
            
            def color_qs(val):
                if val > 2: return 'background-color: #0a8a0a; color: white' # Very Strong
                if val > 1: return 'background-color: #90ee90' # Strong
                if val < -2: return 'background-color: #a52a2a; color: white' # Very Weak
                if val < -1: return 'background-color: #f08080' # Weak
                return ''
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

            # --- Crypto GEX quick-check for the selected scanned asset ---
            gex_col1, gex_col2 = st.columns([1, 1])
            with gex_col1:
                show_gex = st.button("Show Crypto GEX for Selected Asset", key='deriv_show_gex')
            with gex_col2:
                gex_timeframe_deriv = st.selectbox("GEX Timeframe", options=['5m','15m','1h','4h'], index=2, key='deriv_gex_tf')

            if show_gex:
                with st.spinner(f"Calculating GEX for {quantum_symbol_deriv}..."):
                    try:
                        gex_data, gex_err = calculate_gamma_exposure_binance(quantum_symbol_deriv, timeframe=gex_timeframe_deriv)
                        if gex_err and gex_data and 'current_price' in gex_data:
                            st.metric("Current Price", f"${gex_data.get('current_price', 0):,.4f}")
                            st.error(f"Could not calculate GEX: {gex_err}")
                        elif gex_data and 'gex_profile' in gex_data:
                            if gex_data.get('proxy_used'):
                                st.info(gex_data.get('proxy_reason', 'Using proxy GEX profile (no on-chain options data).'))

                            st.metric("Total GEX (Notional)", f"{gex_data['total_gex']:,.0f}")
                            st.metric("Zero Gamma Level", f"${gex_data['zero_gamma_level']:.4f}")
                            st.metric("Current Price", f"${gex_data['current_price']:.4f}")

                            gex_profile_df = gex_data['gex_profile'].reset_index()
                            gex_profile_df.columns = ['strike', 'gamma_exposure']

                            fig = go.Figure()
                            fig.add_trace(go.Bar(x=gex_profile_df['strike'], y=gex_profile_df['gamma_exposure'], name='Gamma Exposure'))
                            fig.add_vline(x=gex_data['zero_gamma_level'], line_width=2, line_dash="dash", line_color="yellow", annotation_text="Zero Gamma", annotation_position="top right")
                            fig.add_vline(x=gex_data['current_price'], line_width=2, line_dash="solid", line_color="cyan", annotation_text="Current Price", annotation_position="top left")
                            fig.update_layout(title=f'Gamma Exposure Profile for {quantum_symbol_deriv}', xaxis_title='Strike Price', yaxis_title='Gamma Exposure (Notional)', template='plotly_dark')
                            st.plotly_chart(fig, use_container_width=True)
                        elif gex_err:
                            st.error(f"Could not calculate GEX: {gex_err}")
                        else:
                            st.warning("No GEX data was returned. The asset may not have an options market on Binance or there was an API issue.")
                    except Exception as e:
                        st.error(f"Error computing GEX: {e}")

            st.divider()
            st.subheader("📈 Historical Factor Trends for Selected Asset")
            st.caption("Track how momentum, flow, volatility, and trend have changed over time for the scanned asset and selected timeframe.")

            if not st.session_state.df_deriv.empty:
                scanned_symbols = st.session_state.df_deriv['symbol'].tolist()
                history_symbol_deriv = st.selectbox("Select scanned crypto for historical factor chart", options=scanned_symbols, key="history_symbol_deriv")

                if st.button(f"Show historic factor chart for {history_symbol_deriv}", key="show_history_factors_deriv"):
                    with st.spinner(f"Loading historical factors for {history_symbol_deriv}..."):
                        history_df = build_derivative_factor_history(history_symbol_deriv, timeframe=timeframe_deriv, flow_timeframe=flow_timeframe)
                        if history_df is not None and not history_df.empty:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(x=history_df.index, y=history_df['momentum_z'], mode='lines', name='Momentum Z', line=dict(color='#00C2FF')))
                            fig.add_trace(go.Scatter(x=history_df.index, y=history_df['flow_z'], mode='lines', name='Flow Z', line=dict(color='#22C55E')))
                            fig.add_trace(go.Scatter(x=history_df.index, y=history_df['volatility_z'], mode='lines', name='Volatility Z', line=dict(color='#F59E0B')))
                            fig.add_trace(go.Scatter(x=history_df.index, y=history_df['trend_z'], mode='lines', name='Trend Z', line=dict(color='#EF4444')))
                            fig.add_hline(y=0, line_color='white', line_width=1, line_dash='dash')
                            fig.update_layout(
                                title=f'Historical Factor Z-Scores for {history_symbol_deriv} ({timeframe_deriv})',
                                xaxis_title='Time',
                                yaxis_title='Z-Score / Signal',
                                template='plotly_dark',
                                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            st.info("Interpretation: positive values suggest bullish momentum/flow/trend, negative values suggest weakening or bearish pressure. Use this to spot when factors are turning up or down before price breaks.")
                        else:
                            st.warning(f"Could not load historical factor data for {history_symbol_deriv}.")

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
            
    with tab2:
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
                    
                    styler = styler.map(color_metrics, subset=['roi'])
                        
                    st.dataframe(styler)

    with tab3:
        st.subheader("🏛️ Top US Indices & VIX Overview")
        st.write("Tracking S&P 500 (SPY), Nasdaq 100 (QQQ), NAS100 futures (NQ=F), Dow Jones (DIA), Dow futures (YM=F), Volatility Index (^VIX), and US Dollar Index (DX-Y.NYB).")
        # Timeframe selector for indices/stocks (15m, 1h, 4h)
        timeframe = st.selectbox("Select timeframe", ["15m", "1h", "4h"], index=1)
        overview_indices = [
            'SPY', 'QQQ', 'NQ=F', 'DIA', 'YM=F', '^VIX', 'DX-Y.NYB'
        ]

        with st.expander("US Indices Alert Settings"):
            index_alert_col1, index_alert_col2 = st.columns(2)
            with index_alert_col1:
                index_alert_enabled = st.checkbox(
                    "Enable overview alerts",
                    value=False,
                    key='index_overview_alert_enabled',
                )
            with index_alert_col2:
                index_alert_z_threshold = st.number_input(
                    "Absolute Z-Score alert level",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    key='index_overview_alert_z_threshold',
                )
            index_alert_telegram = st.checkbox(
                "Send qualifying alerts to Telegram",
                value=True,
                key='index_overview_alert_telegram',
                help="Uses the Telegram Bot Token and Chat ID entered in the sidebar.",
            )
            st.caption(
                "Triggers when the table Z-Score is greater than or equal to the positive "
                "level, or less than or equal to the negative level. Alerts are checked "
                "when indices data is refreshed."
            )
        
        if 'index_stats' not in st.session_state:
            st.session_state.index_stats = []
        if 'index_missing_symbols' not in st.session_state:
            st.session_state.index_missing_symbols = []
        if 'index_zscore_alerts' not in st.session_state:
            st.session_state.index_zscore_alerts = []
        if 'index_zscore_alert_keys' not in st.session_state:
            st.session_state.index_zscore_alert_keys = set()
        if 'daily_ghana_zscore_alerts' not in st.session_state:
            st.session_state.daily_ghana_zscore_alerts = []
        
        if st.button("Refresh Indices Data"):
            with st.spinner("Fetching US Indices Data..."):
                indices = overview_indices
                index_stats = []
                missing_indices = []
                daily_ghana_alerts = []
                for sym in indices:
                    df_idx = fetch_and_analyze(sym, timeframe=timeframe, silent=True)
                    if df_idx is not None and not df_idx.empty:
                        current = df_idx.iloc[-1]
                        previous = df_idx.iloc[-2] if len(df_idx) > 1 else current
                        is_advancing = current['close'] > previous['close']
                        is_declining = current['close'] < previous['close']
                        # Estimate daily volume (last 7 1-hour bars = 7 trading hours)
                        est_vol = df_idx['volume'].tail(7).sum()
                        # Calculate each metric for every candle. Changes are measured
                        # from the previous candle to the current selected-timeframe candle.
                        z_score_series = pd.to_numeric(df_idx['z_score'], errors='coerce').fillna(0.0)
                        historical_momentum = pd.to_numeric(
                            df_idx['momentum'], errors='coerce'
                        )
                        historical_momentum_z = (
                            (historical_momentum - historical_momentum.rolling(50).mean())
                            / historical_momentum.rolling(50).std().replace(0, np.nan)
                        )
                        alert_bar_duration = {
                            '15m': pd.Timedelta(minutes=15),
                            '1h': pd.Timedelta(hours=1),
                            '4h': pd.Timedelta(hours=4),
                        }[timeframe]
                        historical_alert_df = df_idx.copy()
                        historical_alert_df.index = (
                            pd.DatetimeIndex(historical_alert_df.index)
                            + alert_bar_duration
                        )
                        alert_z_score_series = z_score_series.copy()
                        alert_z_score_series.index = historical_alert_df.index
                        historical_momentum_z.index = historical_alert_df.index
                        daily_ghana_alerts.extend(
                            build_daily_zscore_alert_history(
                                historical_alert_df,
                                alert_z_score_series,
                                sym,
                                timeframe,
                                threshold=float(index_alert_z_threshold),
                                momentum_z_series=historical_momentum_z,
                            )
                        )
                        volume_ratio_series = (
                            pd.to_numeric(df_idx['volume'], errors='coerce')
                            / pd.to_numeric(df_idx['vol_ma'], errors='coerce').replace(0, np.nan)
                        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        cmf_series = pd.to_numeric(df_idx['cmf'], errors='coerce')
                        cmf_mean_series = cmf_series.rolling(window=20).mean()
                        cmf_std_series = cmf_series.rolling(window=20).std()
                        flow_z_series = (
                            (cmf_series - cmf_mean_series) / (cmf_std_series + 1e-9)
                        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                        signal_score_series = (
                            -z_score_series
                            + np.log1p(volume_ratio_series.clip(lower=0.0))
                            + flow_z_series
                        )

                        def latest_and_change(series):
                            clean = pd.to_numeric(series, errors='coerce').fillna(0.0)
                            latest_value = float(clean.iloc[-1])
                            prior_value = float(clean.iloc[-2]) if len(clean) > 1 else latest_value
                            return latest_value, latest_value - prior_value

                        momentum, momentum_change = latest_and_change(df_idx['momentum'])
                        z_score, z_score_change = latest_and_change(z_score_series)
                        volume_ratio, _ = latest_and_change(volume_ratio_series)
                        flow_z_score, flow_z_score_change = latest_and_change(flow_z_series)
                        signal_score, signal_score_change = latest_and_change(signal_score_series)
                        
                        index_stats.append({
                            "Symbol": sym,
                            "Candle Time": pd.Timestamp(df_idx.index[-1]),
                            "Price": current['close'],
                            "Momentum": momentum,
                            "Momentum Change": momentum_change,
                            "RSI": current['rsi'],
                            "Trend": "Bullish 🟢" if current['close'] > current['ema_50'] else "Bearish 🔴",
                            "Advancing": is_advancing,
                            "Declining": is_declining,
                            "Est. Daily Volume": est_vol,
                            "Z-Score": z_score,
                            "Z-Score Change": z_score_change,
                            "Volume Ratio": volume_ratio,
                            "Flow Z-Score": flow_z_score,
                            "Flow Z-Score Change": flow_z_score_change,
                            "Signal Score": signal_score,
                            "Signal Score Change": signal_score_change
                        })
                    else:
                        missing_indices.append(sym)
                
                # Save the results to session state
                st.session_state.index_stats = index_stats
                st.session_state.index_stats_timeframe = timeframe
                st.session_state.index_missing_symbols = missing_indices
                st.session_state.daily_ghana_zscore_alerts = sorted(
                    daily_ghana_alerts,
                    key=lambda alert: alert['Ghana Time'],
                    reverse=True,
                )
                st.session_state.daily_ghana_zscore_alert_timeframe = timeframe
                st.session_state.daily_ghana_zscore_alert_threshold = float(
                    index_alert_z_threshold
                )

                if index_alert_enabled and index_stats:
                    new_index_alerts = []
                    for row in index_stats:
                        z_score = float(row['Z-Score'])
                        z_threshold = float(index_alert_z_threshold)
                        if abs(z_score) < z_threshold:
                            continue
                        alert_side = 'POSITIVE Z-SCORE' if z_score >= z_threshold else 'NEGATIVE Z-SCORE'
                        candle_time = pd.Timestamp(row['Candle Time'])
                        alert_key = (
                            row['Symbol'], timeframe, candle_time.isoformat(), alert_side
                        )
                        if alert_key in st.session_state.index_zscore_alert_keys:
                            continue
                        alert_row = {
                            'Candle Time': candle_time,
                            'Symbol': row['Symbol'],
                            'Alert': alert_side,
                            'Price': float(row['Price']),
                            'Z-Score': z_score,
                            'Z-Score Change': float(row['Z-Score Change']),
                            'Timeframe': timeframe,
                        }
                        new_index_alerts.append(alert_row)
                        st.session_state.index_zscore_alert_keys.add(alert_key)

                        if index_alert_telegram and tg_token and tg_chat_id:
                            send_telegram_alert(
                                tg_token,
                                tg_chat_id,
                                row['Symbol'],
                                alert_side,
                                float(row['Price']),
                                f"Z-Score {z_score:+.2f}",
                                [
                                    f"Threshold: +/-{z_threshold:.2f}",
                                    f"Z-Score Change: {float(row['Z-Score Change']):+.2f}",
                                    f"Timeframe: {timeframe}",
                                ],
                            )

                    if new_index_alerts:
                        st.session_state.index_zscore_alerts = (
                            new_index_alerts + st.session_state.index_zscore_alerts
                        )[:100]
                        st.toast(
                            f"{len(new_index_alerts)} new US indices alert(s) triggered.",
                            icon="🚨",
                        )
                    else:
                        st.info("No US indices met the selected alert thresholds.")

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
            change_columns = [
                "Momentum Change", "Signal Score Change",
                "Flow Z-Score Change", "Z-Score Change"
            ]
            missing_change_columns = [col for col in change_columns if col not in df_ind.columns]
            if missing_change_columns:
                for col in missing_change_columns:
                    df_ind[col] = np.nan
                st.info("Refresh Indices Data to calculate the changes shown in brackets.")

            def combine_metric_value(value, change, percentage=False):
                if percentage:
                    value_text = f"{float(value):.2%}"
                    change_text = f"{float(change):+.2%}" if pd.notna(change) else "n/a"
                else:
                    value_text = f"{float(value):.2f}"
                    change_text = f"{float(change):+.2f}" if pd.notna(change) else "n/a"
                return f"{value_text} ({change_text})"

            # Keep one column per metric: current value followed by its change in brackets.
            df_ind['Momentum'] = df_ind.apply(
                lambda row: combine_metric_value(
                    row['Momentum'], row['Momentum Change'], percentage=True
                ),
                axis=1,
            )
            for metric, change_metric in [
                ('Signal Score', 'Signal Score Change'),
                ('Flow Z-Score', 'Flow Z-Score Change'),
                ('Z-Score', 'Z-Score Change'),
            ]:
                df_ind[metric] = df_ind.apply(
                    lambda row, metric=metric, change_metric=change_metric:
                        combine_metric_value(row[metric], row[change_metric]),
                    axis=1,
                )

            df_display = df_ind.drop(columns=['Advancing', 'Declining'])
            df_display = df_display[[
                "Symbol", "Breadth Ratio", "Breadth %", "Price", "Momentum", "RSI",
                "Trend", "Signal Score", "Volume Ratio", "Flow Z-Score",
                "Z-Score", "Est. Daily Volume"
            ]]
            loaded_display_symbols = set(df_ind['Symbol'].astype(str))
            missing_display_symbols = [
                configured_symbol for configured_symbol in overview_indices
                if configured_symbol not in loaded_display_symbols
            ]
            if missing_display_symbols:
                unavailable_rows = pd.DataFrame([
                    {
                        'Symbol': missing_symbol,
                        'Breadth Ratio': 'n/a',
                        'Breadth %': np.nan,
                        'Price': np.nan,
                        'Momentum': 'n/a (n/a)',
                        'RSI': np.nan,
                        'Trend': 'Data unavailable',
                        'Signal Score': 'n/a (n/a)',
                        'Volume Ratio': np.nan,
                        'Flow Z-Score': 'n/a (n/a)',
                        'Z-Score': 'n/a (n/a)',
                        'Est. Daily Volume': np.nan,
                    }
                    for missing_symbol in missing_display_symbols
                ])
                df_display = pd.concat(
                    [df_display, unavailable_rows], ignore_index=True
                )
                st.warning(
                    "Yahoo Finance returned no data for: "
                    + ", ".join(missing_display_symbols)
                    + ". The symbols remain listed and will be retried on refresh."
                )
            results_timeframe = st.session_state.get('index_stats_timeframe', timeframe)
            st.caption(
                f"Bracketed change = current value minus the previous {results_timeframe} candle. "
                "Breadth Ratio = Advancing / Declining. Breadth % = Advancing / Total. "
                "Signal Score = -Z-Score + ln(1 + Volume Ratio) + Flow Z-Score"
            )
            styler = df_display.style.format({
                "Price": "${:.2f}",
                "RSI": "{:.2f}",
                "Est. Daily Volume": format_large_number,
                "Volume Ratio": "{:.2f}x",
                "Breadth %": "{:.0%}"
            })
            
            styler = styler.map(
                color_metrics,
                subset=['Breadth %']
            )

            def color_bracketed_change(value):
                try:
                    change_text = str(value).rsplit('(', 1)[1].rstrip(')').replace('%', '')
                    change_value = float(change_text)
                    color = 'green' if change_value > 0 else 'red' if change_value < 0 else 'gray'
                    return f'color: {color}'
                except (ValueError, IndexError):
                    return 'color: gray'

            styler = styler.map(
                color_bracketed_change,
                subset=['Momentum', 'Signal Score', 'Flow Z-Score', 'Z-Score']
            )
                 
            st.dataframe(styler, width='stretch')

            if st.session_state.index_zscore_alerts:
                st.subheader("Recent US Indices Alerts")
                index_alerts_df = pd.DataFrame(
                    st.session_state.index_zscore_alerts
                )
                index_alerts_styler = index_alerts_df.style.format({
                    'Price': '${:.2f}',
                    'Z-Score': '{:+.2f}',
                    'Z-Score Change': '{:+.2f}',
                }).map(color_metrics, subset=['Z-Score', 'Z-Score Change'])
                st.dataframe(
                    index_alerts_styler,
                    width='stretch',
                    hide_index=True,
                )

            st.subheader("Daily Historical Z-Score Alerts — Ghana Time")
            st.caption(
                "For each Ghana calendar day, this table shows the first time each symbol "
                "had aligned Z-Score and Momentum Z values at the positive threshold, and "
                "the first aligned negative event. Ghana uses GMT (Africa/Accra) year-round."
            )
            if st.session_state.daily_ghana_zscore_alerts:
                daily_ghana_alerts_df = pd.DataFrame(
                    st.session_state.daily_ghana_zscore_alerts
                )
                daily_ghana_alerts_df['Ghana Date'] = pd.to_datetime(
                    daily_ghana_alerts_df['Ghana Date']
                ).dt.date
                available_ghana_dates = sorted(
                    daily_ghana_alerts_df['Ghana Date'].dropna().unique()
                )
                if st.session_state.get('selected_ghana_alert_date') not in available_ghana_dates:
                    st.session_state.pop('selected_ghana_alert_date', None)
                selected_ghana_alert_date = st.selectbox(
                    "Select Ghana alert date",
                    options=available_ghana_dates,
                    index=len(available_ghana_dates) - 1,
                    format_func=lambda selected_date: selected_date.strftime('%A, %B %d, %Y'),
                    key='selected_ghana_alert_date',
                )
                daily_ghana_alerts_df = daily_ghana_alerts_df.loc[
                    daily_ghana_alerts_df['Ghana Date'] == selected_ghana_alert_date
                ].copy()
                daily_ghana_alerts_df['Ghana Date'] = daily_ghana_alerts_df[
                    'Ghana Date'
                ].map(lambda alert_date: alert_date.strftime('%Y-%m-%d'))
                daily_ghana_alerts_df['Ghana Time'] = pd.to_datetime(
                    daily_ghana_alerts_df['Ghana Time'], utc=True
                ).dt.tz_convert('Africa/Accra').dt.strftime('%Y-%m-%d %H:%M')
                daily_ghana_styler = daily_ghana_alerts_df.style.format({
                    'Price': '${:.2f}',
                    'Z-Score': '{:+.2f}',
                    'Momentum Z': '{:+.2f}',
                }).map(color_metrics, subset=['Z-Score', 'Momentum Z'])
                st.dataframe(
                    daily_ghana_styler,
                    width='stretch',
                    hide_index=True,
                )
            else:
                st.info(
                    "No historical candles reached the selected positive or negative "
                    "Z-Score threshold. Refresh the indices data to rebuild this history."
                )

            # --- GEX Analysis for Major Indices / Macro Assets ---
            st.divider()
            st.subheader("🛡️ Index & Macro GEX (Gamma Exposure)")
            st.write("Analyze total GEX (notional) for major indices and macro assets. Polygon options is used when available; otherwise a volatility-based proxy is shown.")

            gex_assets = ['QQQ', 'NQ=F', 'SPY', 'DIA', 'YM=F', '^VIX', 'GBPUSD=X', 'DIX', '^FTSE', 'XAUUSD']
            gex_choice = st.selectbox("Select asset for GEX analysis", options=gex_assets, index=0)
            gex_timeframe = st.selectbox("Select timeframe for price/volatility (proxy)", options=['5m', '15m', '1h', '4h', '12h', '1d'], index=2)

            if st.button("Analyze Index GEX"):
                with st.spinner(f"Analyzing GEX for {gex_choice}..."):
                    # Try to get current price and recent volatility from candles
                    current_price = None
                    recent_vol = None
                    try:
                        df_idx = fetch_and_analyze(gex_choice, timeframe=gex_timeframe, silent=True, limit=200)
                        if df_idx is not None and not df_idx.empty:
                            current_price = float(df_idx['close'].iloc[-1])
                            returns = df_idx['close'].pct_change().dropna()
                            if len(returns) > 1:
                                recent_vol = float(returns.std() * np.sqrt(24))
                    except Exception:
                        current_price = None
                        recent_vol = None

                    gex_data = None
                    gex_error = None
                    if polygon_available:
                        try:
                            gex_data, gex_error = calculate_gamma_exposure(gex_choice, polygon_api_key, asset_type='stocks')
                        except Exception as e:
                            gex_data, gex_error = None, f'Polygon call failed: {e}'
                    else:
                        gex_error = 'Polygon package not installed. Using proxy GEX instead.'

                    if gex_data and 'gex_profile' in gex_data:
                        st.metric("Total GEX (Notional)", f"{gex_data['total_gex']:,.0f}")
                        st.metric("Zero Gamma Level", f"${gex_data['zero_gamma_level']:.2f}")
                        st.metric("Current Price", f"${gex_data['current_price']:.2f}")

                        gex_profile_df = gex_data['gex_profile'].reset_index()
                        gex_profile_df.columns = ['strike', 'gamma_exposure']
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=gex_profile_df['strike'], y=gex_profile_df['gamma_exposure'], name='Gamma Exposure'))
                        fig.add_vline(x=gex_data['zero_gamma_level'], line_width=2, line_dash="dash", line_color="yellow", annotation_text="Zero Gamma", annotation_position="top right")
                        fig.add_vline(x=gex_data['current_price'], line_width=2, line_dash="solid", line_color="cyan", annotation_text="Current Price", annotation_position="top left")
                        fig.update_layout(title=f'Gamma Exposure Profile for {gex_choice}', xaxis_title='Strike Price', yaxis_title='Gamma Exposure (Notional)', template='plotly_dark')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        # Fall back to proxy using price and volatility
                        if current_price is None:
                            st.warning("Could not determine current price for selected asset; try a different timeframe or refresh the indices data.")
                        else:
                            proxy = build_binance_gex_proxy(current_price, recent_vol)
                            summary = summarize_gex_profile(proxy, current_price)
                            st.metric("Total GEX (Notional)", f"{summary['total_gex']:,.0f}")
                            st.metric("Zero Gamma Level", f"${summary['zero_gamma_level']:.2f}")
                            st.metric("Current Price", f"${summary['current_price']:.2f}")
                            gex_profile_df = summary['gex_profile'].reset_index()
                            gex_profile_df.columns = ['strike', 'gamma_exposure']
                            fig = go.Figure()
                            fig.add_trace(go.Bar(x=gex_profile_df['strike'], y=gex_profile_df['gamma_exposure'], name='Gamma Exposure'))
                            fig.add_vline(x=summary['zero_gamma_level'], line_width=2, line_dash="dash", line_color="yellow", annotation_text="Zero Gamma", annotation_position="top right")
                            fig.add_vline(x=summary['current_price'], line_width=2, line_dash="solid", line_color="cyan", annotation_text="Current Price", annotation_position="top left")
                            fig.update_layout(title=f'Gamma Exposure Proxy for {gex_choice}', xaxis_title='Strike Price', yaxis_title='Gamma Exposure (Notional)', template='plotly_dark')
                            st.plotly_chart(fig, use_container_width=True)

        # --- Real-Time Order Flow Table ---
        st.divider()
        st.subheader("📊 Real-Time Order Flow")
        c1, c2 = st.columns(2)
        with c1:
            flow_tf = st.selectbox("Select Order Flow Timeframe", options=['5m', '15m', '1h', '4h'], index=2)
        with c2:
            flow_candles = st.number_input("Number of Candles to Analyze", min_value=50, max_value=1000, value=500, step=50)

        # This function is not available in the provided context, but is needed for the flow calculation.
        # I will add a simplified version here.
        def calculate_cumulative_flow(df):
            df['net_flow_per_candle'] = (df['volume'] * (2 * (df['close'] >= df['open']) - 1))
            df['cumulative_net_flow'] = df['net_flow_per_candle'].cumsum()
            return df

        # Initialize session state for historical flow data
        if 'order_flow_history' not in st.session_state:
            st.session_state.order_flow_history = {}

        if st.button("Refresh Order Flow"):
            with st.spinner(f"Calculating order flow for {flow_tf} timeframe..."):
                indices = ['SPY', 'QQQ', 'NQ=F', 'DIA', 'YM=F', '^VIX', 'DX-Y.NYB']
                order_flow_data = []
                st.session_state.order_flow_history = {} # Clear previous history
                for sym in indices:
                    df_flow_calc = fetch_and_analyze(sym, timeframe=flow_tf, silent=True, limit=flow_candles)
                    if df_flow_calc is not None and not df_flow_calc.empty:
                        df_flow_calc = calculate_cumulative_flow(df_flow_calc)
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
                        # Store the full historical dataframe for later viewing
                        st.session_state.order_flow_history[sym] = df_flow_calc.copy()

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

            # --- Historical Order Flow Data Section ---
            st.subheader("📜 Historical Order Flow (Last 50 Candles)")
            st.caption("Select an asset from the scan above to see its detailed historical order flow data.")

            # Get the list of symbols that were successfully scanned
            scanned_symbols = list(st.session_state.order_flow_history.keys())

            if scanned_symbols:
                history_asset = st.selectbox("Select Asset for Historical View", options=scanned_symbols)
                
                if history_asset in st.session_state.order_flow_history:
                    history_df = st.session_state.order_flow_history[history_asset]
                    # Display the last 50 rows as requested
                    st.subheader(f"Cumulative Net Flow for {history_asset}")
                    if 'cumulative_net_flow' in history_df.columns:
                        st.line_chart(history_df['cumulative_net_flow'])
                    else:
                        st.warning("Cumulative net flow data not available. Please refresh order flow.")

                    st.dataframe(history_df.tail(50), width='stretch', height=300)
                else:
                    st.info(f"No historical data available for {history_asset}. Please refresh the order flow.")


        st.divider()
        st.subheader("💰 Daily ETF Inflow / Outflow (SPY, QQQ, NQ=F, DIA, YM=F, BTC/USD)")
        st.caption("This section shows intraday inflow and outflow for the major index ETFs and BTC/USD across 5m, 15m, 1h, and 4h candles so you can monitor real-time market momentum.")

        timeframes = [('5m', '5m'), ('15m', '15m'), ('1h', '1h'), ('4h', '4h')]
        daily_flow_rows = []
        for symbol in ['SPY', 'QQQ', 'NQ=F', 'DIA', 'YM=F', 'BTC/USD']:
            row = {'Symbol': symbol}
            for label, interval in timeframes:
                flow_summary = get_intraday_money_flow(symbol, interval=interval, period='1d')
                if flow_summary:
                    row[f'{label} Inflow'] = flow_summary['inflow']
                    row[f'{label} Outflow'] = flow_summary['outflow']
                    row[f'{label} Net Flow'] = flow_summary['net_flow']
                else:
                    row[f'{label} Inflow'] = 0.0
                    row[f'{label} Outflow'] = 0.0
                    row[f'{label} Net Flow'] = 0.0
            daily_flow_rows.append(row)

        if daily_flow_rows:
            daily_flow_df = pd.DataFrame(daily_flow_rows)
            # Fill NaNs with 0 only for numeric columns to avoid errors with object columns
            for col in daily_flow_df.select_dtypes(include=np.number).columns:
                daily_flow_df[col] = daily_flow_df[col].fillna(0)

            sort_col = '5m Net Flow'
            if sort_col in daily_flow_df.columns:
                daily_flow_df = daily_flow_df.sort_values(by=sort_col, ascending=False)
            flow_columns = [col for col in daily_flow_df.columns if col != 'Symbol']
            daily_flow_styler = daily_flow_df.style.format({col: format_large_number for col in flow_columns})
            daily_flow_styler = daily_flow_styler.background_gradient(subset=['5m Net Flow', '15m Net Flow', '1h Net Flow', '4h Net Flow'], cmap='RdYlGn')
            st.dataframe(daily_flow_styler, width='stretch')

        else:
            st.info("Daily ETF flow data is currently unavailable.")

        # --- QS Score Comparison for SPY, QQQ, DIA ---
        st.divider()
        st.subheader("Macro Quant Strength (QS) Comparison")
        st.caption("This table compares major indices and currency pairs using a relative strength model. A higher score indicates stronger performance versus the group.")

        indices_for_qs = ['SPY', 'QQQ', 'NQ=F', 'DIA', 'YM=F', 'DX-Y.NYB', '^FTSE', 'GBPUSD=X', 'XAUUSD']
        if st.button("Refresh Macro QS Comparison"):
            qs_indices_data = []
            with st.spinner("Fetching macro comparison data..."):
                for sym in indices_for_qs:
                    df_idx_qs = fetch_and_analyze(sym, timeframe=timeframe, silent=True)
                    if df_idx_qs is not None and not df_idx_qs.empty:
                        current = df_idx_qs.iloc[-1]
                        df_idx_qs['is_up'] = df_idx_qs['close'] >= df_idx_qs['open']
                        inflow = float(df_idx_qs.loc[df_idx_qs['is_up'], 'volume'].sum())
                        outflow = float(df_idx_qs.loc[~df_idx_qs['is_up'], 'volume'].sum())
                        total_flow = inflow + outflow
                        qs_indices_data.append({
                            'symbol': sym,
                            'momentum': current.get('momentum', 0.0),
                            'money_flow_signal': (inflow - outflow) / total_flow if total_flow > 0 else 0.0,
                            'atr14': current.get('atr14', 0.0),
                            'z_score_raw': current.get('z_score', 0.0)
                        })

            if len(qs_indices_data) >= 3:
                st.session_state['macro_qs_data'] = pd.DataFrame(qs_indices_data)
            else:
                st.session_state['macro_qs_data'] = pd.DataFrame()
                st.warning("Could not fetch enough data for the Macro QS comparison.")

        df_qs_indices = st.session_state.get('macro_qs_data', pd.DataFrame()).copy()
        if not df_qs_indices.empty:
            
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
            st.info("Click 'Refresh Macro QS Comparison' to load the macro comparison basket.")

        # --- Historical Z-Score Chart ---
        st.divider()
        st.subheader("Historical Z-Score Component Analysis")
        st.caption("This chart shows the historical Z-scores for each component of the QS score for a single asset. This helps identify which factors are strengthening or weakening over time. A value of +2 means the factor is 2 standard deviations above its own recent history.")

        z_chart_base_symbols = (
            df_qs_indices['symbol'].tolist()
            if not df_qs_indices.empty
            else indices_for_qs
        )
        z_chart_symbols = list(dict.fromkeys(
            ['BTC/USD', 'NQ=F', 'YM=F'] + z_chart_base_symbols
        ))
        if z_chart_symbols:
            z_chart_asset = st.selectbox("Select Asset for Z-Score Chart", options=z_chart_symbols)
            z_chart_timeframe = st.selectbox("Select timeframe for historical components", options=['5m', '15m', '1h', '4h', '12h', '1d'], index=2, key='z_chart_timeframe')

            z_chart_config = (z_chart_asset, z_chart_timeframe)
            if st.button(f"Generate Historical Z-Chart for {z_chart_asset}"):
                with st.spinner(f"Calculating historical Z-scores for {z_chart_asset}..."):
                    df_z_hist = fetch_and_analyze(z_chart_asset, timeframe=z_chart_timeframe, silent=True)
                if df_z_hist is not None and not df_z_hist.empty:
                    st.session_state['historical_z_chart_data'] = df_z_hist
                    st.session_state['historical_z_chart_config'] = z_chart_config
                else:
                    st.error(f"Could not fetch data to generate Z-chart for {z_chart_asset}.")

            # Retain the fetched history so changing the date redraws the analysis without another API call.
            if st.session_state.get('historical_z_chart_config') == z_chart_config:
                df_z_hist = st.session_state.get('historical_z_chart_data').copy()
                historical_index = pd.DatetimeIndex(df_z_hist.index)
                if historical_index.tz is None:
                    ghana_history_index = historical_index.tz_localize('UTC')
                else:
                    ghana_history_index = historical_index.tz_convert('UTC')
                available_dates = sorted(np.unique(ghana_history_index.date))

                if available_dates:
                    date_key = f"z_chart_analysis_date_{z_chart_asset}_{z_chart_timeframe}"
                    with st.popover("Select analysis date"):
                        selected_z_date = st.date_input(
                            "Date to analyze",
                            value=available_dates[-1],
                            min_value=available_dates[0],
                            max_value=available_dates[-1],
                            key=date_key,
                        )
                    st.caption(f"Showing component analysis for {selected_z_date:%B %d, %Y}.")

                    # Calculate each component against the complete fetched history, then show the chosen date.
                    df_z_hist['is_up'] = df_z_hist['close'] >= df_z_hist['open']
                    inflow = df_z_hist['volume'].where(df_z_hist['is_up'], 0)
                    outflow = df_z_hist['volume'].where(~df_z_hist['is_up'], 0)
                    rolling_inflow = inflow.rolling(window=20).sum()
                    rolling_outflow = outflow.rolling(window=20).sum()
                    df_z_hist['money_flow_signal'] = (rolling_inflow - rolling_outflow) / (rolling_inflow + rolling_outflow)

                    def rolling_zscore(series, window=50):
                        return (series - series.rolling(window).mean()) / series.rolling(window).std()

                    full_z_df = pd.DataFrame(index=df_z_hist.index)
                    full_z_df['Momentum (z)'] = rolling_zscore(df_z_hist['momentum'])
                    full_z_df['Flow (z)'] = rolling_zscore(df_z_hist['money_flow_signal'])
                    full_z_df['Volatility (z)'] = rolling_zscore(df_z_hist['atr14'])
                    full_z_df['Trend (z)'] = rolling_zscore(df_z_hist['z_score'])
                    full_component_changes = full_z_df.diff()
                    selected_ghana_mask = np.asarray(
                        ghana_history_index.date
                    ) == selected_z_date
                    z_df = full_z_df.loc[selected_ghana_mask]
                    selected_component_changes = full_component_changes.loc[
                        selected_ghana_mask
                    ]

                    if z_df.empty:
                        st.info("No historical Z-score data is available for the selected date.")
                    else:
                        fig_z = go.Figure()
                        for col in z_df.columns:
                            fig_z.add_trace(go.Scatter(x=z_df.index, y=z_df[col], mode='lines', name=col))

                        fig_z.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
                        fig_z.update_layout(title=f'Historical Z-Score Components for {z_chart_asset} — {selected_z_date:%Y-%m-%d}',
                                            yaxis_title='Z-Score (Standard Deviations from Mean)',
                                            template='plotly_dark')
                        st.plotly_chart(fig_z, width='stretch')

                        # All Flow Z rise/drop alerts during the selected Ghana day.
                        selected_session_alerts = []
                        selected_flow_changes = selected_component_changes[
                            'Flow (z)'
                        ].dropna()
                        selected_event_keys = [
                            (
                                'Flow Z Rise Alert'
                                if flow_change_value > 0
                                else 'Flow Z Drop Alert',
                                flow_change_time,
                            )
                            for flow_change_time, flow_change_value
                            in selected_flow_changes.items()
                            if flow_change_value != 0
                        ]

                        for alert_name, alert_time in selected_event_keys:
                            flow_change_alert = float(
                                selected_component_changes.at[alert_time, 'Flow (z)']
                            )
                            momentum_change_alert = float(
                                selected_component_changes.at[alert_time, 'Momentum (z)']
                            )
                            volatility_change_alert = float(
                                selected_component_changes.at[alert_time, 'Volatility (z)']
                            )
                            trend_change_alert = float(
                                selected_component_changes.at[alert_time, 'Trend (z)']
                            )
                            alert_price_row = df_z_hist.loc[alert_time]
                            if isinstance(alert_price_row, pd.DataFrame):
                                alert_price_row = alert_price_row.iloc[-1]
                            alert_position = df_z_hist.index.get_indexer([alert_time])[0]
                            alert_previous_close = (
                                float(df_z_hist['close'].iloc[alert_position - 1])
                                if alert_position > 0
                                else float(alert_price_row['open'])
                            )
                            alert_price_change = (
                                float(alert_price_row['close']) - alert_previous_close
                            )
                            alert_flow_contribution = (
                                0.60 * np.tanh(flow_change_alert)
                            )
                            alert_momentum_contribution = (
                                0.25 * np.tanh(momentum_change_alert)
                            )
                            alert_volatility_direction = (
                                np.sign(alert_price_change)
                                * np.tanh(abs(volatility_change_alert))
                                if volatility_change_alert > 0 else 0.0
                            )
                            alert_volatility_contribution = (
                                0.15 * alert_volatility_direction
                            )
                            alert_direction_score = 100 * (
                                alert_flow_contribution
                                + alert_momentum_contribution
                                + alert_volatility_contribution
                            )
                            if alert_direction_score >= 10:
                                alert_classification = 'Demand Alert'
                            elif alert_direction_score <= -10:
                                alert_classification = 'Supply Alert'
                            else:
                                alert_classification = 'Unresolved Alert'
                            alert_timestamp = pd.Timestamp(alert_time)
                            if alert_timestamp.tzinfo is None:
                                alert_ghana_time = alert_timestamp.tz_localize('UTC')
                            else:
                                alert_ghana_time = alert_timestamp.tz_convert('UTC')
                            selected_session_alerts.append({
                                'Asset': z_chart_asset,
                                'Date': selected_z_date,
                                'Timeframe': z_chart_timeframe,
                                'Alert': alert_name,
                                'Ghana Time': alert_ghana_time,
                                'Flow Z': float(z_df.at[alert_time, 'Flow (z)']),
                                'Flow Z Change': flow_change_alert,
                                'Momentum Z Change': momentum_change_alert,
                                'Trend Z Change': trend_change_alert,
                                'Volatility Z Change': volatility_change_alert,
                                'Price Change': alert_price_change,
                                'Zone Bottom': float(alert_price_row['low']),
                                'Zone Top': float(alert_price_row['high']),
                                'Direction Score': alert_direction_score,
                                'Demand/Supply Alert': alert_classification,
                            })

                        st.subheader(
                            "All Historical Demand & Supply Rise/Drop Alerts — Ghana Day"
                        )
                        st.caption(
                            f"All Flow Z rise and drop events for {z_chart_asset} during the "
                            f"Ghana calendar day {selected_z_date:%Y-%m-%d}, using "
                            f"{z_chart_timeframe} candles. The first event after Ghana midnight "
                            "is compared with the preceding candle."
                        )
                        if selected_session_alerts:
                            selected_alerts_df = pd.DataFrame(selected_session_alerts)
                            alert_counts = selected_alerts_df['Alert'].value_counts()
                            alert_count_columns = st.columns(3)
                            alert_count_columns[0].metric(
                                "Total Alerts", len(selected_alerts_df)
                            )
                            alert_count_columns[1].metric(
                                "Rise Alerts",
                                int(alert_counts.get('Flow Z Rise Alert', 0)),
                            )
                            alert_count_columns[2].metric(
                                "Drop Alerts",
                                int(alert_counts.get('Flow Z Drop Alert', 0)),
                            )
                            selected_alerts_df = selected_alerts_df.sort_values(
                                'Ghana Time'
                            )
                            st.dataframe(
                                selected_alerts_df.style.format({
                                    'Flow Z': '{:+.2f}',
                                    'Flow Z Change': '{:+.2f}',
                                    'Momentum Z Change': '{:+.2f}',
                                    'Trend Z Change': '{:+.2f}',
                                    'Volatility Z Change': '{:+.2f}',
                                    'Price Change': '{:+,.2f}',
                                    'Zone Bottom': '{:,.2f}',
                                    'Zone Top': '{:,.2f}',
                                    'Direction Score': '{:+.1f}',
                                }).map(
                                    color_metrics,
                                    subset=[
                                        'Flow Z', 'Flow Z Change',
                                        'Momentum Z Change', 'Trend Z Change',
                                        'Volatility Z Change', 'Price Change',
                                        'Direction Score',
                                    ],
                                ),
                                width='stretch',
                                hide_index=True,
                            )
                        else:
                            st.info(
                                "No valid Flow Z rise or drop alert exists for this asset "
                                "during the selected date/session."
                            )

                        drop_events = []
                        for col in z_df.columns:
                            series = z_df[col].dropna()
                            if len(series) < 2:
                                continue
                            changes = series.diff().dropna()
                            for idx in changes[changes < 0].index:
                                prev_val = float(series.shift(1).loc[idx])
                                curr_val = float(series.loc[idx])
                                drop_events.append({
                                    'Component': col,
                                    'Drop Time': pd.Timestamp(idx),
                                    'Previous Value': prev_val,
                                    'Current Value': curr_val,
                                    'Drop Size': prev_val - curr_val
                                })

                        if drop_events:
                            drops_df = pd.DataFrame(drop_events).sort_values('Drop Time', ascending=False).head(50)
                            drops_df['Drop Time'] = drops_df['Drop Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                            st.subheader("Drop Events for Selected Date (up to 50)")
                            st.dataframe(drops_df[['Component', 'Drop Time', 'Previous Value', 'Current Value', 'Drop Size']].style.format({
                                'Previous Value': '{:.3f}',
                                'Current Value': '{:.3f}',
                                'Drop Size': '{:.3f}'
                            }), width='stretch')

                            # --- Demand / Supply analysis at the largest Flow Z drop ---
                            flow_drop_events = [
                                event for event in drop_events
                                if event['Component'] == 'Flow (z)'
                            ]
                            if not flow_drop_events:
                                st.info(
                                    "No downward Flow Z event was detected on the selected date, "
                                    "so no demand/supply zone can be classified from Flow."
                                )
                            highest_drop = max(
                                flow_drop_events,
                                key=lambda event: event['Drop Size'],
                                default={
                                    'Component': 'Flow (z)',
                                    'Drop Time': z_df.index[0],
                                    'Previous Value': np.nan,
                                    'Current Value': np.nan,
                                    'Drop Size': np.nan,
                                },
                            )
                            highest_drop_time = pd.Timestamp(highest_drop['Drop Time'])
                            drop_position = z_df.index.get_indexer([highest_drop_time])[0]
                            if drop_position > 0:
                                current_components = z_df.iloc[drop_position]
                                previous_components = z_df.iloc[drop_position - 1]
                                flow_change = float(
                                    current_components['Flow (z)']
                                    - previous_components['Flow (z)']
                                )
                                momentum_change = float(
                                    current_components['Momentum (z)']
                                    - previous_components['Momentum (z)']
                                )
                                volatility_change = float(
                                    current_components['Volatility (z)']
                                    - previous_components['Volatility (z)']
                                )

                                price_row = df_z_hist.loc[highest_drop_time]
                                if isinstance(price_row, pd.DataFrame):
                                    price_row = price_row.iloc[-1]
                                full_price_position = df_z_hist.index.get_indexer(
                                    [highest_drop_time]
                                )[0]
                                previous_close = (
                                    float(df_z_hist['close'].iloc[full_price_position - 1])
                                    if full_price_position > 0
                                    else float(price_row['open'])
                                )
                                price_change = float(price_row['close']) - previous_close

                                flow_contribution = 0.60 * np.tanh(flow_change)
                                momentum_contribution = 0.25 * np.tanh(momentum_change)
                                # Volatility supplies magnitude, not direction. Expansion
                                # confirms the direction of the contemporaneous price move;
                                # compression is recorded as neutral directional evidence.
                                volatility_direction = (
                                    np.sign(price_change) * np.tanh(abs(volatility_change))
                                    if volatility_change > 0 else 0.0
                                )
                                volatility_contribution = 0.15 * volatility_direction
                                zone_direction_score = 100 * (
                                    flow_contribution
                                    + momentum_contribution
                                    + volatility_contribution
                                )
                                if zone_direction_score >= 10:
                                    zone_classification = 'Candidate Demand Zone'
                                elif zone_direction_score <= -10:
                                    zone_classification = 'Candidate Supply Zone'
                                else:
                                    zone_classification = 'Unresolved / Balanced Zone'

                                volatility_regime = (
                                    'Expansion'
                                    if volatility_change > 0
                                    else 'Compression'
                                    if volatility_change < 0
                                    else 'Unchanged'
                                )
                                st.subheader(
                                    "Demand & Supply Analysis — Highest Flow Z Drop"
                                )
                                st.caption(
                                    "Zone Direction Score = 60% Flow Z change + 25% "
                                    "Momentum Z change + 15% directional volatility regime. "
                                    "The candle range is a candidate zone, not order-book proof."
                                )
                                demand_supply_metrics = st.columns(5)
                                demand_supply_metrics[0].metric(
                                    "Classification", zone_classification
                                )
                                demand_supply_metrics[1].metric(
                                    "Direction Score",
                                    f"{zone_direction_score:+.1f}",
                                )
                                demand_supply_metrics[2].metric(
                                    "Analyzed Component",
                                    "Flow (z)",
                                )
                                demand_supply_metrics[3].metric(
                                    "Drop Size", f"{highest_drop['Drop Size']:.2f}σ"
                                )
                                demand_supply_metrics[4].metric(
                                    "Volatility Regime", volatility_regime
                                )

                                zone_details = pd.DataFrame([{
                                    'Asset': z_chart_asset,
                                    'Event Time': highest_drop_time,
                                    'Zone Bottom': float(price_row['low']),
                                    'Zone Top': float(price_row['high']),
                                    'Close': float(price_row['close']),
                                    'Flow Z Change': flow_change,
                                    'Momentum Z Change': momentum_change,
                                    'Volatility Z Change': volatility_change,
                                    'Price Change': price_change,
                                    'Flow Contribution': flow_contribution * 100,
                                    'Momentum Contribution': momentum_contribution * 100,
                                    'Volatility Contribution': volatility_contribution * 100,
                                }])
                                st.dataframe(
                                    zone_details.style.format({
                                        'Zone Bottom': '{:.2f}',
                                        'Zone Top': '{:.2f}',
                                        'Close': '{:.2f}',
                                        'Flow Z Change': '{:+.3f}',
                                        'Momentum Z Change': '{:+.3f}',
                                        'Volatility Z Change': '{:+.3f}',
                                        'Price Change': '{:+.2f}',
                                        'Flow Contribution': '{:+.1f}',
                                        'Momentum Contribution': '{:+.1f}',
                                        'Volatility Contribution': '{:+.1f}',
                                    }),
                                    width='stretch',
                                    hide_index=True,
                                )
                        else:
                            st.info("No downward moves were detected on the selected date.")

                        # --- Demand / Supply analysis at the largest Flow Z rise ---
                        flow_series_for_rise = z_df['Flow (z)'].dropna()
                        flow_rise_changes = flow_series_for_rise.diff().dropna()
                        positive_flow_rises = flow_rise_changes[
                            flow_rise_changes > 0
                        ]
                        if positive_flow_rises.empty:
                            st.info(
                                "No upward Flow Z event was detected on the selected date, "
                                "so no rise-event demand/supply zone can be classified."
                            )
                        else:
                            highest_rise_time = pd.Timestamp(
                                positive_flow_rises.idxmax()
                            )
                            highest_rise_size = float(
                                positive_flow_rises.loc[highest_rise_time]
                            )
                            rise_position = z_df.index.get_indexer(
                                [highest_rise_time]
                            )[0]
                            if rise_position > 0:
                                rise_current = z_df.iloc[rise_position]
                                rise_previous = z_df.iloc[rise_position - 1]
                                rise_flow_change = float(
                                    rise_current['Flow (z)']
                                    - rise_previous['Flow (z)']
                                )
                                rise_momentum_change = float(
                                    rise_current['Momentum (z)']
                                    - rise_previous['Momentum (z)']
                                )
                                rise_volatility_change = float(
                                    rise_current['Volatility (z)']
                                    - rise_previous['Volatility (z)']
                                )

                                rise_price_row = df_z_hist.loc[highest_rise_time]
                                if isinstance(rise_price_row, pd.DataFrame):
                                    rise_price_row = rise_price_row.iloc[-1]
                                rise_full_position = df_z_hist.index.get_indexer(
                                    [highest_rise_time]
                                )[0]
                                rise_previous_close = (
                                    float(df_z_hist['close'].iloc[rise_full_position - 1])
                                    if rise_full_position > 0
                                    else float(rise_price_row['open'])
                                )
                                rise_price_change = (
                                    float(rise_price_row['close'])
                                    - rise_previous_close
                                )

                                rise_flow_contribution = (
                                    0.60 * np.tanh(rise_flow_change)
                                )
                                rise_momentum_contribution = (
                                    0.25 * np.tanh(rise_momentum_change)
                                )
                                rise_volatility_direction = (
                                    np.sign(rise_price_change)
                                    * np.tanh(abs(rise_volatility_change))
                                    if rise_volatility_change > 0
                                    else 0.0
                                )
                                rise_volatility_contribution = (
                                    0.15 * rise_volatility_direction
                                )
                                rise_direction_score = 100 * (
                                    rise_flow_contribution
                                    + rise_momentum_contribution
                                    + rise_volatility_contribution
                                )
                                if rise_direction_score >= 10:
                                    rise_classification = 'Candidate Demand Zone'
                                elif rise_direction_score <= -10:
                                    rise_classification = 'Candidate Supply Zone'
                                else:
                                    rise_classification = 'Unresolved / Balanced Zone'

                                rise_volatility_regime = (
                                    'Expansion'
                                    if rise_volatility_change > 0
                                    else 'Compression'
                                    if rise_volatility_change < 0
                                    else 'Unchanged'
                                )
                                st.subheader(
                                    "Demand & Supply Analysis — Highest Flow Z Rise"
                                )
                                st.caption(
                                    "The largest upward Flow Z event is the primary demand "
                                    "candidate. Momentum Z and directional volatility provide "
                                    "confirmation using the same 60% / 25% / 15% weighting."
                                )
                                rise_metrics = st.columns(5)
                                rise_metrics[0].metric(
                                    "Classification", rise_classification
                                )
                                rise_metrics[1].metric(
                                    "Direction Score",
                                    f"{rise_direction_score:+.1f}",
                                )
                                rise_metrics[2].metric(
                                    "Analyzed Component", "Flow (z)"
                                )
                                rise_metrics[3].metric(
                                    "Rise Size", f"{highest_rise_size:.2f}σ"
                                )
                                rise_metrics[4].metric(
                                    "Volatility Regime", rise_volatility_regime
                                )

                                rise_zone_details = pd.DataFrame([{
                                    'Asset': z_chart_asset,
                                    'Event Time': highest_rise_time,
                                    'Zone Bottom': float(rise_price_row['low']),
                                    'Zone Top': float(rise_price_row['high']),
                                    'Close': float(rise_price_row['close']),
                                    'Flow Z Change': rise_flow_change,
                                    'Momentum Z Change': rise_momentum_change,
                                    'Volatility Z Change': rise_volatility_change,
                                    'Price Change': rise_price_change,
                                    'Flow Contribution': rise_flow_contribution * 100,
                                    'Momentum Contribution': rise_momentum_contribution * 100,
                                    'Volatility Contribution': rise_volatility_contribution * 100,
                                }])
                                st.dataframe(
                                    rise_zone_details.style.format({
                                        'Zone Bottom': '{:.2f}',
                                        'Zone Top': '{:.2f}',
                                        'Close': '{:.2f}',
                                        'Flow Z Change': '{:+.3f}',
                                        'Momentum Z Change': '{:+.3f}',
                                        'Volatility Z Change': '{:+.3f}',
                                        'Price Change': '{:+.2f}',
                                        'Flow Contribution': '{:+.1f}',
                                        'Momentum Contribution': '{:+.1f}',
                                        'Volatility Contribution': '{:+.1f}',
                                    }),
                                    width='stretch',
                                    hide_index=True,
                                )

        # --- Zone Battle Score ---
        st.divider()
        st.subheader("⚔️ Zone Battle Score")
        st.caption(
            "Measures who won after price entered a bullish or bearish FVG zone. "
            "Battle Score = 0.40(Rz) + 0.30(Fz) + 0.20(Mz) + 0.10(Az)."
        )

        zone_col1, zone_col2, zone_col3 = st.columns(3)
        with zone_col1:
            zone_asset = st.selectbox(
                "Zone Battle asset",
                options=[
                    'SPY', 'QQQ', 'NQ=F', 'DIA', 'YM=F', '^VIX', 'DX-Y.NYB',
                    '^FTSE', 'XAUUSD', 'GBPUSD=X',
                ],
                index=0,
                key='zone_battle_asset',
            )
        with zone_col2:
            zone_timeframe = st.selectbox(
                "Zone Battle timeframe",
                options=['5m', '15m', '1h', '4h', '1d'],
                index=2,
                key='zone_battle_timeframe',
            )
        with zone_col3:
            zone_observation_bars = st.number_input(
                "Reaction observation candles",
                min_value=3,
                max_value=50,
                value=10,
                step=1,
                key='zone_observation_bars',
            )

        zone_config = (
            zone_asset,
            zone_timeframe,
            int(zone_observation_bars),
        )
        if st.button(
            f"Analyze Zone Battles for {zone_asset}",
            key='run_zone_battle_analysis',
        ):
            with st.spinner(
                f"Detecting {zone_asset} zones on {zone_timeframe} candles..."
            ):
                zone_source_df = fetch_and_analyze(
                    zone_asset,
                    timeframe=zone_timeframe,
                    silent=True,
                )
                zone_results, zone_indicator_data = calculate_zone_battle_scores(
                    zone_source_df,
                    observation_bars=int(zone_observation_bars),
                    max_zones=50,
                )
            st.session_state['zone_battle_results'] = zone_results
            st.session_state['zone_battle_indicators'] = zone_indicator_data
            st.session_state['zone_battle_config'] = zone_config

        saved_zone_config = st.session_state.get('zone_battle_config')
        if saved_zone_config == zone_config:
            zone_results = st.session_state.get(
                'zone_battle_results', pd.DataFrame()
            )
            if not zone_results.empty:
                latest_zone = zone_results.sort_values(
                    'Zone Entry', ascending=False
                ).iloc[0]
                battle_col1, battle_col2, battle_col3, battle_col4 = st.columns(4)
                battle_col1.metric(
                    "Latest Battle Score",
                    f"{latest_zone['Battle Score']:+.2f}",
                )
                battle_col2.metric("Result", latest_zone['Result'])
                battle_col3.metric("Zone Type", latest_zone['Zone Type'])
                battle_col4.metric(
                    "Zone Range",
                    f"{latest_zone['Zone Bottom']:.2f} – {latest_zone['Zone Top']:.2f}",
                )

                score_chart_df = zone_results.sort_values('Zone Entry')
                zone_fig = go.Figure()
                zone_fig.add_trace(go.Scatter(
                    x=score_chart_df['Zone Entry'],
                    y=score_chart_df['Battle Score'],
                    mode='lines+markers',
                    name='Battle Score',
                    marker=dict(
                        color=np.where(
                            score_chart_df['Battle Score'] > 1,
                            '#22c55e',
                            np.where(
                                score_chart_df['Battle Score'] < -1,
                                '#ef4444',
                                '#f59e0b',
                            ),
                        ),
                        size=8,
                    ),
                ))
                zone_fig.add_hline(
                    y=1.0,
                    line_dash='dash',
                    line_color='#22c55e',
                    annotation_text='Strong Buyer Victory',
                )
                zone_fig.add_hline(
                    y=-1.0,
                    line_dash='dash',
                    line_color='#ef4444',
                    annotation_text='Strong Seller Victory',
                )
                zone_fig.update_layout(
                    title=f'{zone_asset} Zone Battle History ({zone_timeframe})',
                    xaxis_title='Zone Entry Time',
                    yaxis_title='Battle Score',
                    template='plotly_dark',
                )
                st.plotly_chart(zone_fig, width='stretch')

                zone_display_columns = [
                    'Zone Entry', 'Zone Type', 'Zone Bottom', 'Zone Top',
                    'Entry Price', 'Reaction Close', 'Bars Observed',
                    'Reaction Rz', 'Flow Fz', 'Momentum Mz',
                    'Acceptance Az', 'Battle Score', 'Result',
                ]
                zone_styler = zone_results[zone_display_columns].style.format({
                    'Zone Bottom': '{:.2f}',
                    'Zone Top': '{:.2f}',
                    'Entry Price': '{:.2f}',
                    'Reaction Close': '{:.2f}',
                    'Reaction Rz': '{:+.2f}',
                    'Flow Fz': '{:+.2f}',
                    'Momentum Mz': '{:+.2f}',
                    'Acceptance Az': '{:+.2f}',
                    'Battle Score': '{:+.2f}',
                }).map(
                    color_metrics,
                    subset=[
                        'Reaction Rz', 'Flow Fz', 'Momentum Mz',
                        'Acceptance Az', 'Battle Score',
                    ],
                )
                st.dataframe(zone_styler, width='stretch', hide_index=True)

                with st.expander("How the Zone Battle components are calculated"):
                    st.markdown(
                        """
- **Reaction Rz:** `(reaction close - zone entry price) / ATR(15)`.
- **Flow Fz:** the Historical Flow Z-score at the end of the reaction window.
- **Momentum Mz:** the Historical Momentum Z-score at the end of the reaction window.
- **Acceptance Az:** average candle location: `+1` above the zone, `0` inside it, and `-1` below it.
- **Battle Score > +1:** strong buyer victory.
- **Battle Score < -1:** strong seller victory.
- **Between -1 and +1:** battle unresolved.
                        """
                    )
            else:
                st.info(
                    "No completed FVG zone reactions were found for this asset and timeframe."
                )
        elif saved_zone_config is not None:
            st.info(
                "The asset, timeframe, or observation window changed. "
                "Run the analysis again to refresh the Zone Battle results."
            )

        # --- Liquidity Zones ---
        st.divider()
        st.subheader("Liquidity Zones")
        st.caption(
            "Ranks price bands by volume, absolute order-flow imbalance, ATR "
            "compression, swing density, time at price, and proximity to VWAP. "
            "The combined Liquidity Score is normalized from 0 to 100."
        )

        liquidity_col1, liquidity_col2, liquidity_col3 = st.columns(3)
        with liquidity_col1:
            liquidity_asset = st.selectbox(
                "Liquidity Zone asset",
                options=[
                    'SPY', 'QQQ', 'NQ=F', 'DIA', 'YM=F', '^VIX', 'DX-Y.NYB',
                    '^FTSE', 'XAUUSD', 'GBPUSD=X',
                ],
                index=0,
                key='liquidity_zone_asset',
            )
        with liquidity_col2:
            liquidity_timeframe = st.selectbox(
                "Liquidity Zone timeframe",
                options=['5m', '15m', '1h', '4h', '1d'],
                index=2,
                key='liquidity_zone_timeframe',
            )
        with liquidity_col3:
            liquidity_price_bins = st.number_input(
                "Number of price bands",
                min_value=10,
                max_value=60,
                value=24,
                step=2,
                key='liquidity_zone_price_bins',
            )

        liquidity_config = (
            liquidity_asset, liquidity_timeframe, int(liquidity_price_bins)
        )
        if st.button(
            f"Find Liquidity Zones for {liquidity_asset}",
            key='run_liquidity_zone_analysis',
        ):
            with st.spinner(
                f"Mapping {liquidity_asset} liquidity on {liquidity_timeframe} candles..."
            ):
                liquidity_source_df = fetch_and_analyze(
                    liquidity_asset,
                    timeframe=liquidity_timeframe,
                    silent=True,
                    limit=750,
                )
                liquidity_zones, liquidity_history = calculate_liquidity_zones(
                    liquidity_source_df,
                    price_bins=int(liquidity_price_bins),
                    top_zones=8,
                )
            st.session_state['liquidity_zone_results'] = liquidity_zones
            st.session_state['liquidity_zone_history'] = liquidity_history
            st.session_state['liquidity_zone_config'] = liquidity_config

        saved_liquidity_config = st.session_state.get('liquidity_zone_config')
        if saved_liquidity_config == liquidity_config:
            liquidity_zones = st.session_state.get(
                'liquidity_zone_results', pd.DataFrame()
            )
            liquidity_history = st.session_state.get(
                'liquidity_zone_history', pd.DataFrame()
            )
            if not liquidity_zones.empty and not liquidity_history.empty:
                strongest_zone = liquidity_zones.iloc[0]
                latest_liquidity_price = float(liquidity_history['close'].iloc[-1])
                liquidity_metric1, liquidity_metric2, liquidity_metric3, liquidity_metric4 = st.columns(4)
                liquidity_metric1.metric(
                    "Strongest Zone",
                    f"{strongest_zone['Zone Bottom']:.2f} - {strongest_zone['Zone Top']:.2f}",
                )
                liquidity_metric2.metric(
                    "Liquidity Score", f"{strongest_zone['Liquidity Score']:.1f}/100"
                )
                liquidity_metric3.metric("Flow Bias", strongest_zone['Flow Bias'])
                liquidity_metric4.metric("Last Price", f"{latest_liquidity_price:.2f}")

                liquidity_fig = go.Figure()
                liquidity_fig.add_trace(go.Bar(
                    x=liquidity_zones['Liquidity Score'],
                    y=liquidity_zones['Zone Midpoint'],
                    orientation='h',
                    width=(
                        liquidity_zones['Zone Top'] - liquidity_zones['Zone Bottom']
                    ) * 0.82,
                    marker=dict(
                        color=liquidity_zones['Order Flow Imbalance'],
                        colorscale=[[0, '#ef4444'], [0.5, '#64748b'], [1, '#22c55e']],
                        cmin=-1,
                        cmax=1,
                        colorbar=dict(title='Flow'),
                    ),
                    customdata=np.column_stack((
                        liquidity_zones['Zone Bottom'],
                        liquidity_zones['Zone Top'],
                        liquidity_zones['Flow Bias'],
                    )),
                    hovertemplate=(
                        'Zone: %{customdata[0]:.2f} - %{customdata[1]:.2f}<br>'
                        'Score: %{x:.1f}<br>Flow: %{customdata[2]}<extra></extra>'
                    ),
                    name='Liquidity Score',
                ))
                liquidity_fig.add_hline(
                    y=latest_liquidity_price,
                    line_dash='dash',
                    line_color='#f8fafc',
                    annotation_text='Last Price',
                )
                liquidity_fig.update_layout(
                    title=f'{liquidity_asset} Top Liquidity Zones ({liquidity_timeframe})',
                    xaxis_title='Liquidity Score (0-100)',
                    yaxis_title='Price',
                    template='plotly_dark',
                    xaxis=dict(range=[0, 105]),
                )
                st.plotly_chart(liquidity_fig, width='stretch')

                liquidity_display_columns = [
                    'Zone Bottom', 'Zone Top', 'Liquidity Score',
                    'Location vs Price', 'Flow Bias', 'Order Flow Imbalance',
                    'Volume Z-Score', 'ATR Compression Z', 'Swing Density Z',
                    'Time at Price Z', 'VWAP Proximity Z',
                ]
                liquidity_styler = liquidity_zones[liquidity_display_columns].style.format({
                    'Zone Bottom': '{:.2f}',
                    'Zone Top': '{:.2f}',
                    'Liquidity Score': '{:.1f}',
                    'Order Flow Imbalance': '{:+.2%}',
                    'Volume Z-Score': '{:+.2f}',
                    'ATR Compression Z': '{:+.2f}',
                    'Swing Density Z': '{:+.2f}',
                    'Time at Price Z': '{:+.2f}',
                    'VWAP Proximity Z': '{:+.2f}',
                }).background_gradient(
                    subset=['Liquidity Score'], cmap='viridis'
                )
                st.dataframe(liquidity_styler, width='stretch', hide_index=True)

                with st.expander("How the Liquidity Score is calculated"):
                    st.markdown(
                        """
- **25% Volume Z-Score:** unusually concentrated traded volume in the price band.
- **20% Flow Strength:** absolute buy/sell volume imbalance; the sign is retained as Flow Bias.
- **15% ATR Compression:** preference for bands formed during volatility contraction.
- **15% Swing Density:** clustered local highs and lows, where stops may accumulate.
- **15% Time at Price:** number of candles accepted within the band.
- **10% VWAP Proximity:** preference for liquidity near volume-weighted fair value.

Each component is standardized across the displayed price bands before the weighted score is normalized to **0-100**. A high score identifies a potential liquidity pool, not a guaranteed reversal or breakout.
                        """
                    )
            else:
                st.info("Not enough price and volume history was available to map liquidity zones.")
        elif saved_liquidity_config is not None:
            st.info(
                "The Liquidity Zone asset, timeframe, or band count changed. "
                "Run the analysis again to refresh the results."
            )

        # --- Volume Delta for US Indices and Macro Assets ---
        st.divider()
        st.subheader("Volume Delta: SPY, QQQ, NQ=F, DIA, YM=F, XAU/USD, and FTSE")
        st.caption("Volume Delta = volume on up candles minus volume on down candles. Positive values indicate net buying pressure; negative values indicate net selling pressure.")
        macro_delta_timeframe = st.selectbox(
            "Volume Delta timeframe",
            ['5m', '15m', '1h', '4h'],
            index=2,
            key='macro_volume_delta_timeframe'
        )
        macro_delta_assets = {
            'SPY': 'SPY',
            'QQQ': 'QQQ',
            'NQ=F': 'NQ=F',
            'DIA': 'DIA',
            'YM=F': 'YM=F',
            'XAU/USD': 'XAUUSD',
            'FTSE': '^FTSE'
        }
        macro_delta_config = (tuple(macro_delta_assets.items()), macro_delta_timeframe)

        if st.button("Refresh Index & Macro Volume Delta", key='refresh_macro_volume_delta'):
            macro_delta_rows = []
            macro_delta_history = {}
            with st.spinner("Calculating volume delta for SPY, QQQ, NQ=F, DIA, YM=F, XAU/USD, and FTSE..."):
                for display_symbol, data_symbol in macro_delta_assets.items():
                    delta_df = fetch_and_analyze(data_symbol, timeframe=macro_delta_timeframe, silent=True, limit=500)
                    if delta_df is None or delta_df.empty:
                        continue
                    delta_df = delta_df.copy()
                    delta_df['Buy Volume'] = delta_df['volume'].where(delta_df['close'] >= delta_df['open'], 0)
                    delta_df['Sell Volume'] = delta_df['volume'].where(delta_df['close'] < delta_df['open'], 0)
                    delta_df['Volume Delta'] = delta_df['Buy Volume'] - delta_df['Sell Volume']
                    delta_df['Cumulative Volume Delta'] = delta_df['Volume Delta'].cumsum()
                    delta_df['20-Candle Volume Delta'] = delta_df['Volume Delta'].rolling(20).sum()
                    macro_delta_history[display_symbol] = delta_df
                    macro_delta_rows.append({
                        'Asset': display_symbol,
                        'Latest Volume Delta': float(delta_df['Volume Delta'].iloc[-1]),
                        '20-Candle Volume Delta': float(delta_df['20-Candle Volume Delta'].iloc[-1]),
                        'Cumulative Volume Delta': float(delta_df['Cumulative Volume Delta'].iloc[-1])
                    })
            st.session_state['macro_volume_delta_summary'] = pd.DataFrame(macro_delta_rows)
            st.session_state['macro_volume_delta_history'] = macro_delta_history
            st.session_state['macro_volume_delta_config'] = macro_delta_config

        if st.session_state.get('macro_volume_delta_config') == macro_delta_config:
            macro_delta_summary = st.session_state.get('macro_volume_delta_summary', pd.DataFrame()).copy()
            macro_delta_history = st.session_state.get('macro_volume_delta_history', {})
            if not macro_delta_summary.empty:
                st.dataframe(macro_delta_summary.style.format({
                    'Latest Volume Delta': format_large_number,
                    '20-Candle Volume Delta': format_large_number,
                    'Cumulative Volume Delta': format_large_number
                }), width='stretch')
                macro_delta_asset = st.selectbox(
                    "Select asset for detailed Volume Delta",
                    macro_delta_summary['Asset'].tolist(),
                    key='macro_volume_delta_asset'
                )
                selected_macro_delta = macro_delta_history[macro_delta_asset]
                macro_delta_metrics = st.columns(3)
                macro_delta_metrics[0].metric("Latest Volume Delta", format_large_number(float(selected_macro_delta['Volume Delta'].iloc[-1])))
                macro_delta_metrics[1].metric("20-Candle Volume Delta", format_large_number(float(selected_macro_delta['20-Candle Volume Delta'].iloc[-1])))
                macro_delta_metrics[2].metric("Cumulative Volume Delta", format_large_number(float(selected_macro_delta['Cumulative Volume Delta'].iloc[-1])))
                st.line_chart(selected_macro_delta['Cumulative Volume Delta'])
                st.dataframe(selected_macro_delta[['Buy Volume', 'Sell Volume', 'Volume Delta', 'Cumulative Volume Delta']].tail(50).style.format({
                    'Buy Volume': format_large_number,
                    'Sell Volume': format_large_number,
                    'Volume Delta': format_large_number,
                    'Cumulative Volume Delta': format_large_number
                }), width='stretch', height=300)
            else:
                st.info("No volume-delta data was returned. Try another timeframe.")

        # --- Key Support Levels (Put Support Proxy) ---
        st.divider()
        st.subheader("Key Support Levels (Put Support Proxy)")
        st.caption("These technical levels often act as strong support, similar to areas with high Put option open interest.")
        
        if st.button("Refresh Key Support Levels"):
            support_data = []
            with st.spinner("Fetching key support levels..."):
                for sym in indices_for_qs:
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
            st.session_state['support_levels_data'] = pd.DataFrame(support_data)

        df_support_levels = st.session_state.get('support_levels_data', pd.DataFrame()).copy()
        if not df_support_levels.empty:
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
            st.info("Click 'Refresh Key Support Levels' to load the latest support-level basket.")
            
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

            styler_stocks = styler_stocks.map(color_metrics, subset=['momentum'])

            st.dataframe(styler_stocks, width='stretch')
        else:
            st.info("Click 'Refresh Stocks Data' to load the latest metrics for Top US Stocks.")

    if False:  # Removed: Composite Derivative Backtest
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

    with tab4:
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

                    # --- Implied Volatility Expected-Move Section ---
                    st.divider()
                    st.subheader("Expected One-Day Move from Implied Volatility")

                    current_price = float(current_data['close'])
                    fallback_daily_vol = None
                    daily_closes = df_quantum['close'].resample('1D').last().dropna()
                    daily_returns = np.log(daily_closes / daily_closes.shift(1)).dropna()
                    if len(daily_returns) >= 10:
                        fallback_daily_vol = float(daily_returns.tail(20).std())

                    iv_analysis = get_implied_volatility_analysis(
                        quantum_symbol,
                        current_price,
                        fallback_daily_vol,
                    )

                    if iv_analysis:
                        iv_col1, iv_col2, iv_col3 = st.columns(3)
                        annualized_label = (
                            "Annualized Implied Volatility"
                            if iv_analysis['is_implied']
                            else "Annualized Historical Volatility"
                        )
                        iv_col1.metric(annualized_label, f"{iv_analysis['annualized_vol']:.2%}")
                        iv_col2.metric("Expected Daily Move", f"±{iv_analysis['daily_move_pct']:.2%}")
                        iv_col3.metric("Expected Dollar Move", f"±${iv_analysis['daily_move_amount']:,.2f}")

                        st.success(
                            f"At a current price of **${current_price:,.2f}**, {quantum_symbol}'s estimated "
                            f"one-day range is **${iv_analysis['lower_price']:,.2f} to "
                            f"${iv_analysis['upper_price']:,.2f}**."
                        )
                        st.caption(
                            f"Daily move: {iv_analysis['annualized_vol']:.2%} / sqrt(252) = "
                            f"{iv_analysis['daily_move_pct']:.2%}. Source: {iv_analysis['source']}"
                            + (f" (expiration {iv_analysis['expiration']})." if iv_analysis['expiration'] else ".")
                            + " Conversion rule: daily volatility x sqrt(252) = annualized volatility; "
                            + "annualized volatility / sqrt(252) = daily volatility. "
                            + "This is an approximately one-standard-deviation estimate, not a directional forecast or guaranteed range."
                        )
                    else:
                        st.warning("Implied volatility and a historical-volatility fallback were unavailable for this asset.")
                else:
                    st.warning(f"Could not generate volatility surface for {quantum_symbol}.")
            
            if auto_refresh_quantum:
                time.sleep(301) # Wait 5 minutes
                st.rerun()

    if False:  # Removed: GBP/USD Quantum Backtest
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

    with tab5:
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
                    
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.warning(f"Could not fetch data for {gamma_asset} on the {gamma_tf} timeframe.")

    if False:  # Removed: Options Gamma Exposure
        st.subheader("🛡️ Options Gamma Exposure (GEX)")
        st.info("""
        This tool analyzes real options data to calculate the total Gamma Exposure (GEX) of market makers. High GEX can suppress volatility, while certain strike levels can act as 'magnets' or 'pins' for the price.
        - **Total GEX**: The overall gamma imbalance. A large positive value suggests volatility suppression (a 'gamma trap').
        - **Zero Gamma**: The price level where market maker gamma exposure flips from positive to negative. This level can act as a pivot point.
        - **GEX Profile**: The bar chart shows which strike prices hold the most positive (call) and negative (put) gamma.
        """)

        gex_asset = st.selectbox("Select Asset (US Stocks/ETFs)", options=['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'TSLA', 'NVDA', 'AMZN'], key='gex_asset')

        @st.cache_data(ttl=600)
        def calculate_gamma_exposure(symbol, api_key, asset_type='stocks'):
            """Fetches options data from Polygon.io and calculates Gamma Exposure."""
            if not polygon_available:
                return None, "Polygon package is not installed. Install the `polygon-api-client` package or use the proxy GEX fallback."
            if not api_key:
                return None, "Polygon.io API Key is required. Please enter it in the sidebar."

            try:
                client = RESTClient(api_key)
                
                # 1. Get current price of the underlying asset
                try:
                    ticker_symbol = f"X:{symbol.replace('/USDT', 'USD')}" if asset_type == 'crypto' else symbol
                    aggs = client.get_aggs(ticker_symbol, 1, "day", (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d'))
                    current_price = aggs[-1].close
                except Exception as e:
                     return None, f"Could not fetch current price for {ticker_symbol}: {e}"

                # 2. Fetch all options contracts for the underlying (we'll filter by expiration later)
                # Scan expirations within the next 60 days for relevance
                today = datetime.now().strftime('%Y-%m-%d')
                sixty_days_from_now = (datetime.now() + timedelta(days=60)).strftime('%Y-%m-%d')
                
                contracts = []
                for contract in client.list_options_contracts(underlying_ticker=ticker_symbol, expiration_date_gte=today, expiration_date_lte=sixty_days_from_now, limit=1000):
                    contracts.append(contract)

                if not contracts:
                    return {"current_price": current_price}, "No options contracts found for this asset in the next 60 days."

                # 3. Get a snapshot of all options to get greeks and OI
                snapshot = client.get_options_snapshot(symbol)

                options_data = []
                for option in snapshot:
                    # Ensure the option has greeks and open interest data
                    if option.greeks and option.open_interest and option.details:
                        options_data.append({
                            'strike': option.details.strike_price,
                            'gamma': option.greeks.gamma,
                            'openInterest': option.open_interest,
                            'type': option.details.contract_type
                        })
                
                if not options_data:
                    return {"current_price": current_price}, "GEX calculation failed: No options contracts with Greeks and Open Interest were found via Polygon API."

                df_valid = pd.DataFrame(options_data)
                
                # GEX = Gamma * Open Interest * 100 shares/contract
                # Puts have a negative impact on dealer gamma as they are short puts (long stock hedge)
                df_valid['gamma_exposure'] = df_valid['gamma'] * df_valid['openInterest'] * 100 * np.where(df_valid['type'] == 'put', -1, 1)
                
                # Group by strike to see the profile
                gex_profile = df_valid.groupby('strike')['gamma_exposure'].sum()

                total_gex = gex_profile.sum()

                # Find Zero Gamma Level (where cumulative GEX flips from negative to positive)
                cumulative_gex = gex_profile.sort_index().cumsum()
                zero_gamma_level_series = cumulative_gex[cumulative_gex > 0]
                zero_gamma_level = zero_gamma_level_series.index.min() if not zero_gamma_level_series.empty else 0

                return {
                    "total_gex": total_gex,
                    "zero_gamma_level": zero_gamma_level,
                    "gex_profile": gex_profile,
                    "current_price": current_price
                }, None
            except Exception as e:
                return None, f"An unexpected error occurred with Polygon API: {e}"

        if st.button("Analyze Gamma Exposure", key='run_gex'):
            with st.spinner(f"Fetching options chain for {gex_asset}..."):
                gex_data, error = calculate_gamma_exposure(gex_asset, polygon_api_key, asset_type='stocks')
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
                    st.plotly_chart(fig, width='stretch')

    if False:  # Removed: Crypto GEX (Polygon)
        st.subheader("🛡️ Crypto Gamma Exposure (GEX) via Binance")
        st.info("""
        This tool analyzes real options data for crypto assets to calculate the total Gamma Exposure (GEX) of market makers.
        This feature uses live data from the Binance exchange.
        - **Total GEX**: The overall gamma imbalance. A large positive value suggests volatility suppression.
        - **Zero Gamma**: The price level where market maker gamma exposure flips. This can act as a pivot point.
        - **GEX Profile**: The bar chart shows which strike prices hold the most positive (Call) and negative (Put) gamma.
        """)

        # Use session state to get available crypto symbols from the derivative scan, or fallback
        if 'df_deriv' in st.session_state and not st.session_state.df_deriv.empty:
            crypto_options_list = st.session_state.df_deriv[st.session_state.df_deriv['symbol'].str.contains('/USDT', na=False)]['symbol'].unique().tolist()
        else:
            crypto_options_list = [opt for opt in asset_options if '/USDT' in opt]

        if crypto_options_list:
            gex_asset_crypto = st.selectbox("Select Crypto Asset for GEX Analysis", options=crypto_options_list, key='gex_asset_crypto_tab11')
            gex_crypto_tf = st.selectbox("Select Timeframe", options=['5m', '15m', '1h', '4h'], index=2, key='gex_crypto_tf_tab11')

            if st.button("Analyze Crypto Gamma Exposure", key='run_gex_crypto_tab11'):
                with st.spinner(f"Fetching options chain for {gex_asset_crypto} from Binance..."):
                    try:
                        gex_data, error = calculate_gamma_exposure_binance(gex_asset_crypto, timeframe=gex_crypto_tf)
                        if error and gex_data and 'current_price' in gex_data:
                            st.metric("Current Price", f"${gex_data.get('current_price', 0):,.2f}")
                            st.error(f"Could not calculate GEX: {error}")
                        elif gex_data and 'gex_profile' in gex_data:
                            if gex_data.get('proxy_used'):
                                st.info(gex_data.get('proxy_reason', 'Binance options chains are unavailable, so a proxy GEX profile is shown.'))

                            st.metric("Total GEX (Notional)", f"{gex_data['total_gex']:,.0f}")
                            st.metric("Zero Gamma Level", f"${gex_data['zero_gamma_level']:.2f}")
                            st.metric("Current Price", f"${gex_data['current_price']:.2f}")

                            gex_profile_df = gex_data['gex_profile'].reset_index()
                            gex_profile_df.columns = ['strike', 'gamma_exposure']

                            fig = go.Figure()
                            fig.add_trace(go.Bar(x=gex_profile_df['strike'], y=gex_profile_df['gamma_exposure'], name='Gamma Exposure'))
                            fig.add_vline(x=gex_data['zero_gamma_level'], line_width=2, line_dash="dash", line_color="yellow", annotation_text="Zero Gamma", annotation_position="top right")
                            fig.add_vline(x=gex_data['current_price'], line_width=2, line_dash="solid", line_color="cyan", annotation_text="Current Price", annotation_position="top left")
                            fig.update_layout(title=f'Gamma Exposure Profile for {gex_asset_crypto}', xaxis_title='Strike Price', yaxis_title='Gamma Exposure (Notional)', template='plotly_dark')
                            st.plotly_chart(fig, use_container_width=True)
                        elif error:
                            st.error(f"Could not calculate GEX: {error}")
                        else:
                            st.warning("No GEX data was returned. The asset may not have an options market on Binance or there was an API issue.")
                    except Exception as e:
                        st.error(f"An error occurred during analysis: {e}")
        else:
            st.warning("Run a derivative scan in the 'Derivatives Trend Scan' tab first to populate the list of available crypto assets.")

    with tab6:
        st.subheader("🔥 Derivative Crypto Analysis")
        st.write("Scan top Binance derivatives, contextualized with real-time ETF order flow from major US markets.")

        # --- 1. Derivative Scan Section (Copied from Tab 3) ---
        st.divider()
        st.subheader("Derivative Asset Scan")
        c1_deriv, c2_deriv, c3_deriv = st.columns(3)
        with c1_deriv:
            timeframe_deriv_new = st.selectbox("Select timeframe", ["5m", "15m", "1h", "4h"], index=2, key="tf_deriv_new")
        with c2_deriv:
            flow_timeframe_new = st.selectbox("Inflow/Outflow timeframe", ["5m", "15m", "1h", "4h"], index=2, key="flow_tf_new")
        with c3_deriv:
            volume_timeframe_new = st.selectbox("Short-term volume timeframe", ["5m", "15m", "1h", "4h"], index=2, key="vol_tf_new")

        if 'df_deriv_new' not in st.session_state:
            st.session_state.df_deriv_new = pd.DataFrame()

        if st.button("Scan Top Derivatives", key="scan_deriv_new"):
            with st.spinner("Scanning top derivative assets..."):
                df_deriv_new = scan_top_derivative_assets(timeframe=timeframe_deriv_new, flow_timeframe=flow_timeframe_new, volume_timeframe=volume_timeframe_new, top_n=20)
                if df_deriv_new is not None and not df_deriv_new.empty:
                    st.session_state.df_deriv_new = df_deriv_new
                else:
                    st.warning("No derivative asset data returned.")

        if not st.session_state.df_deriv_new.empty:
            df_deriv_new = st.session_state.df_deriv_new.copy()
            # Perform QS score calculation
            df_deriv_new['z_momentum'] = (df_deriv_new['momentum'] - df_deriv_new['momentum'].mean()) / df_deriv_new['momentum'].std()
            df_deriv_new['z_flow'] = (df_deriv_new['money_flow_signal'] - df_deriv_new['money_flow_signal'].mean()) / df_deriv_new['money_flow_signal'].std()
            df_deriv_new['z_volume'] = (df_deriv_new['vol_ratio'] - df_deriv_new['vol_ratio'].mean()) / df_deriv_new['vol_ratio'].std()
            df_deriv_new['z_volatility'] = (df_deriv_new['atr14'] - df_deriv_new['atr14'].mean()) / df_deriv_new['atr14'].std()
            df_deriv_new['z_trend'] = (df_deriv_new['z_score'] - df_deriv_new['z_score'].mean()) / df_deriv_new['z_score'].std()
            df_deriv_new['qs_score'] = (
                0.35 * df_deriv_new['z_momentum'] + 0.25 * df_deriv_new['z_flow'] + 0.20 * df_deriv_new['z_volume'] -
                0.10 * df_deriv_new['z_volatility'] + 0.10 * df_deriv_new['z_trend']
            ).fillna(0)
            df_deriv_new.sort_values(by='qs_score', ascending=False, inplace=True)
            st.dataframe(df_deriv_new.style.format({"price": "${:.2f}", "momentum": "{:.2%}", "qs_score": "{:.2f}"}), width='stretch')

            # --- Historical Z-Score Component Analysis for scanned derivative assets ---
            st.divider()
            st.subheader("Historical Z-Score Component Analysis")
            st.caption("Select a scanned derivative asset, then choose a historical date to review its component Z-scores and downward-move events.")
            deriv_z_symbols = df_deriv_new['symbol'].dropna().tolist()
            deriv_z_asset = st.selectbox("Select scanned asset for Z-Score analysis", deriv_z_symbols, key='deriv_z_chart_asset')
            deriv_z_timeframe = st.selectbox(
                "Select timeframe for derivative historical components",
                ['5m', '15m', '1h', '4h', '12h', '1d'],
                index=2,
                key='deriv_z_chart_timeframe'
            )
            deriv_z_config = (deriv_z_asset, deriv_z_timeframe)

            if st.button(f"Generate Derivative Historical Z-Chart for {deriv_z_asset}", key='generate_deriv_z_chart'):
                with st.spinner(f"Calculating historical Z-scores for {deriv_z_asset}..."):
                    deriv_z_history = fetch_and_analyze(deriv_z_asset, timeframe=deriv_z_timeframe, silent=True)
                if deriv_z_history is not None and not deriv_z_history.empty:
                    st.session_state['deriv_historical_z_data'] = deriv_z_history
                    st.session_state['deriv_historical_z_config'] = deriv_z_config
                else:
                    st.error(f"Could not fetch data to generate the Z-score chart for {deriv_z_asset}.")

            if st.session_state.get('deriv_historical_z_config') == deriv_z_config:
                deriv_z_history = st.session_state['deriv_historical_z_data'].copy()
                deriv_available_dates = sorted(np.unique(pd.DatetimeIndex(deriv_z_history.index).date))
                if deriv_available_dates:
                    with st.popover("Select derivative analysis date"):
                        deriv_z_date = st.date_input(
                            "Date to analyze",
                            value=deriv_available_dates[-1],
                            min_value=deriv_available_dates[0],
                            max_value=deriv_available_dates[-1],
                            key=f"deriv_z_analysis_date_{deriv_z_asset}_{deriv_z_timeframe}"
                        )
                    st.caption(f"Showing component analysis for {deriv_z_date:%B %d, %Y}.")

                    deriv_z_history['is_up'] = deriv_z_history['close'] >= deriv_z_history['open']
                    deriv_inflow = deriv_z_history['volume'].where(deriv_z_history['is_up'], 0)
                    deriv_outflow = deriv_z_history['volume'].where(~deriv_z_history['is_up'], 0)
                    deriv_inflow = deriv_inflow.rolling(window=20).sum()
                    deriv_outflow = deriv_outflow.rolling(window=20).sum()
                    deriv_z_history['money_flow_signal'] = (deriv_inflow - deriv_outflow) / (deriv_inflow + deriv_outflow)

                    def deriv_rolling_zscore(series, window=50):
                        return (series - series.rolling(window).mean()) / series.rolling(window).std()

                    deriv_z_df = pd.DataFrame(index=deriv_z_history.index)
                    deriv_z_df['Momentum (z)'] = deriv_rolling_zscore(deriv_z_history['momentum'])
                    deriv_z_df['Flow (z)'] = deriv_rolling_zscore(deriv_z_history['money_flow_signal'])
                    deriv_z_df['Volatility (z)'] = deriv_rolling_zscore(deriv_z_history['atr14'])
                    deriv_z_df['Trend (z)'] = deriv_rolling_zscore(deriv_z_history['z_score'])
                    deriv_z_df = deriv_z_df.loc[pd.DatetimeIndex(deriv_z_df.index).date == deriv_z_date]

                    if deriv_z_df.empty:
                        st.info("No historical Z-score data is available for the selected date.")
                    else:
                        deriv_fig_z = go.Figure()
                        for column in deriv_z_df.columns:
                            deriv_fig_z.add_trace(go.Scatter(x=deriv_z_df.index, y=deriv_z_df[column], mode='lines', name=column))
                        deriv_fig_z.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey")
                        deriv_fig_z.update_layout(
                            title=f'Historical Z-Score Components for {deriv_z_asset} — {deriv_z_date:%Y-%m-%d}',
                            yaxis_title='Z-Score (Standard Deviations from Mean)',
                            template='plotly_dark'
                        )
                        st.plotly_chart(deriv_fig_z, width='stretch')

                        deriv_drop_events = []
                        for column in deriv_z_df.columns:
                            series = deriv_z_df[column].dropna()
                            if len(series) < 2:
                                continue
                            for idx in series.diff().dropna()[lambda changes: changes < 0].index:
                                previous_value = float(series.shift(1).loc[idx])
                                current_value = float(series.loc[idx])
                                deriv_drop_events.append({
                                    'Component': column,
                                    'Drop Time': pd.Timestamp(idx),
                                    'Previous Value': previous_value,
                                    'Current Value': current_value,
                                    'Drop Size': previous_value - current_value
                                })

                        if deriv_drop_events:
                            deriv_drops_df = pd.DataFrame(deriv_drop_events).sort_values('Drop Time', ascending=False).head(50)
                            deriv_drops_df['Drop Time'] = deriv_drops_df['Drop Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                            st.subheader("Drop Events for Selected Date (up to 50)")
                            st.dataframe(deriv_drops_df[['Component', 'Drop Time', 'Previous Value', 'Current Value', 'Drop Size']].style.format({
                                'Previous Value': '{:.3f}',
                                'Current Value': '{:.3f}',
                                'Drop Size': '{:.3f}'
                            }), width='stretch')
                        else:
                            st.info("No downward moves were detected on the selected date.")

                        # Volume Delta uses the same historical data, timeframe, and selected date as the Z-score analysis.
                        deriv_z_history['Buy Volume'] = deriv_z_history['volume'].where(deriv_z_history['is_up'], 0)
                        deriv_z_history['Sell Volume'] = deriv_z_history['volume'].where(~deriv_z_history['is_up'], 0)
                        deriv_z_history['Volume Delta'] = deriv_z_history['Buy Volume'] - deriv_z_history['Sell Volume']
                        deriv_z_history['Cumulative Volume Delta'] = deriv_z_history['Volume Delta'].cumsum()
                        deriv_selected_delta = deriv_z_history.loc[
                            pd.DatetimeIndex(deriv_z_history.index).date == deriv_z_date
                        ]

                        if not deriv_selected_delta.empty:
                            st.subheader("Volume Delta for Selected Date")
                            st.caption("Volume Delta = volume on up candles minus volume on down candles. Positive values indicate net buying pressure; negative values indicate net selling pressure.")
                            deriv_delta_metrics = st.columns(3)
                            deriv_delta_metrics[0].metric(
                                "Selected-Date Volume Delta",
                                format_large_number(float(deriv_selected_delta['Volume Delta'].sum()))
                            )
                            deriv_delta_metrics[1].metric(
                                "Latest Volume Delta",
                                format_large_number(float(deriv_selected_delta['Volume Delta'].iloc[-1]))
                            )
                            deriv_delta_metrics[2].metric(
                                "Cumulative Volume Delta",
                                format_large_number(float(deriv_selected_delta['Cumulative Volume Delta'].iloc[-1]))
                            )
                            st.line_chart(deriv_selected_delta['Cumulative Volume Delta'])
                            st.dataframe(deriv_selected_delta[['Buy Volume', 'Sell Volume', 'Volume Delta', 'Cumulative Volume Delta']].style.format({
                                'Buy Volume': format_large_number,
                                'Sell Volume': format_large_number,
                                'Volume Delta': format_large_number,
                                'Cumulative Volume Delta': format_large_number
                            }), width='stretch', height=300)

        # --- 2. Real-Time ETF Order Flow Section ---
        st.divider()
        st.subheader("📊 Real-Time ETF Order Flow (Macro Context)")
        st.caption("Monitor buying vs. selling pressure in major US ETFs to gauge overall market sentiment.")
        c1_etf, c2_etf = st.columns(2)
        with c1_etf:
            etf_flow_tf = st.selectbox("Select ETF Order Flow Timeframe", options=['5m', '15m', '1h', '4h', '1d'], index=2, key="etf_flow_tf_new")
        with c2_etf:
            etf_flow_candles = st.number_input("Number of Candles to Analyze", min_value=50, max_value=1000, value=200, step=50, key="etf_flow_candles_new")

        if 'etf_order_flow_history' not in st.session_state:
            st.session_state.etf_order_flow_history = {}

        if st.button("Refresh ETF Order Flow", key="refresh_etf_flow_new"):
            with st.spinner(f"Calculating ETF order flow for {etf_flow_tf} timeframe..."):
                etf_indices = ['SPY', 'QQQ', 'DIA', 'IWM'] # Top ETFs
                etf_order_flow_data = []
                st.session_state.etf_order_flow_history.clear()
                for sym in etf_indices:
                    df_flow_calc = fetch_and_analyze(sym, timeframe=etf_flow_tf, silent=True, limit=etf_flow_candles)
                    if df_flow_calc is not None and not df_flow_calc.empty:
                        df_flow_calc['is_up'] = df_flow_calc['close'] >= df_flow_calc['open']
                        inflow = float(df_flow_calc.loc[df_flow_calc['is_up'], 'volume'].sum())
                        outflow = float(df_flow_calc.loc[~df_flow_calc['is_up'], 'volume'].sum())
                        etf_order_flow_data.append({
                            'Symbol': sym,
                            'Inflow': inflow,
                            'Outflow': outflow,
                            'Net Flow': inflow - outflow,
                            'Money Flow Signal': (inflow - outflow) / (inflow + outflow) if (inflow + outflow) > 0 else 0
                        })
                        st.session_state.etf_order_flow_history[sym] = df_flow_calc.copy()
                st.session_state.etf_order_flow_summary = etf_order_flow_data

        if 'etf_order_flow_summary' in st.session_state and st.session_state.etf_order_flow_summary:
            df_etf_flow = pd.DataFrame(st.session_state.etf_order_flow_summary)
            styler_etf_flow = df_etf_flow.style.format({
                'Inflow': format_large_number, 'Outflow': format_large_number, 'Net Flow': format_large_number, 'Money Flow Signal': '{:.2f}'
            }).background_gradient(subset=['Net Flow'], cmap='RdYlGn').bar(subset=['Money Flow Signal'], align='zero', color=['#d65f5f', '#5fba7d'])
            st.dataframe(styler_etf_flow, width='stretch')

        # --- 3. Historical Inflow/Outflow Section ---
        st.divider()
        st.subheader("📜 Historical Inflow/Outflow Analysis")
        st.caption("Select an asset from the scans above to see its detailed historical order flow data and cumulative flow chart.")

        # Combine symbols from both scans for the dropdown
        crypto_symbols = st.session_state.df_deriv_new['symbol'].tolist() if not st.session_state.df_deriv_new.empty else []
        etf_symbols = list(st.session_state.etf_order_flow_history.keys())
        all_symbols = etf_symbols + crypto_symbols

        if all_symbols:
            history_asset_new = st.selectbox("Select Asset for Historical View", options=all_symbols, key="history_asset_new")

            # Determine if the selected asset is an ETF or Crypto and get its data
            history_df = None
            if history_asset_new in etf_symbols:
                history_df = st.session_state.etf_order_flow_history.get(history_asset_new)
            elif history_asset_new in crypto_symbols:
                # For crypto, we need to fetch and calculate on the fly or store it during the scan
                # For simplicity and responsiveness, we'll fetch it here.
                with st.spinner(f"Fetching historical flow for {history_asset_new}..."):
                    # Use the timeframes selected for the crypto scan
                    history_df = fetch_and_analyze(history_asset_new, timeframe=flow_timeframe_new, silent=True, limit=500)

            if history_df is not None and not history_df.empty:
                # Calculate cumulative flow for the selected asset
                history_df['net_flow_per_candle'] = history_df['volume'] * np.where(history_df['close'] >= history_df['open'], 1, -1)
                history_df['cumulative_net_flow'] = history_df['net_flow_per_candle'].cumsum()

                st.subheader(f"Cumulative Net Flow for {history_asset_new}")
                st.line_chart(history_df['cumulative_net_flow'])

                st.subheader(f"Historical Data Table for {history_asset_new} (Last 50 Candles)")
                st.dataframe(history_df.tail(50), width='stretch', height=300)
            else:
                st.info(f"No historical data available for {history_asset_new}. Please run the appropriate scan.")
        else:
            st.info("Run a derivative or ETF scan to populate the asset list for historical analysis.")

    with tab7:
        st.subheader("🧪 SPY, QQQ & XAUUSD Flow/Momentum Z Backtest")
        st.write(
            "Backtests the Historical Z-Score logic from the US Indices tab on fixed "
            "1-hour candles. A trade requires an above-average Flow Z move confirmed "
            "by Momentum Z in the same direction."
        )
        st.info(
            "**Entry:** next candle open after confirmation. "
            "**Stop:** 1 ATR (1R). **Take profit:** 1.5 ATR (1.5R). "
            "If stop and target are both touched in one candle, the stop is counted first."
        )

        earliest_zbt_date = (datetime.now() - timedelta(days=59)).date()
        latest_zbt_date = datetime.now().date()
        default_zbt_start = (datetime.now() - timedelta(days=30)).date()

        zbt_col1, zbt_col2, zbt_col3, zbt_col4 = st.columns(4)
        with zbt_col1:
            zbt_asset = st.selectbox(
                "Select asset",
                options=['SPY', 'QQQ', 'XAUUSD'],
                key="zbt_asset",
            )
        with zbt_col2:
            zbt_start_date = st.date_input(
                "Backtest start date",
                value=default_zbt_start,
                min_value=earliest_zbt_date,
                max_value=latest_zbt_date,
                key="zbt_start_date",
            )
        with zbt_col3:
            zbt_end_date = st.date_input(
                "Backtest end date",
                value=latest_zbt_date,
                min_value=earliest_zbt_date,
                max_value=latest_zbt_date,
                key="zbt_end_date",
            )
        with zbt_col4:
            zbt_atr_risk = st.number_input(
                "Stop distance (ATR multiple)",
                min_value=0.25,
                max_value=5.0,
                value=1.0,
                step=0.25,
                key="zbt_atr_risk",
            )

        if st.button(f"Run {zbt_asset} Z Backtest", key="run_index_gold_zbt"):
            if zbt_start_date > zbt_end_date:
                st.error("The backtest start date must be on or before the end date.")
            else:
                # Fetch earlier candles for the 50-bar Z-score and 20-bar flow
                # warm-up, while restricting actual entries to the selected dates.
                warmup_start = max(
                    earliest_zbt_date,
                    zbt_start_date - timedelta(days=14),
                )
                end_exclusive = zbt_end_date + timedelta(days=1)

                with st.spinner(
                    f"Running {zbt_asset} 1-hour Z-score backtest..."
                ):
                    asset_df = fetch_and_analyze(
                        zbt_asset,
                        timeframe='1h',
                        start_date=warmup_start.strftime('%Y-%m-%d'),
                        end_date=end_exclusive.strftime('%Y-%m-%d'),
                        silent=True,
                    )
                    stats, trades_df, signal_df = backtest_flow_momentum_z_strategy(
                        asset_df,
                        zbt_asset,
                        reward_risk=1.5,
                        atr_risk=float(zbt_atr_risk),
                        trade_start=zbt_start_date,
                        trade_end=end_exclusive,
                    )

                st.session_state['index_gold_zbt_summary'] = (
                    pd.DataFrame([stats]) if stats is not None else pd.DataFrame()
                )
                st.session_state['index_gold_zbt_trades'] = trades_df
                st.session_state['index_gold_zbt_signals'] = (
                    {zbt_asset: signal_df} if not signal_df.empty else {}
                )
                st.session_state['index_gold_zbt_config'] = {
                    'asset': zbt_asset,
                    'start_date': zbt_start_date,
                    'end_date': zbt_end_date,
                    'atr_risk': zbt_atr_risk,
                }

        zbt_summary = st.session_state.get('index_gold_zbt_summary', pd.DataFrame())
        if not zbt_summary.empty:
            completed_config = st.session_state.get('index_gold_zbt_config', {})
            completed_asset = completed_config.get('asset', zbt_summary.iloc[0]['Symbol'])
            completed_start = completed_config.get('start_date')
            completed_end = completed_config.get('end_date')
            st.subheader(f"{completed_asset} Backtest Results")
            if completed_start and completed_end:
                st.caption(
                    f"Selected period: {completed_start:%B %d, %Y} through "
                    f"{completed_end:%B %d, %Y} (1-hour timeframe)."
                )
            summary_styler = zbt_summary.style.format({
                'Win Rate %': '{:.1f}%',
                'Net R': '{:+.2f}R',
                'Average R': '{:+.2f}R',
                'Profit Factor': lambda value: '∞' if np.isinf(value) else f'{value:.2f}',
            }).map(color_metrics, subset=['Net R', 'Average R'])
            st.dataframe(summary_styler, width='stretch', hide_index=True)

            zbt_trades = st.session_state.get('index_gold_zbt_trades', pd.DataFrame())
            if not zbt_trades.empty:
                total_trades = len(zbt_trades)
                combined_net_r = float(zbt_trades['R Multiple'].sum())
                combined_win_rate = float((zbt_trades['R Multiple'] > 0).mean() * 100)
                metric1, metric2, metric3 = st.columns(3)
                metric1.metric("Total Trades", total_trades)
                metric2.metric("Win Rate", f"{combined_win_rate:.1f}%")
                metric3.metric("Net Result", f"{combined_net_r:+.2f}R")

                st.subheader("Trade Log")
                trade_columns = [
                    'Symbol', 'Direction', 'Signal Time', 'Entry Time', 'Entry Price',
                    'Stop Price', 'Target Price', 'Exit Time', 'Exit Price',
                    'Exit Reason', 'R Multiple', 'Return %', 'Flow Z',
                    'Flow Z Change', 'Flow Move Size', 'Flow Move Size Change',
                    'Momentum Z', 'Momentum Z Change',
                ]
                trade_styler = zbt_trades[trade_columns].sort_values(
                    'Entry Time', ascending=False
                ).style.format({
                    'Entry Price': '{:.2f}',
                    'Stop Price': '{:.2f}',
                    'Target Price': '{:.2f}',
                    'Exit Price': '{:.2f}',
                    'R Multiple': '{:+.2f}R',
                    'Return %': '{:+.2f}%',
                    'Flow Z': '{:.2f}',
                    'Flow Z Change': '{:+.2f}',
                    'Flow Move Size': '{:.2f}',
                    'Flow Move Size Change': '{:+.2f}',
                    'Momentum Z': '{:.2f}',
                    'Momentum Z Change': '{:+.2f}',
                }).map(color_metrics, subset=['R Multiple', 'Return %'])
                st.dataframe(trade_styler, width='stretch', hide_index=True)
            else:
                st.info("No confirmed trades occurred during this lookback period.")

            signal_assets = list(
                st.session_state.get('index_gold_zbt_signals', {}).keys()
            )
            if signal_assets:
                selected_signal_asset = st.selectbox(
                    "Inspect Flow Z and Momentum Z signals",
                    signal_assets,
                    key="zbt_signal_asset",
                )
                signal_df = st.session_state['index_gold_zbt_signals'][
                    selected_signal_asset
                ]
                signal_chart = signal_df[
                    ['Flow Z', 'Momentum Z', 'Large Flow Threshold']
                ].dropna().tail(300)
                st.line_chart(signal_chart)
        else:
            st.caption("Select one asset and date range, then run its backtest.")

    with tab8:
        st.subheader("NAS100 Backtest from Daily VIX Z-Score Alerts")
        st.write(
            "Backtests NAS100 futures (Yahoo Finance ticker NQ=F) using confirmed VIX "
            "and Momentum Z alerts. Select the entry logic below."
        )
        st.info(
            "VIX and NAS100 alerts are calculated from completed 1-hour candles. "
            "Entry uses the NAS100 5-minute candle close available at the exact alert time. "
            "The stop is based on 5-minute NAS100 ATR(14). Take profit is the nearest "
            "1-hour liquidity zone in the trade direction, calculated without future data. "
            "Stop/target evaluation uses subsequent 5-minute candles. If both are touched "
            "in one candle, the stop is counted first."
        )

        nas_vix_earliest = (datetime.now() - timedelta(days=59)).date()
        nas_vix_latest = datetime.now().date()
        nas_vix_default_start = (datetime.now() - timedelta(days=30)).date()
        nas_vix_col1, nas_vix_col2, nas_vix_col3, nas_vix_col4, nas_vix_col5 = st.columns(5)
        with nas_vix_col1:
            st.metric("VIX alert timeframe", "1 hour")
        with nas_vix_col2:
            st.metric("NAS100 entry & exits", "5 minutes")
        with nas_vix_col3:
            nas_vix_start = st.date_input(
                "Start date",
                value=nas_vix_default_start,
                min_value=nas_vix_earliest,
                max_value=nas_vix_latest,
                key='nas_vix_backtest_start',
            )
        with nas_vix_col4:
            nas_vix_end = st.date_input(
                "End date",
                value=nas_vix_latest,
                min_value=nas_vix_earliest,
                max_value=nas_vix_latest,
                key='nas_vix_backtest_end',
            )
        with nas_vix_col5:
            nas_vix_atr_risk = st.number_input(
                "Stop distance (ATR multiple)",
                min_value=0.25,
                max_value=5.0,
                value=1.0,
                step=0.25,
                key='nas_vix_backtest_atr_risk',
            )

        nas_logic_col1, nas_logic_col2 = st.columns(2)
        with nas_logic_col1:
            nas_vix_logic = st.selectbox(
                "Backtest logic",
                options=['logic_1', 'logic_2', 'logic_3'],
                format_func=lambda logic: (
                    "Logic 1 — Trade opposite the VIX alert"
                    if logic == 'logic_1'
                    else "Logic 2 — Wait for matching NAS100 alert"
                    if logic == 'logic_2'
                    else "Logic 3 — Z-component reversal"
                ),
                key='nas_vix_backtest_logic',
            )
        with nas_logic_col2:
            nas_vix_threshold = float(st.session_state.get(
                'daily_ghana_zscore_alert_threshold', 1.0
            ))
            st.metric("Stored alert threshold", f"±{nas_vix_threshold:.2f}")
        if nas_vix_logic == 'logic_1':
            st.caption(
                "Logic 1: positive aligned VIX alert → SELL NAS100; "
                "negative aligned VIX alert → BUY NAS100."
            )
        elif nas_vix_logic == 'logic_2':
            st.caption(
                "Logic 2: wait for the first VIX alert, then the first later NAS100 "
                "alert with the same sign. Positive + positive → BUY; "
                "negative + negative → SELL."
            )
        else:
            st.caption(
                "Logic 3: use the stored NAS100 historical alert, then wait for "
                "the strongest/weakest Z component to reverse by the selected amount, "
                "with Momentum Z, Flow Z, and dispersion confirming the reversal."
            )

        component_drop_threshold = 0.5
        if nas_vix_logic == 'logic_3':
            component_drop_threshold = st.number_input(
                "Minimum component reversal",
                min_value=0.1,
                max_value=3.0,
                value=0.5,
                step=0.1,
                key='nas_component_drop_threshold',
            )

        stored_backtest_alerts = st.session_state.get(
            'daily_ghana_zscore_alerts', []
        )
        stored_alert_symbols = {
            alert.get('Symbol') for alert in stored_backtest_alerts
            if isinstance(alert, dict)
        }
        required_stored_symbols = (
            {'^VIX'}
            if nas_vix_logic == 'logic_1'
            else {'^VIX', 'NQ=F'}
            if nas_vix_logic == 'logic_2'
            else {'NQ=F'}
        )
        stored_alerts_ready = (
            bool(stored_backtest_alerts)
            and st.session_state.get('daily_ghana_zscore_alert_timeframe') == '1h'
            and required_stored_symbols.issubset(stored_alert_symbols)
            and all(
                'Momentum Z' in alert
                for alert in stored_backtest_alerts
                if isinstance(alert, dict)
            )
        )

        if st.button("Run NAS100 / VIX Alert Backtest", key='run_nas_vix_backtest'):
            if nas_vix_start > nas_vix_end:
                st.error("The start date must be on or before the end date.")
            elif not stored_alerts_ready:
                st.error(
                    "First open the US Indices tab, select the 1h timeframe, and click "
                    "Refresh Indices Data. The backtest requires that stored Ghana-time table."
                )
            else:
                warmup_start = max(
                    nas_vix_earliest,
                    nas_vix_start - timedelta(days=14),
                )
                end_exclusive = nas_vix_end + timedelta(days=1)
                with st.spinner("Loading the stored alert table and simulating NAS100 trades..."):
                    nas100_backtest_df = fetch_and_analyze(
                        'NQ=F',
                        timeframe='5m',
                        start_date=warmup_start.strftime('%Y-%m-%d'),
                        end_date=end_exclusive.strftime('%Y-%m-%d'),
                        silent=True,
                    )
                    nas100_liquidity_df = fetch_and_analyze(
                        'NQ=F',
                        timeframe='1h',
                        start_date=warmup_start.strftime('%Y-%m-%d'),
                        end_date=end_exclusive.strftime('%Y-%m-%d'),
                        silent=True,
                    )
                    nas_vix_summary, nas_vix_trades, nas_vix_signals = (
                        backtest_nas100_from_vix_daily_alerts(
                            nas100_backtest_df,
                            pd.DataFrame(),
                            nas100_alert_df=nas100_liquidity_df,
                            alert_timeframe='1h',
                            nas100_alert_timeframe='1h',
                            entry_timeframe='5m',
                            z_threshold=float(nas_vix_threshold),
                            atr_risk=float(nas_vix_atr_risk),
                            start_date=nas_vix_start,
                            end_date=nas_vix_end,
                            strategy_logic=nas_vix_logic,
                            stored_daily_alerts=stored_backtest_alerts,
                            component_drop_threshold=float(component_drop_threshold),
                        )
                    )
                st.session_state['nas_vix_backtest_summary'] = nas_vix_summary
                st.session_state['nas_vix_backtest_trades'] = nas_vix_trades
                st.session_state['nas_vix_backtest_signals'] = nas_vix_signals
                st.session_state['nas_vix_backtest_config'] = (
                    '1h',
                    '1h',
                    '5m',
                    nas_vix_start,
                    nas_vix_end,
                    float(nas_vix_threshold),
                    float(nas_vix_atr_risk),
                    nas_vix_logic,
                    float(component_drop_threshold),
                )

        current_nas_vix_config = (
            '1h',
            '1h',
            '5m',
            nas_vix_start,
            nas_vix_end,
            float(nas_vix_threshold),
            float(nas_vix_atr_risk),
            nas_vix_logic,
            float(component_drop_threshold),
        )
        if st.session_state.get('nas_vix_backtest_config') == current_nas_vix_config:
            nas_vix_summary = st.session_state.get('nas_vix_backtest_summary', {})
            nas_vix_trades = st.session_state.get(
                'nas_vix_backtest_trades', pd.DataFrame()
            )
            nas_vix_signals = st.session_state.get(
                'nas_vix_backtest_signals', pd.DataFrame()
            )
            if nas_vix_summary and not nas_vix_trades.empty:
                result_columns = st.columns(6)
                result_columns[0].metric("Trades", nas_vix_summary['Trades'])
                result_columns[1].metric("Win Rate", f"{nas_vix_summary['Win Rate %']:.1f}%")
                result_columns[2].metric("Net R", f"{nas_vix_summary['Net R']:+.2f}R")
                result_columns[3].metric("Average R", f"{nas_vix_summary['Average R']:+.2f}R")
                profit_factor = nas_vix_summary['Profit Factor']
                result_columns[4].metric(
                    "Profit Factor",
                    "∞" if np.isinf(profit_factor) else f"{profit_factor:.2f}",
                )
                result_columns[5].metric(
                    "Net Points", f"{nas_vix_summary['Net Points']:+,.2f}"
                )

                equity_figure = go.Figure()
                equity_figure.add_trace(go.Scatter(
                    x=nas_vix_trades['Exit Time'],
                    y=nas_vix_trades['Cumulative R'],
                    mode='lines+markers',
                    name='Cumulative R',
                ))
                equity_figure.add_hline(y=0, line_dash='dash', line_color='gray')
                equity_figure.update_layout(
                    title=f'NAS100 {nas_vix_logic.replace("_", " ").title()} Backtest Equity Curve',
                    xaxis_title='Trade Exit Time',
                    yaxis_title='Cumulative R',
                    template='plotly_dark',
                )
                st.plotly_chart(equity_figure, width='stretch')

                trade_columns = [
                    'Ghana Alert Date', 'VIX Alert Time', 'VIX Z-Score',
                    'VIX Momentum Z',
                    'Direction', 'Entry Time', 'Entry Price', 'Stop Price',
                    'Target Price', 'Target Zone Bottom', 'Target Zone Top',
                    'Target Liquidity Score', 'Planned Reward/Risk',
                    'Exit Time', 'Exit Price', 'Exit Reason',
                    'R Multiple', 'PnL Points', 'Cumulative R',
                ]
                if nas_vix_logic == 'logic_2':
                    trade_columns[4:4] = [
                        'NAS100 Alert Time', 'NAS100 Z-Score',
                        'NAS100 Momentum Z',
                    ]
                elif nas_vix_logic == 'logic_3':
                    trade_columns = [
                        'Ghana Alert Date', 'Alert Source',
                        'Historical Alert Time', 'Historical Alert Z-Score',
                        'Historical Alert Momentum Z',
                        'Highest Component Z', 'Lowest Component Z',
                        'Highest-Z Drop', 'Lowest-Z Recovery',
                        'Momentum Z Change', 'Flow Z Change',
                        'Component Dispersion', 'Dispersion Change',
                        'Direction', 'Entry Time', 'Entry Price', 'Stop Price',
                        'Target Price', 'Target Zone Bottom', 'Target Zone Top',
                        'Target Liquidity Score', 'Planned Reward/Risk',
                        'Exit Time', 'Exit Price', 'Exit Reason',
                        'R Multiple', 'PnL Points', 'Cumulative R',
                    ]
                trade_styler = nas_vix_trades[trade_columns].style.format({
                    'VIX Z-Score': '{:+.2f}',
                    'VIX Momentum Z': '{:+.2f}',
                    'Historical Alert Z-Score': '{:+.2f}',
                    'Historical Alert Momentum Z': '{:+.2f}',
                    'NAS100 Z-Score': '{:+.2f}',
                    'NAS100 Momentum Z': '{:+.2f}',
                    'Highest Component Z': '{:+.2f}',
                    'Lowest Component Z': '{:+.2f}',
                    'Highest-Z Drop': '{:+.2f}',
                    'Lowest-Z Recovery': '{:+.2f}',
                    'Momentum Z Change': '{:+.2f}',
                    'Flow Z Change': '{:+.2f}',
                    'Component Dispersion': '{:.2f}',
                    'Dispersion Change': '{:+.2f}',
                    'Entry Price': '{:.2f}',
                    'Stop Price': '{:.2f}',
                    'Target Price': '{:.2f}',
                    'Target Zone Bottom': '{:.2f}',
                    'Target Zone Top': '{:.2f}',
                    'Target Liquidity Score': '{:.1f}',
                    'Planned Reward/Risk': '{:.2f}',
                    'Exit Price': '{:.2f}',
                    'R Multiple': '{:+.2f}R',
                    'PnL Points': '{:+.2f}',
                    'Cumulative R': '{:+.2f}R',
                }).map(
                    color_metrics,
                    subset=[column for column in [
                        'VIX Z-Score', 'VIX Momentum Z', 'NAS100 Z-Score',
                        'NAS100 Momentum Z', 'R Multiple', 'Cumulative R',
                        'Highest Component Z', 'Lowest Component Z',
                        'Highest-Z Drop', 'Lowest-Z Recovery',
                        'Momentum Z Change', 'Flow Z Change',
                        'Dispersion Change',
                        'Historical Alert Z-Score',
                        'Historical Alert Momentum Z',
                    ] if column in trade_columns],
                )
                st.dataframe(trade_styler, width='stretch', hide_index=True)
            elif isinstance(nas_vix_signals, pd.DataFrame) and not nas_vix_signals.empty:
                st.warning(
                    "Qualifying alerts were found, but no NAS100 trades could be completed "
                    "with the available candles."
                )
            else:
                st.info(
                    "No alerts completed all requirements for the selected logic and period."
                )
        elif st.session_state.get('nas_vix_backtest_config') is not None:
            st.info("The settings changed. Run the backtest again to refresh the results.")

    with tab9:
        st.subheader("Dexscreener Meme-Coin Quant Analysis")
        st.write(
            "Applies the US Indices overview structure to decentralized-exchange pairs: "
            "breadth, momentum, transaction flow, relative volume, aligned Z-score alerts, "
            "and a composite signal score."
        )
        st.info(
            "Dexscreener's documented API supplies live multi-window pair statistics, not "
            "historical OHLCV candles. Live Z-scores below compare coins in the current basket. "
            "The Historical Component Analysis is kept separate and uses snapshots stored by "
            "this dashboard each time the tab refreshes."
        )

        dex_control_1, dex_control_2, dex_control_3, dex_control_4 = st.columns([1.2, 2, 1, 1])
        with dex_control_1:
            dex_discovery = st.selectbox(
                "Coin source", ['Top boosted', 'Search'], key='dex_meme_discovery'
            )
        with dex_control_2:
            dex_query = st.text_input(
                "Meme name, symbol, or contract",
                value='meme', disabled=dex_discovery != 'Search', key='dex_meme_query',
            )
        with dex_control_3:
            dex_horizon = st.selectbox(
                "Analysis window", ['m5', 'h1', 'h6', 'h24'], index=1,
                key='dex_meme_horizon',
            )
        with dex_control_4:
            dex_limit = st.selectbox(
                "Maximum pairs", [10, 15, 20, 30], index=2, key='dex_meme_limit'
            )

        if dex_discovery == 'Top boosted':
            st.caption(
                "Top boosted is a discovery feed and may contain non-meme tokens or paid "
                "promotion. Confirm the contract, liquidity, and token identity before using it."
            )
        dex_refresh_col, dex_status_col = st.columns([1, 3])
        with dex_refresh_col:
            dex_refresh = st.button(
                "Refresh Meme-Coin Tab",
                key='refresh_dex_meme',
                type='primary',
                width='stretch',
            )
        with dex_status_col:
            last_dex_refresh = st.session_state.get('dex_meme_last_refresh')
            if last_dex_refresh:
                st.caption(f"Last refreshed: {last_dex_refresh} UTC")
            else:
                st.caption("Press refresh to load the latest Dexscreener data.")
        if dex_refresh:
            fetch_dexscreener_pairs.clear()

        try:
            with st.spinner("Loading Dexscreener pairs and calculating factors..."):
                dex_pairs = fetch_dexscreener_pairs(
                    dex_discovery, dex_query, int(dex_limit)
                )
                dex_overview = build_dexscreener_overview(dex_pairs, dex_horizon)
                store_dexscreener_snapshot(dex_overview, dex_horizon)
                if dex_refresh:
                    st.session_state['dex_meme_last_refresh'] = (
                        pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S')
                    )
                    st.success("Meme-coin data refreshed from Dexscreener.")
        except Exception as exc:
            dex_overview = pd.DataFrame()
            st.error(f"Dexscreener data could not be loaded: {exc}")

        if not dex_overview.empty:
            advancing = int((dex_overview['Momentum %'] > 0).sum())
            declining = int((dex_overview['Momentum %'] < 0).sum())
            breadth_ratio = advancing / max(declining, 1)
            positive_alerts = int((dex_overview['Alert'] == 'Positive aligned').sum())
            negative_alerts = int((dex_overview['Alert'] == 'Negative aligned').sum())
            total_liquidity = dex_overview['Liquidity USD'].sum()
            dex_metrics = st.columns(6)
            dex_metrics[0].metric("Pairs", len(dex_overview))
            dex_metrics[1].metric("Advancing / Declining", f"{advancing} / {declining}")
            dex_metrics[2].metric("Breadth Ratio", f"{breadth_ratio:.2f}")
            dex_metrics[3].metric("Positive Alerts", positive_alerts)
            dex_metrics[4].metric("Negative Alerts", negative_alerts)
            dex_metrics[5].metric("Basket Liquidity", f"${total_liquidity:,.0f}")

            st.markdown("#### Meme-Coin Overview and Z-Score Alerts")
            overview_columns = [
                'Asset', 'Name', 'Chain', 'DEX', 'Price USD', 'Momentum %',
                'Buy RSI', 'Trend', 'Signal Score', 'Volume Ratio',
                'Flow Z-Score', 'Z-Score', 'Liquidity USD', 'Market Cap', 'Alert',
                'Dexscreener URL',
            ]
            dex_styler = dex_overview[overview_columns].style.format({
                'Price USD': '${:,.10g}',
                'Momentum %': '{:+.2f}%',
                'Buy RSI': '{:.1f}',
                'Signal Score': '{:+.2f}',
                'Volume Ratio': '{:.2f}x',
                'Flow Z-Score': '{:+.2f}',
                'Z-Score': '{:+.2f}',
                'Liquidity USD': '${:,.0f}',
                'Market Cap': '${:,.0f}',
            }).map(
                color_metrics,
                subset=['Momentum %', 'Signal Score', 'Flow Z-Score', 'Z-Score'],
            )
            st.dataframe(
                dex_styler,
                width='stretch',
                hide_index=True,
                column_config={
                    'Dexscreener URL': st.column_config.LinkColumn(
                        'Dexscreener', display_text='Open pair'
                    )
                },
            )
            st.caption(
                "Z-Score and Momentum Z are the cross-sectional Z-score of the selected-window "
                "price change. Flow Z standardizes buy-versus-sell transaction pressure. "
                "An alert requires Z-Score and Momentum Z to align at +1 or -1."
            )

            st.markdown(
                "#### Historical Demand & Supply Analysis — Highest Flow Z Rise and Drop"
            )
            st.caption(
                f"Every asset in the latest scan is analyzed from its stored {dex_horizon} "
                "snapshots. Event times are shown in Ghana time (GMT/UTC)."
            )
            flow_event_analysis, flow_event_coverage = (
                build_dexscreener_flow_event_analysis(dex_overview, dex_horizon)
            )
            ready_assets = (
                int(flow_event_coverage['Ready'].sum())
                if not flow_event_coverage.empty else 0
            )
            flow_history_metrics = st.columns(3)
            flow_history_metrics[0].metric("Scanned Assets", len(dex_overview))
            flow_history_metrics[1].metric("Assets Ready", ready_assets)
            flow_history_metrics[2].metric("Selected Timeframe", dex_horizon)

            if flow_event_analysis.empty:
                minimum_snapshots = (
                    int(flow_event_coverage['Stored Snapshots'].min())
                    if not flow_event_coverage.empty else 0
                )
                maximum_snapshots = (
                    int(flow_event_coverage['Stored Snapshots'].max())
                    if not flow_event_coverage.empty else 0
                )
                st.info(
                    "No asset has enough stored observations for a historical Flow Z change "
                    f"yet. Current coverage is {minimum_snapshots}–{maximum_snapshots} snapshots; "
                    "each asset needs at least 4 snapshots in this exact timeframe."
                )
            else:
                flow_event_columns = [
                    'Asset', 'Chain', 'Timeframe', 'Event', 'Event Time (Ghana)',
                    'Flow Z', 'Flow Z Change', 'Momentum Z Change', 'Trend Z Change',
                    'Volatility Z Change', 'Price Change %', 'Volatility Regime',
                    'Demand Score', 'Supply Score', 'Demand/Supply Verdict',
                    'History Confidence', 'Stored Snapshots', 'Pair Address',
                ]
                flow_event_styler = flow_event_analysis[
                    flow_event_columns
                ].sort_values(['Asset', 'Event']).style.format({
                    'Flow Z': '{:+.2f}',
                    'Flow Z Change': '{:+.2f}',
                    'Momentum Z Change': '{:+.2f}',
                    'Trend Z Change': '{:+.2f}',
                    'Volatility Z Change': '{:+.2f}',
                    'Price Change %': '{:+.3f}%',
                    'Demand Score': '{:.0f}/100',
                    'Supply Score': '{:.0f}/100',
                }).map(
                    color_metrics,
                    subset=[
                        'Flow Z', 'Flow Z Change', 'Momentum Z Change',
                        'Trend Z Change', 'Volatility Z Change', 'Price Change %',
                    ],
                )
                st.dataframe(
                    flow_event_styler,
                    width='stretch',
                    hide_index=True,
                    height=min(700, 38 * len(flow_event_analysis) + 38),
                )
                st.caption(
                    "Each asset contributes its single largest historical Flow Z drop and "
                    "single largest historical Flow Z rise within the stored data for this "
                    "timeframe. Scoring is 60% flow direction, 25% momentum direction and "
                    "15% aligned volatility expansion."
                )

            if ready_assets < len(dex_overview) and not flow_event_coverage.empty:
                with st.expander("Assets still building historical coverage"):
                    pending_coverage = flow_event_coverage[
                        ~flow_event_coverage['Ready']
                    ][['Asset', 'Chain', 'Stored Snapshots', 'Pair Address']]
                    st.dataframe(pending_coverage, width='stretch', hide_index=True)

            st.markdown("#### Historical Z-Score Component Analysis")
            asset_options = {
                f"{row['Asset']} | {row['Chain']} | {str(row['Pair Address'])[:10]}...": (
                    row['Chain'], row['Pair Address'], row['Dexscreener URL']
                )
                for _, row in dex_overview.iterrows()
            }
            selected_asset_label = st.selectbox(
                "Select a specific pair", list(asset_options), key='dex_history_asset'
            )
            selected_chain, selected_pair, selected_url = asset_options[selected_asset_label]
            if selected_url:
                st.link_button("Open selected pair on Dexscreener", selected_url)
            dex_history = load_dexscreener_component_history(
                selected_chain, selected_pair, dex_horizon
            )
            if len(dex_history) < 4:
                st.warning(
                    f"{len(dex_history)} stored snapshot(s) are available for this pair/window. "
                    "At least 4 observations are needed for provisional component events; "
                    "10 observations improve confidence and 50 provide the full lookback."
                )
                st.markdown("##### Demand and Supply Analysis — Highest Drop")
                st.info(
                    f"Waiting for historical observations: {len(dex_history)}/4 stored "
                    f"snapshots for this exact pair and {dex_horizon} window."
                )
                st.markdown("##### Demand and Supply Analysis — Highest Rise")
                st.info(
                    f"Waiting for historical observations: {len(dex_history)}/4 stored "
                    f"snapshots for this exact pair and {dex_horizon} window."
                )
                st.caption(
                    "Each click of Refresh Meme-Coin Tab stores one observation per minute. "
                    "Changing the selected pair or analysis window uses a different history."
                )
            else:
                valid_components = dex_history[[
                    'Momentum Z', 'Flow Z', 'Volatility Z', 'Trend Z'
                ]].dropna(how='all')
                if valid_components.empty:
                    st.info("The stored observations do not yet have enough price variation.")
                else:
                    component_chart = go.Figure()
                    for component in valid_components.columns:
                        component_chart.add_trace(go.Scatter(
                            x=valid_components.index,
                            y=valid_components[component],
                            mode='lines',
                            name=component,
                        ))
                    component_chart.add_hline(y=1, line_dash='dash', line_color='green')
                    component_chart.add_hline(y=-1, line_dash='dash', line_color='red')
                    component_chart.update_layout(
                        title=f"{selected_asset_label} — stored component history",
                        xaxis_title='Snapshot time (UTC)',
                        yaxis_title='Z-score',
                        template='plotly_dark',
                    )
                    st.plotly_chart(component_chart, width='stretch')

                    latest_components = valid_components.iloc[-1].dropna()
                    if not latest_components.empty:
                        highest_component = latest_components.idxmax()
                        lowest_component = latest_components.idxmin()
                        component_metrics = st.columns(4)
                        component_metrics[0].metric(
                            "Highest Component", highest_component,
                            f"{latest_components[highest_component]:+.2f} Z",
                        )
                        component_metrics[1].metric(
                            "Lowest Component", lowest_component,
                            f"{latest_components[lowest_component]:+.2f} Z",
                        )
                        component_metrics[2].metric(
                            "Component Spread",
                            f"{latest_components.max() - latest_components.min():.2f}",
                        )
                        component_metrics[3].metric(
                            "Stored Observations", len(dex_history)
                        )

                    component_moves = valid_components.diff()
                    stacked_moves = component_moves.stack().dropna()
                    if not stacked_moves.empty:
                        drop_event = stacked_moves.idxmin()
                        rise_event = stacked_moves.idxmax()
                        price_change = dex_history['Price USD'].pct_change() * 100

                        def analyze_dex_event(event_name, event_key):
                            event_time, event_component = event_key
                            flow_change = component_moves.at[event_time, 'Flow Z']
                            momentum_change = component_moves.at[event_time, 'Momentum Z']
                            volatility_change = component_moves.at[event_time, 'Volatility Z']
                            trend_change = component_moves.at[event_time, 'Trend Z']
                            event_price_change = price_change.get(event_time, np.nan)
                            volatility_z = valid_components.at[event_time, 'Volatility Z']

                            # Direction score: 60% transaction flow, 25% momentum and
                            # 15% volatility expansion in the direction of price movement.
                            demand_score = (
                                (60 if flow_change > 0 else 0)
                                + (25 if momentum_change > 0 else 0)
                                + (15 if volatility_change > 0 and event_price_change > 0 else 0)
                            )
                            supply_score = (
                                (60 if flow_change < 0 else 0)
                                + (25 if momentum_change < 0 else 0)
                                + (15 if volatility_change > 0 and event_price_change < 0 else 0)
                            )
                            if demand_score >= 60 and demand_score > supply_score:
                                strength = 'Strong' if demand_score >= 85 else 'Moderate'
                                verdict = f'{strength} demand'
                            elif supply_score >= 60 and supply_score > demand_score:
                                strength = 'Strong' if supply_score >= 85 else 'Moderate'
                                verdict = f'{strength} supply'
                            else:
                                verdict = 'Mixed / unconfirmed'

                            if volatility_z >= 1:
                                volatility_regime = 'Expansion'
                            elif volatility_z <= -1:
                                volatility_regime = 'Compression'
                            else:
                                volatility_regime = 'Normal'
                            return {
                                'Event': event_name,
                                'Time (UTC)': event_time,
                                'Dropped/Rose Component': event_component,
                                'Event Z Change': component_moves.at[event_time, event_component],
                                'Price Change %': event_price_change,
                                'Flow Z Change': flow_change,
                                'Momentum Z Change': momentum_change,
                                'Trend Z Change': trend_change,
                                'Volatility Z Change': volatility_change,
                                'Volatility Regime': volatility_regime,
                                'Demand Score': demand_score,
                                'Supply Score': supply_score,
                                'Demand/Supply Verdict': verdict,
                            }

                        drop_analysis = analyze_dex_event(
                            'Highest component drop', drop_event
                        )
                        rise_analysis = analyze_dex_event(
                            'Highest component rise', rise_event
                        )

                        st.markdown("##### Demand and Supply Analysis — Highest Drop")
                        st.dataframe(
                            pd.DataFrame([drop_analysis]).style.format({
                                'Event Z Change': '{:+.2f}',
                                'Price Change %': '{:+.3f}%',
                                'Flow Z Change': '{:+.2f}',
                                'Momentum Z Change': '{:+.2f}',
                                'Trend Z Change': '{:+.2f}',
                                'Volatility Z Change': '{:+.2f}',
                                'Demand Score': '{:.0f}/100',
                                'Supply Score': '{:.0f}/100',
                            }).map(
                                color_metrics,
                                subset=['Event Z Change', 'Price Change %', 'Flow Z Change',
                                        'Momentum Z Change', 'Trend Z Change'],
                            ),
                            width='stretch', hide_index=True,
                        )

                        st.markdown("##### Demand and Supply Analysis — Highest Rise")
                        st.dataframe(
                            pd.DataFrame([rise_analysis]).style.format({
                                'Event Z Change': '{:+.2f}',
                                'Price Change %': '{:+.3f}%',
                                'Flow Z Change': '{:+.2f}',
                                'Momentum Z Change': '{:+.2f}',
                                'Trend Z Change': '{:+.2f}',
                                'Volatility Z Change': '{:+.2f}',
                                'Demand Score': '{:.0f}/100',
                                'Supply Score': '{:.0f}/100',
                            }).map(
                                color_metrics,
                                subset=['Event Z Change', 'Price Change %', 'Flow Z Change',
                                        'Momentum Z Change', 'Trend Z Change'],
                            ),
                            width='stretch', hide_index=True,
                        )
                        st.caption(
                            "Scoring: 60% Flow Z direction + 25% Momentum Z direction + "
                            "15% volatility expansion aligned with the stored price move. A drop "
                            "is not automatically supply and a rise is not automatically demand; "
                            "the verdict depends on the factors at that exact event time."
                        )
        else:
            st.warning(
                "No eligible Dexscreener pairs were returned. Try Search with a token symbol "
                "or contract address."
            )

    if False:  # Removed: Index Spike Alert
        st.subheader("📢 Index Spike Alert Engine")
        st.info("""
        This tool provides historical data and real-time alerts for simultaneous spikes in VIX, SPY, and QQQ.
        An alert is triggered when:
        - **VIX** shows a strong move (Momentum & Z-Score are both > 1 or both < -1).
        - **AND** at the same time, either **SPY** or **QQQ** shows a strong move in the same direction (Momentum & Z-Score are both > 1 or both < -1).
        This can signal a significant market-wide risk-on or risk-off event.
        """)

        alert_symbols = ['SPY', 'QQQ', '^VIX']
        alert_timeframe = st.selectbox("Select Timeframe for Analysis", ["5m", "15m", "1h", "4h"], index=1, key="alert_tf")
        
        # Session state to store historical data and last alert time
        if 'alert_history_df' not in st.session_state:
            st.session_state.alert_history_df = pd.DataFrame()
        if 'last_alert_time' not in st.session_state:
            st.session_state.last_alert_time = None

        def fetch_alert_data(symbols, timeframe):
            """Fetches and processes data for the alert table."""
            all_stats = []
            for sym in symbols:
                df = fetch_and_analyze(sym, timeframe=timeframe, silent=True, limit=200)
                if df is not None and not df.empty:
                    # Get the last 50 records for historical view
                    for i in range(min(50, len(df))):
                        record = df.iloc[-(i+1)]
                        all_stats.append({
                            'Time': record.name,
                            'Symbol': sym,
                            'Momentum': record.get('momentum', 0.0),
                            'Z-Score': record.get('z_score', 0.0)
                        })
            if all_stats:
                history_df = pd.DataFrame(all_stats)
                history_df.sort_values(by='Time', ascending=False, inplace=True)
                return history_df
            return pd.DataFrame()

        def check_and_send_alert(df, token, chat_id):
            """Checks the latest data for alert conditions."""
            if df.empty or token == "" or chat_id == "":
                return

            latest_time = df['Time'].max()
            
            # Avoid sending multiple alerts for the same event
            if st.session_state.last_alert_time and st.session_state.last_alert_time == latest_time:
                return

            latest_data = df[df['Time'] == latest_time].set_index('Symbol')
            
            if not all(s in latest_data.index for s in alert_symbols):
                return # Not all data is ready

            vix_mom = latest_data.loc['^VIX', 'Momentum']
            vix_z = latest_data.loc['^VIX', 'Z-Score']
            spy_mom = latest_data.loc['SPY', 'Momentum']
            spy_z = latest_data.loc['SPY', 'Z-Score']
            qqq_mom = latest_data.loc['QQQ', 'Momentum']
            qqq_z = latest_data.loc['QQQ', 'Z-Score']

            # Define strong move conditions
            vix_bull = vix_mom > 1 and vix_z > 1
            vix_bear = vix_mom < -1 and vix_z < -1
            spy_bull = spy_mom > 1 and spy_z > 1
            spy_bear = spy_mom < -1 and spy_z < -1
            qqq_bull = qqq_mom > 1 and qqq_z > 1
            qqq_bear = qqq_mom < -1 and qqq_z < -1

            alert_msg = None
            if (vix_bull and spy_bull and qqq_bull):
                alert_msg = (
                    f"🚨 **Index Spike Alert (BULLISH)**\n\n"
                    f"A simultaneous bullish spike was detected across VIX, SPY, and QQQ, suggesting a strong, unified market risk-on event.\n\n"
                    f"**VIX:**\n- Momentum: `{vix_mom:.2f}`\n- Z-Score: `{vix_z:.2f}`\n\n"
                    f"**SPY:**\n- Momentum: `{spy_mom:.2f}`\n- Z-Score: `{spy_z:.2f}`\n\n"
                    f"**QQQ:**\n- Momentum: `{qqq_mom:.2f}`\n- Z-Score: `{qqq_z:.2f}`"
                )
            elif (vix_bear and spy_bear and qqq_bear):
                alert_msg = (
                    f"🚨 **Index Spike Alert (BEARISH)**\n\n"
                    f"A simultaneous bearish spike was detected in VIX and major indices, suggesting a strong risk-off event.\n\n"
                    f"**VIX:**\n- Momentum: `{vix_mom:.2f}`\n- Z-Score: `{vix_z:.2f}`\n\n"
                    f"**SPY:**\n- Momentum: `{spy_mom:.2f}`\n- Z-Score: `{spy_z:.2f}`\n\n"
                    f"**QQQ:**\n- Momentum: `{qqq_mom:.2f}`\n- Z-Score: `{qqq_z:.2f}`"
                )

            if alert_msg:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                try:
                    requests.post(url, json={"chat_id": chat_id, "text": alert_msg, "parse_mode": "Markdown"}, timeout=5)
                    st.session_state.last_alert_time = latest_time
                    st.success(f"Alert sent to Telegram at {latest_time}!")
                except Exception as e:
                    st.error(f"Telegram Error: {e}")

        auto_refresh_alert = st.checkbox("🔄 Auto-Refresh & Alert (every 5 mins)")

        if st.button("Fetch Historical Data") or auto_refresh_alert:
            with st.spinner("Fetching latest index data..."):
                history_df = fetch_alert_data(alert_symbols, alert_timeframe)
                if not history_df.empty:
                    st.session_state.alert_history_df = history_df
                    check_and_send_alert(history_df, tg_token, tg_chat_id)

        if not st.session_state.alert_history_df.empty:
            st.subheader("Historical Momentum and Z-Score Data")
            st.caption(f"Showing the last 50 data points for the {alert_timeframe} timeframe.")
            
            styler = st.session_state.alert_history_df.style.format({
                "Momentum": "{:.2f}",
                "Z-Score": "{:.2f}"
            }).map(color_metrics, subset=['Momentum', 'Z-Score'])
            st.dataframe(styler, height=400, width='stretch')

        if auto_refresh_alert:
            time.sleep(301)
            st.rerun()

if __name__ == "__main__":
     try:
         main()
     except Exception:
         import traceback
         traceback.print_exc()
         input("Press Enter to exit...")

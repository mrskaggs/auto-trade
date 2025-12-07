import os
import time
import math
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import yfinance as yf
import pandas as pd
import logging
import requests
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

load_dotenv()

logging.basicConfig(level=logging.INFO)

# Load Alpaca API credentials from environment variables (REQUIRED)
API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
TRADE_BASE_URL = os.getenv('APCA_API_BASE_URL', 'https://paper-api.alpaca.markets')

if not API_KEY or not SECRET_KEY:
    raise ValueError("APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables are required")

# Trading parameters from environment
TRADING_SYMBOL = os.getenv('TRADING_SYMBOL', 'BTCUSD')
TRADING_RISK_PCT = float(os.getenv('TRADING_RISK_PCT', '1.0'))
RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
RSI_BUY_THRESH = int(os.getenv('RSI_BUY_THRESH', '30'))
RSI_SELL_THRESH = int(os.getenv('RSI_SELL_THRESH', '70'))
EMA_SHORT = int(os.getenv('EMA_SHORT', '9'))
EMA_LONG = int(os.getenv('EMA_LONG', '21'))

# Initialize Alpaca clients
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)

def get_live_price(symbol):
    """Fetch real-time price for crypto"""
    try:
        if '/' in symbol:
            coin_map = {
                'BTC/USD': 'bitcoin',
                'ETH/USD': 'ethereum',
                'SOL/USD': 'solana',
                'ADA/USD': 'cardano',
                'DOGE/USD': 'dogecoin'
            }
            coin_id = coin_map.get(symbol)
            if coin_id:
                response = requests.get(
                    f'https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd',
                    timeout=5
                )
                data = response.json()
                if coin_id in data:
                    return float(data[coin_id]['usd'])
    except Exception as e:
        logging.error(f"[Live Price] Error: {e}")
    return None

def calculate_rsi(prices, period=14):
    """Calculate RSI using Wilder's smoothing method (traditional RSI)"""
    # Calculate price changes
    delta = prices.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # First values are simple average
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    # Wilder's smoothing (EMA with alpha = 1/period)
    for i in range(period, len(prices)):
        avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_ema(prices, span):
    """Calculate Exponential Moving Average"""
    return prices.ewm(span=span, adjust=False).mean()

def detect_ema_crossover(ema_short, ema_long):
    """Detect EMA crossover: returns 'BULLISH', 'BEARISH', or 'NONE'"""
    # Current values
    short_now = ema_short.iloc[-1]
    long_now = ema_long.iloc[-1]
    
    # Previous values
    short_prev = ema_short.iloc[-2]
    long_prev = ema_long.iloc[-2]
    
    # Bullish crossover: short was below, now above
    if short_prev <= long_prev and short_now > long_now:
        return 'BULLISH'
    
    # Bearish crossover: short was above, now below
    if short_prev >= long_prev and short_now < long_now:
        return 'BEARISH'
    
    # Check if currently in bullish or bearish state
    if short_now > long_now:
        return 'BULLISH_TREND'
    elif short_now < long_now:
        return 'BEARISH_TREND'
    
    return 'NONE'

def safe_float(value, default=0.0):
    """Safely extract float from potential Series or scalar"""
    try:
        if isinstance(value, pd.Series):
            if len(value) > 0:
                return float(value.iloc[0])
            return default
        return float(value)
    except:
        return default

class HybridStrategy:
    def __init__(self, symbol='AAPL', rsi_period=14, rsi_buy=30, rsi_sell=70, ema_short=9, ema_long=21):
        self.symbol = symbol
        self.rsi_period = rsi_period
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.last_data = None
        self.last_fetch_time = 0
        # Cache for 5 minutes (hourly data updates every hour, but check more frequently)
        self.fetch_interval = 300

    def get_symbol_for_api(self):
        """Convert symbol to API format"""
        if self.symbol.endswith('USD') and '/' not in self.symbol:
            base = self.symbol[:-3]
            return f"{base}/USD"
        return self.symbol

    def get_signal(self):
        """Get trading signal based on RSI"""
        is_crypto = self.symbol.endswith('USD')
        api_symbol = self.get_symbol_for_api()
        
        # Check if we can use cached data
        current_time = time.time()
        if self.last_data is not None and (current_time - self.last_fetch_time) < self.fetch_interval:
            bars_df = self.last_data
            logging.info(f"[Data] {self.symbol} - Using cached data ({len(bars_df)} bars)")
        else:
            # Fetch fresh data - use CoinGecko for crypto, Yahoo for stocks
            bars_df = None
            for attempt in range(3):
                try:
                    if is_crypto:
                        # Use CoinGecko API for crypto historical data (more reliable)
                        coin_map = {
                            'BTC/USD': 'bitcoin',
                            'ETH/USD': 'ethereum',
                            'SOL/USD': 'solana',
                            'ADA/USD': 'cardano',
                            'DOGE/USD': 'dogecoin'
                        }
                        coin_id = coin_map.get(api_symbol)
                        if coin_id:
                            # CoinGecko auto-returns hourly for <90 days, daily for >90 days
                            # Fetch 14 days of data (will be hourly automatically)
                            url = f'https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart'
                            params = {'vs_currency': 'usd', 'days': '14'}
                            response = requests.get(url, params=params, timeout=10)
                            
                            if response.status_code != 200:
                                logging.error(f"[Data] CoinGecko API error: {response.status_code}")
                                continue
                            
                            data = response.json()
                            
                            if 'prices' in data and len(data['prices']) > 0:
                                # Convert to DataFrame
                                prices = data['prices']
                                df = pd.DataFrame(prices, columns=['timestamp', 'Close'])
                                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                                df = df.set_index('timestamp')
                                
                                # CoinGecko returns hourly data for <90 days automatically
                                bars_df = df
                                self.last_data = bars_df
                                self.last_fetch_time = current_time
                                
                                # Determine interval from data
                                if len(df) > 1:
                                    time_diff = (df.index[1] - df.index[0]).total_seconds() / 3600
                                    interval = "hourly" if time_diff <= 1.5 else "daily"
                                else:
                                    interval = "unknown"
                                
                                logging.info(f"[Data] {self.symbol} - Fetched {len(bars_df)} {interval} bars from CoinGecko")
                                break
                            else:
                                logging.warning(f"[Data] No price data in CoinGecko response")
                    else:
                        # Use Yahoo Finance for stocks (daily bars)
                        data = yf.download(self.symbol, period='60d', interval='1d', progress=False, auto_adjust=True)
                        if not data.empty and len(data) >= self.period + 1:
                            bars_df = data.tail(30)
                            self.last_data = bars_df
                            self.last_fetch_time = current_time
                            logging.info(f"[Data] {self.symbol} - Fetched {len(bars_df)} daily bars from Yahoo Finance")
                            break
                    
                    if bars_df is None or bars_df.empty or len(bars_df) < self.rsi_period + 1:
                        logging.warning(f"[Data] Attempt {attempt+1}: Insufficient data ({len(bars_df) if bars_df is not None else 0} bars)")
                        if attempt < 2:
                            time.sleep(2)
                except Exception as e:
                    logging.error(f"[Data] Attempt {attempt+1} error: {e}")
                    if attempt < 2:
                        time.sleep(2)
            
            # If all retries failed, use cached data if available
            if bars_df is None:
                if self.last_data is not None:
                    bars_df = self.last_data
                    logging.warning(f"[Data] Using stale cached data")
                else:
                    logging.error(f"[Data] No data available")
                    return 'HOLD'

        # Extract close prices (handle both single and multi-column DataFrames)
        if 'Close' in bars_df.columns:
            close_prices = bars_df['Close'].squeeze()
        elif len(bars_df.columns) > 0:
            close_prices = bars_df.iloc[:, 3].squeeze()  # Usually 4th column is close
        else:
            logging.error("[Data] No close prices found")
            return 'HOLD'

        # Calculate indicators
        rsi_series = calculate_rsi(close_prices, period=self.rsi_period)
        ema_short_series = calculate_ema(close_prices, span=self.ema_short)
        ema_long_series = calculate_ema(close_prices, span=self.ema_long)
        
        # Get current values
        rsi_value = safe_float(rsi_series.iloc[-1], 50.0)
        ema_short_val = safe_float(ema_short_series.iloc[-1], 0.0)
        ema_long_val = safe_float(ema_long_series.iloc[-1], 0.0)
        historical_price = safe_float(close_prices.iloc[-1], 0.0)
        
        # Validate RSI
        if pd.isna(rsi_value) or rsi_value < 0 or rsi_value > 100:
            logging.warning(f"[RSI] Invalid: {rsi_value}, holding")
            return 'HOLD'

        # Detect EMA crossover
        ema_signal = detect_ema_crossover(ema_short_series, ema_long_series)

        # Get live price for crypto
        live_price = get_live_price(api_symbol) if is_crypto else None
        current_price = live_price if live_price else historical_price

        # Calculate RSI change
        prev_rsi = safe_float(rsi_series.iloc[-2], rsi_value)
        rsi_change = rsi_value - prev_rsi
        
        price_source = "(live)" if live_price else "(historical)"
        logging.info(f"[Heartbeat] {self.symbol} - Price: ${current_price:.2f} {price_source}, RSI: {rsi_value:.2f} ({rsi_change:+.2f}), EMA: {ema_signal}")

        # Get current position
        qty = 0
        try:
            positions = trading_client.get_all_positions()
            for pos in positions:
                if pos.symbol == api_symbol or pos.symbol == self.symbol:
                    qty = float(pos.qty)
                    break
        except Exception as e:
            logging.error(f"[Position] Error: {e}")

        # HYBRID TRADING LOGIC
        # BUY: RSI oversold (≤30) AND bullish EMA trend
        if rsi_value <= self.rsi_buy and ema_signal in ['BULLISH', 'BULLISH_TREND'] and qty == 0:
            logging.info(f"🎯 BUY SIGNAL: RSI {rsi_value:.2f} ≤ {self.rsi_buy} + {ema_signal}")
            return 'BUY'
        
        # SELL: RSI overbought (≥70) OR bearish EMA crossover (protect profits)
        elif (rsi_value >= self.rsi_sell or ema_signal == 'BEARISH') and qty > 0:
            reason = f"RSI {rsi_value:.2f} ≥ {self.rsi_sell}" if rsi_value >= self.rsi_sell else f"{ema_signal} crossover"
            logging.info(f"🎯 SELL SIGNAL: {reason}")
            return 'SELL'
        
        else:
            return 'HOLD'


class TradingBot:
    def __init__(self, strategy, risk_per_trade_pct=1.0):
        self.strategy = strategy
        self.risk_per_trade_pct = risk_per_trade_pct

    def print_portfolio_status(self):
        try:
            account = trading_client.get_account()
            portfolio_value = float(account.portfolio_value)
            cash_value = float(account.cash)
            positions = trading_client.get_all_positions()
            symbols = ', '.join([f"{p.symbol}({p.qty})" for p in positions]) if positions else 'None'
            logging.info(f"Portfolio: ${portfolio_value:.2f}, Cash: ${cash_value:.2f}, Positions: {symbols}")
        except Exception as e:
            logging.error(f"[Portfolio] Error: {e}")

    def run(self):
        is_crypto = self.strategy.symbol.endswith('USD')
        api_symbol = self.strategy.get_symbol_for_api()
        
        logging.info(f"=== HYBRID TRADING BOT SETTINGS ===")
        logging.info(f"Symbol: {self.strategy.symbol}")
        logging.info(f"API Symbol: {api_symbol}")
        logging.info(f"Asset Type: {'crypto' if is_crypto else 'stock'}")
        logging.info(f"Strategy: RSI + EMA Crossover (Hybrid)")
        logging.info(f"RSI Period: {self.strategy.rsi_period}")
        logging.info(f"RSI Buy: ≤{self.strategy.rsi_buy}, Sell: ≥{self.strategy.rsi_sell}")
        logging.info(f"EMA: {self.strategy.ema_short}/{self.strategy.ema_long}")
        logging.info(f"Risk per Trade: {self.risk_per_trade_pct}%")
        logging.info(f"===============================")
        logging.info(f"Starting hybrid trading bot...")
        
        self.print_portfolio_status()
        cycle_count = 0
        
        while True:
            try:
                signal = self.strategy.get_signal()
                
                if signal == 'BUY':
                    try:
                        account = trading_client.get_account()
                        portfolio_value = float(account.portfolio_value)
                        cash = float(account.cash)
                        
                        # Get current price
                        live_price = get_live_price(api_symbol)
                        if not live_price:
                            logging.warning("[BUY] No price available")
                            time.sleep(60)
                            continue
                        
                        price = float(live_price)
                        
                        # Calculate quantity based on risk
                        risk_amount = portfolio_value * self.risk_per_trade_pct / 100
                        quantity = risk_amount / price
                        
                        # Round for stocks
                        if not is_crypto:
                            quantity = math.floor(quantity)
                        
                        # Check minimum
                        if quantity < 0.001:
                            logging.info(f"[BUY] Quantity too small: {quantity:.6f}")
                            time.sleep(60)
                            continue
                        
                        # Check cash
                        cost = price * quantity
                        if cash < cost:
                            logging.warning(f"[BUY] Insufficient cash: ${cash:.2f} < ${cost:.2f}")
                            time.sleep(60)
                            continue
                        
                        # Place order
                        order_data = MarketOrderRequest(
                            symbol=api_symbol,
                            qty=quantity,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.GTC
                        )
                        trading_client.submit_order(order_data)
                        logging.info(f"✅ BUY {quantity:.4f} {self.strategy.symbol} @ ${price:.2f} (${risk_amount:.2f} risk)")
                        
                    except Exception as e:
                        logging.error(f"[BUY] Error: {e}")
                
                elif signal == 'SELL':
                    try:
                        # Get position (match both formats)
                        positions = trading_client.get_all_positions()
                        qty = 0
                        for pos in positions:
                            # Match both "ETH/USD" and "ETHUSD" formats
                            if pos.symbol == api_symbol or pos.symbol == self.strategy.symbol:
                                qty = float(pos.qty)
                                logging.info(f"[SELL] Found position: {qty} {pos.symbol}")
                                break
                        
                        if qty > 0:
                            order_data = MarketOrderRequest(
                                symbol=api_symbol,
                                qty=qty,
                                side=OrderSide.SELL,
                                time_in_force=TimeInForce.GTC
                            )
                            trading_client.submit_order(order_data)
                            logging.info(f"✅ SELL {qty:.4f} {self.strategy.symbol}")
                        
                    except Exception as e:
                        logging.error(f"[SELL] Error: {e}")
                
            except Exception as e:
                logging.error(f"[Loop] Error: {e}")
            
            cycle_count += 1
            if cycle_count % 10 == 0:
                self.print_portfolio_status()
            
            time.sleep(60)


if __name__ == "__main__":
    strategy = HybridStrategy(
        symbol=TRADING_SYMBOL,
        rsi_period=RSI_PERIOD,
        rsi_buy=RSI_BUY_THRESH,
        rsi_sell=RSI_SELL_THRESH,
        ema_short=EMA_SHORT,
        ema_long=EMA_LONG
    )
    bot = TradingBot(strategy, risk_per_trade_pct=TRADING_RISK_PCT)
    bot.run()

import streamlit as st
import os
import time
import math
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, OrderType, TimeInForce
from alpaca.data import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
import logging
import threading

load_dotenv()

# Set up logging to capture in list
logs = []
class StreamlitHandler(logging.Handler):
    def emit(self, record):
        logs.append(self.format(record))

handler = StreamlitHandler()
logging.getLogger().addHandler(handler)
logging.getLogger().setLevel(logging.INFO)

# API setup from .env
API_KEY = os.getenv('APCA_API_KEY_ID')
SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
TRADING_BASE_URL = 'https://paper-api.alpaca.markets'

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True) if API_KEY and SECRET_KEY else None
stock_data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY) if API_KEY and SECRET_KEY else None
crypto_data_client = CryptoHistoricalDataClient(API_KEY, SECRET_KEY) if API_KEY and SECRET_KEY else None

# Session state for charts
if 'rsi_data' not in st.session_state:
    st.session_state.rsi_data = [
        {'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=5), 'rsi': 25, 'buy_threshold': 30, 'sell_threshold': 70},
        {'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=4), 'rsi': 35, 'buy_threshold': 30, 'sell_threshold': 70},
        {'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=3), 'rsi': 45, 'buy_threshold': 30, 'sell_threshold': 70},
        {'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=2), 'rsi': 65, 'buy_threshold': 30, 'sell_threshold': 70},
        {'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=1), 'rsi': 75, 'buy_threshold': 30, 'sell_threshold': 70},
    ]
if 'portfolio_data' not in st.session_state:
    st.session_state.portfolio_data = [
        {'timestamp': pd.Timestamp.now() - pd.Timedelta(hours=1), 'value': 10000},
        {'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=30), 'value': 10050},
        {'timestamp': pd.Timestamp.now() - pd.Timedelta(minutes=15), 'value': 9960},
        {'timestamp': pd.Timestamp.now(), 'value': 10100},
    ]

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

class RSIStrategy:
    def __init__(self, symbol='AAPL', period=14, buy_threshold=30, sell_threshold=70):
        self.symbol = symbol
        self.period = period
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def get_signal(self):
        # Determine asset type
        is_crypto = self.symbol.endswith('USD')
        client = crypto_data_client if is_crypto else stock_data_client
        request_class = CryptoBarsRequest if is_crypto else StockBarsRequest

        # Get recent bars (last 50 1-minute bars)
        bars_request = request_class(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Minute,
            limit=50
        )
        bars = client.get_crypto_bars(bars_request) if is_crypto else client.get_stock_bars(bars_request)
        bars_df = bars.df
        if bars_df.empty or len(bars_df) < self.period:
            return 'HOLD'

        # Calculate RSI
        close_series = bars_df['close']
        rsi_series = calculate_rsi(close_series, period=self.period)
        rsi_value = rsi_series.iloc[-1]
        # Append to session state for chart (limit to last 100 points)
        st.session_state.rsi_data.append({
            'timestamp': pd.Timestamp.now(),
            'rsi': rsi_value,
            'buy_threshold': self.buy_threshold,
            'sell_threshold': self.sell_threshold
        })
        if len(st.session_state.rsi_data) > 100:
            st.session_state.rsi_data.pop(0)

        # Get current positions
        positions = trading_client.get_all_positions()
        qty = 0
        if positions:
            for pos in positions:
                if pos.symbol == self.symbol:
                    qty = float(pos.qty)
                    break

        if rsi_value <= self.buy_threshold and qty == 0:
            return 'BUY'
        elif rsi_value >= self.sell_threshold and qty > 0:
            return 'SELL'
        else:
            return 'HOLD'

class TradingBot:
    def __init__(self, strategy, risk_per_trade_pct=1.0):
        self.strategy = strategy
        self.risk_per_trade_pct = risk_per_trade_pct
        self.running = False

    def start(self):
        self.running = True
        threading.Thread(target=self.run).start()

    def stop(self):
        self.running = False

    def print_portfolio_status(self):
        try:
            account = trading_client.get_account()
            portfolio_value = float(account.portfolio_value)
            cash_value = float(account.cash)
            positions = trading_client.get_all_positions()
            symbols = ', '.join([f"{p.symbol}({p.qty})" for p in positions]) if positions else 'None'
            logging.info(f"Portfolio Value: ${portfolio_value:.2f}, Cash: ${cash_value:.2f}, Positions: {symbols}")
            # Add to portfolio chart data
            st.session_state.portfolio_data.append({
                'timestamp': pd.Timestamp.now(),
                'value': portfolio_value
            })
            if len(st.session_state.portfolio_data) > 1000:
                st.session_state.portfolio_data.pop(0)
        except Exception as e:
            logging.error(f"Error getting portfolio status: {e}")

    def run(self):
        is_crypto = self.strategy.symbol.endswith('USD')
        asset_type = "crypto" if is_crypto else "stock"
        logging.info(f"Starting real-time trading bot... (RSI-based strategy on {self.strategy.symbol}, {asset_type})")
        self.print_portfolio_status()  # Initial status
        cycle_count = 0
        while self.running and trading_client is not None:
            try:
                signal = self.strategy.get_signal()
                if signal == 'BUY':
                    # Get account
                    account = trading_client.get_account()
                    portfolio_value = float(account.portfolio_value)
                    cash = float(account.cash)
                    bars_request = (CryptoBarsRequest if is_crypto else StockBarsRequest)(
                        symbol_or_symbols=self.strategy.symbol,
                        timeframe=TimeFrame.Minute,
                        limit=1
                    )
                    client = crypto_data_client if is_crypto else stock_data_client
                    bars = client.get_crypto_bars(bars_request) if is_crypto else client.get_stock_bars(bars_request)
                    bars_df = bars.df
                    if not bars_df.empty:
                        price = float(bars_df.iloc[-1]['close'])
                        # Calculate best quantity based on risk per trade
                        quantity = (portfolio_value * self.risk_per_trade_pct / 100) / price
                        if not is_crypto:
                            quantity = math.floor(quantity)
                        if quantity < 0.001:  # too small
                            logging.info(".2f")
                            continue
                        if cash >= price * quantity:
                            order_data = MarketOrderRequest(
                                symbol=self.strategy.symbol,
                                qty=quantity,
                                side=OrderSide.BUY,
                                time_in_force=TimeInForce.GTC
                            )
                            trading_client.submit_order(order_data)
                            logging.info(f"BUY {quantity:.4f} {'units' if is_crypto else 'shares'} of {self.strategy.symbol} @ ${price:.2f} (risk ${portfolio_value * self.risk_per_trade_pct / 100:.2f})")
                        else:
                            logging.warning(f"Insufficient cash: {cash:.2f}, needed ${price * quantity:.2f}")
                elif signal == 'SELL':
                    if qty > 0:
                        order_data = MarketOrderRequest(
                            symbol=api_symbol,
                            qty=qty,
                            side=OrderSide.SELL,
                            time_in_force=TimeInForce.GTC
                        )
                        trading_client.submit_order(order_data)
                        logging.info(f"SELL {qty:.4f} {'units' if is_crypto else 'shares'} of {self.strategy.symbol}")
            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
            cycle_count += 1
            if cycle_count % 10 == 0:  # Every 10 minutes
                self.print_portfolio_status()
            time.sleep(60)  # Check every minute
        logging.info("Trading bot stopped.")

# Streamlit UI
st.title("Real-Time Stock Trading Bot (RSI Strategy)")
st.markdown("""
This bot monitors AAPL stock using RSI (Relative Strength Index) and executes buy/sell orders in paper trading mode.
- Buys when RSI ≤30 (oversold)
- Sells when RSI ≥70 (overbought)
- Only holds one position at a time
- Zero real money risk - paper trading only
""")

# Inputs
symbol = st.text_input("Stock Symbol (e.g., AAPL)", "AAPL")
risk_pct = st.slider("Risk per Trade (%)", 0.1, 5.0, 1.0, step=0.1)
buy_thresh = st.slider("RSI Buy Threshold", 10, 40, 30)
sell_thresh = st.slider("RSI Sell Threshold", 60, 90, 70)

if not API_KEY or not SECRET_KEY:
    st.error("API keys not found. Please create a .env file from .env.example and add your Alpaca keys.")
    st.stop()
else:
    st.success("API keys loaded successfully.")
    strategy = RSIStrategy(symbol, buy_threshold=buy_thresh, sell_threshold=sell_thresh)
    # Quantity will be calculated dynamically based on portfolio value
    bot = TradingBot(strategy, risk_per_trade_pct=risk_pct)

# Display current settings
asset_type = "Crypto" if symbol.upper().endswith('USD') else "Stock"
st.write(f"**Current Settings:** {symbol} ({asset_type}), Risk per Trade: {risk_pct}%, RSI Buy: ≤{buy_thresh}, Sell: ≥{sell_thresh}")

# Buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Start Bot"):
        if not bot.running:
            bot.start()
            st.success("Bot started! Monitoring market...")
        else:
            st.error("Bot already running.")
with col2:
    if st.button("Stop Bot"):
        bot.stop()
        st.success("Bot stopped.")

# Display logs and portfolio
if st.button("Refresh Logs"):
    pass  # Trigger rerun

st.text_area("Bot Logs", value='\n'.join(logs[-20:]), height=300, help="Last 20 log messages")

# Portfolio info
if trading_client:
    try:
        account = trading_client.get_account()
        st.metric("Portfolio Value", f"${float(account.portfolio_value):.2f}")
        st.metric("Cash Available", f"${float(account.cash):.2f}")
        positions = trading_client.get_all_positions()
        if positions:
            pos_list = [f"{p.symbol}: {p.qty} @ ${float(p.avg_entry_price):.2f}" for p in positions]
            st.write("Positions:")
            for p in pos_list:
                st.write(p)
        else:
            st.write("No open positions.")
    except Exception as e:
        st.error(f"Error fetching portfolio: {e}")

# Charts
st.subheader("RSI Indicator Chart (Live)")
if st.session_state.rsi_data:
    df_rsi = pd.DataFrame(st.session_state.rsi_data)
    st.line_chart(df_rsi.set_index('timestamp')[['rsi']])
    st.caption("RSI over time (with buy/sell thresholds)")

st.subheader("Portfolio Value Chart")
if st.session_state.portfolio_data:
    df_port = pd.DataFrame(st.session_state.portfolio_data)
    st.line_chart(df_port.set_index('timestamp')[['value']])
    st.caption("Portfolio value over time")

st.markdown("---")
st.markdown("**Setup Instructions:**")
st.markdown("""
1. Copy `.env.example` to `.env` and add your Alpaca paper trading API keys:
   ```
   APCA_API_KEY_ID=PK123456789ABCDEFGH
   APCA_API_SECRET_KEY=SK123456789ABCDEFGH
   ```
2. Restart the app - keys will be loaded automatically.
3. Choose symbol and quantity in the UI.
4. Adjust RSI thresholds if desired (default: buy ≤30, sell ≥70).
5. Click "Start Bot" - it will monitor every minute during market hours.
6. View logs and portfolio status here.
7. Stop when done to halt trading.
""")
st.markdown("*Note: Run during market hours (9:30 AM - 4:00 PM ET) for live trading.*")

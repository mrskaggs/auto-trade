import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)

class Portfolio:
    def __init__(self, initial_cash=10000):
        self.cash = initial_cash
        self.positions = {}  # symbol: {'quantity': qty, 'avg_price': price}
        self.trade_history = []
        self.portfolio_history = []

    def buy(self, symbol, quantity, price, date):
        cost = quantity * price
        if cost > self.cash:
            raise ValueError(f"Insufficient cash: {self.cash}, needed: {cost}")
        self.cash -= cost
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0, 'avg_price': 0}
        # Weighted average
        total_qty = self.positions[symbol]['quantity'] + quantity
        self.positions[symbol]['avg_price'] = (
            (self.positions[symbol]['quantity'] * self.positions[symbol]['avg_price'] + cost) / total_qty
        )
        self.positions[symbol]['quantity'] = total_qty
        self.trade_history.append({
            'date': date,
            'action': 'BUY',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'total': cost
        })
        logging.info(f"BUY: {quantity} {symbol} @ ${price:.2f}")

    def sell(self, symbol, quantity, price, date):
        if symbol not in self.positions or self.positions[symbol]['quantity'] < quantity:
            raise ValueError(f"Insufficient shares: {self.positions.get(symbol, {}).get('quantity', 0)}, selling: {quantity}")
        revenue = quantity * price
        self.cash += revenue
        avg_price = self.positions[symbol]['avg_price']
        profit_loss = (price - avg_price) * quantity
        self.positions[symbol]['quantity'] -= quantity
        if self.positions[symbol]['quantity'] == 0:
            del self.positions[symbol]
        self.trade_history.append({
            'date': date,
            'action': 'SELL',
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'total': revenue,
            'profit_loss': profit_loss
        })
        logging.info(f"SELL: {quantity} {symbol} @ ${price:.2f}, P&L: ${profit_loss:.2f}")

    def get_portfolio_value(self, current_prices):
        value = self.cash
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                value += pos['quantity'] * current_prices[symbol]
        return value

    def record_portfolio_state(self, date, current_prices):
        self.portfolio_history.append({
            'date': date,
            'value': self.get_portfolio_value(current_prices),
            'cash': self.cash
        })

class TradingStrategy:
    def __init__(self, short_window=9, long_window=21):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        signals = pd.DataFrame(index=data.index)
        signals['price'] = data['Close']
        signals['short_ma'] = data['Close'].rolling(window=self.short_window).mean()
        signals['long_ma'] = data['Close'].rolling(window=self.long_window).mean()
        signals['signal'] = 0  # 1 for buy, -1 for sell, 0 hold
        signals.loc[signals.index[self.short_window]:, 'signal'] = np.where(signals['short_ma'][self.short_window:] > signals['long_ma'][self.short_window:], 1, -1)
        signals['positions'] = signals['signal'].shift()  # use previous day's signal
        return signals

class Simulator:
    def __init__(self, portfolio, strategy):
        self.portfolio = portfolio
        self.strategy = strategy

    def run(self, symbol, start_date, end_date):
        logging.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        signals = self.strategy.generate_signals(data)

        position = 0  # current position, +1 long, -1 short, for simplicity simple long/short, no short accounting though
        trade_quantity = 10  # hard-coded for simplicity

        for date, row in signals.iterrows():
            price = row['price']
            current_prices = {symbol: price}
            self.portfolio.record_portfolio_state(date, current_prices)

            if row['positions'] == 1 and position <= 0:  # buy signal
                try:
                    self.portfolio.buy(symbol, trade_quantity, price, date)
                    position = 1
                except ValueError as e:
                    logging.warning(f"Could not buy: {e}")
            elif row['positions'] == -1 and position >= 0:  # sell signal
                try:
                    self.portfolio.sell(symbol, trade_quantity, price, date)
                    position = -1
                except ValueError as e:
                    logging.warning(f"Could not sell: {e}")

        logging.info("Simulation completed.")

    def report(self):
        df_trades = pd.DataFrame(self.portfolio.trade_history)
        df_portfolio = pd.DataFrame(self.portfolio.portfolio_history)
        df_portfolio['date'] = pd.to_datetime(df_portfolio['date'])

        initial_value = df_portfolio['value'].iloc[0]
        final_value = df_portfolio['value'].iloc[-1]
        total_return = final_value - initial_value
        total_return_pct = (total_return / initial_value) * 100

        # Plot portfolio value over time
        plt.figure(figsize=(12, 6))
        plt.plot(df_portfolio['date'], df_portfolio['value'], label='Portfolio Value')
        plt.title(f'Portfolio Value Over Time - Total Return: ${total_return:.2f} ({total_return_pct:.2f}%)')
        plt.xlabel('Date')
        plt.ylabel('Value ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig('portfolio_performance.png')
        plt.show()

        print(f"\nSimulation Results:")
        print(f"Initial Portfolio Value: ${initial_value:.2f}")
        print(f"Final Portfolio Value: ${final_value:.2f}")
        print(f"Total Return: ${total_return:.2f} ({total_return_pct:.2f}%)")
        print(f"Trades Executed: {len(df_trades)}")

        if not df_trades.empty:
            profit_trades = df_trades[df_trades['action'] == 'SELL']['profit_loss'].sum()
            print(f"Total Profit/Loss from Trades: ${profit_trades:.2f}")

        print("\nTop 10 Trades:")
        if not df_trades.empty:
            print(df_trades[['date', 'action', 'symbol', 'quantity', 'price', 'profit_loss']].tail(10))

if __name__ == "__main__":
    # Example usage
    portfolio = Portfolio(initial_cash=10000)
    strategy = TradingStrategy()
    simulator = Simulator(portfolio, strategy)

    # Simulate for Apple stock over 1 year
    start_date = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')

    symbol = 'AAPL'
    simulator.run(symbol, start_date, end_date)
    simulator.report()

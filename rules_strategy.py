import pandas as pd
from capital_manager import CapitalManager
from decimal import Decimal
from datetime import timedelta


class RulesBasedStrategy:
    def __init__(self, drop_threshold=0.05, rise_threshold=0.07, sma_period=50, min_delay_minutes=15):
        self.drop_threshold = Decimal(str(drop_threshold))
        self.rise_threshold = Decimal(str(rise_threshold))
        self.sma_period = sma_period
        self.min_delay = timedelta(minutes=min_delay_minutes)

        self.last_trade_time = None
        self.last_peak_price = None
        self.last_buy_price = None

    def run(self, df: pd.DataFrame, capital_manager: CapitalManager):
        df = df.copy()
        df['sma'] = df['close'].rolling(window=self.sma_period).mean()
        df.dropna(inplace=True)

        for i, row in df.iterrows():
            timestamp = pd.to_datetime(row['timestamp'])
            price = Decimal(str(row['close']))
            sma = Decimal(str(row['sma']))

            if self.last_trade_time and (timestamp - self.last_trade_time) < self.min_delay:
                continue

            # Update last peak price
            if self.last_peak_price is None or price > self.last_peak_price:
                self.last_peak_price = price

            # Buy condition
            if (
                self.last_peak_price and
                price < self.last_peak_price * (Decimal("1") - self.drop_threshold) and
                price < sma
            ):
                invest_amount = capital_manager.fiat * Decimal("0.10")
                if invest_amount >= Decimal("1.0"):
                    capital_manager.buy(invest_amount, price, timestamp.isoformat())
                    self.last_buy_price = price
                    self.last_trade_time = timestamp
                    self.last_peak_price = None  # reset after buy
                continue

            # Sell condition
            if (
                self.last_buy_price and
                price > self.last_buy_price * (Decimal("1") + self.rise_threshold)
            ):
                capital_manager.sell(capital_manager.doge, price, timestamp.isoformat())
                self.last_buy_price = None
                self.last_trade_time = timestamp
                self.last_peak_price = price  # reset peak after sell

        return capital_manager.get_portfolio_summary(price)

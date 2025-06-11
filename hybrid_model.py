import pandas as pd
from capital_manager import CapitalManager
from ml_model import MLTradingModel
from decimal import Decimal
from datetime import timedelta


class HybridStrategy:
    def __init__(self, ml_model: MLTradingModel, prob_threshold=0.6, min_delay_minutes=15):
        self.ml_model = ml_model
        self.prob_threshold = Decimal(str(prob_threshold))
        self.min_delay = timedelta(minutes=min_delay_minutes)

        self.last_trade_time = None
        self.last_buy_price = None

    def run(self, df: pd.DataFrame, capital_manager: CapitalManager):
        df = df.copy()
        df = df.sort_values('timestamp').reset_index(drop=True)

        self.ml_model.train(df)

        for i in range(50, len(df)):
            window = df.iloc[i-50:i+1]  # sliding window
            timestamp = df.iloc[i]['timestamp']
            price = Decimal(str(df.iloc[i]['close']))

            if self.last_trade_time and (timestamp - self.last_trade_time) < self.min_delay:
                continue

            prob = Decimal(str(self.ml_model.predict(window)))

            # BUY condition: ML model confidence is high, no recent trade
            if prob > self.prob_threshold and capital_manager.fiat >= Decimal("1.0"):
                invest_amount = capital_manager.fiat * Decimal("0.10")
                capital_manager.buy(invest_amount, price, timestamp.isoformat())
                self.last_buy_price = price
                self.last_trade_time = timestamp
                continue

            # SELL condition: price has appreciated 5% since last buy
            if (
                self.last_buy_price and
                price > self.last_buy_price * Decimal("1.05") and
                capital_manager.doge > Decimal("0")
            ):
                capital_manager.sell(capital_manager.doge, price, timestamp.isoformat())
                self.last_buy_price = None
                self.last_trade_time = timestamp

        return capital_manager.get_portfolio_summary(price)

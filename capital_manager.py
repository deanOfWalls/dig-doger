import datetime
from collections import deque
from decimal import Decimal, getcontext

getcontext().prec = 10  # High precision for financial calculations


class TradeType:
    BUY = "BUY"
    SELL = "SELL"


class TradeRecord:
    def __init__(self, timestamp, trade_type, amount_doge, price_per_coin, fee, fiat_before, fiat_after):
        self.timestamp = timestamp
        self.trade_type = trade_type
        self.amount_doge = Decimal(str(amount_doge))
        self.price_per_coin = Decimal(str(price_per_coin))
        self.fee = Decimal(str(fee))
        self.fiat_before = Decimal(str(fiat_before))
        self.fiat_after = Decimal(str(fiat_after))


class CapitalManager:
    def __init__(self, initial_fiat=10000.0, fee_rate=0.001, tax_rate=0.40):
        self.fiat = Decimal(str(initial_fiat))
        self.doge = Decimal('0')
        self.cost_basis_queue = deque()  # FIFO queue for cost basis tracking
        self.fee_rate = Decimal(str(fee_rate))
        self.tax_rate = Decimal(str(tax_rate))

        self.trade_log = []
        self.realized_gain = Decimal('0')
        self.unrealized_gain = Decimal('0')
        self.total_roi = Decimal('0')

    def _calculate_fee(self, amount_fiat):
        return amount_fiat * self.fee_rate

    def _log_trade(self, trade: TradeRecord):
        self.trade_log.append(trade)

    def buy(self, amount_fiat, price_per_coin, timestamp=None):
        price_per_coin = Decimal(str(price_per_coin))
        amount_fiat = Decimal(str(amount_fiat))
        timestamp = timestamp or datetime.datetime.utcnow().isoformat()

        fee = self._calculate_fee(amount_fiat)
        net_fiat = amount_fiat - fee
        amount_doge = net_fiat / price_per_coin

        if amount_fiat > self.fiat:
            raise ValueError("Insufficient fiat balance")

        # Update portfolio
        self.fiat -= amount_fiat
        self.doge += amount_doge
        self.cost_basis_queue.append((amount_doge, price_per_coin))

        # Log trade
        self._log_trade(TradeRecord(timestamp, TradeType.BUY, amount_doge, price_per_coin, fee, self.fiat + amount_fiat, self.fiat))

    def sell(self, amount_doge, price_per_coin, timestamp=None):
        price_per_coin = Decimal(str(price_per_coin))
        amount_doge = Decimal(str(amount_doge))
        timestamp = timestamp or datetime.datetime.utcnow().isoformat()

        if amount_doge > self.doge:
            raise ValueError("Insufficient DOGE balance")

        gross_fiat = amount_doge * price_per_coin
        fee = self._calculate_fee(gross_fiat)
        net_fiat = gross_fiat - fee

        # Update portfolio
        self.fiat += net_fiat
        self.doge -= amount_doge

        # Calculate realized gain using FIFO
        sold = amount_doge
        realized_gain = Decimal('0')
        while sold > 0 and self.cost_basis_queue:
            cb_amount, cb_price = self.cost_basis_queue[0]
            if cb_amount <= sold:
                gain = (price_per_coin - cb_price) * cb_amount
                realized_gain += gain
                sold -= cb_amount
                self.cost_basis_queue.popleft()
            else:
                gain = (price_per_coin - cb_price) * sold
                realized_gain += gain
                self.cost_basis_queue[0] = (cb_amount - sold, cb_price)
                sold = Decimal('0')

        self.realized_gain += realized_gain

        # Log trade
        self._log_trade(TradeRecord(timestamp, TradeType.SELL, amount_doge, price_per_coin, fee, self.fiat - net_fiat, self.fiat))

    def get_portfolio_summary(self, current_price):
        current_price = Decimal(str(current_price))
        unrealized = Decimal('0')
        for amount, cb_price in self.cost_basis_queue:
            unrealized += (current_price - cb_price) * amount
        self.unrealized_gain = unrealized

        invested = sum([amount * cb_price for amount, cb_price in self.cost_basis_queue])
        total_value = self.fiat + (self.doge * current_price)
        self.total_roi = ((total_value - invested) / invested) if invested else Decimal('0')

        return {
            "fiat": float(self.fiat),
            "doge": float(self.doge),
            "avg_cost": float(invested / self.doge) if self.doge > 0 else 0.0,
            "realized_gain": float(self.realized_gain),
            "unrealized_gain": float(self.unrealized_gain),
            "total_roi": float(self.total_roi),
            "portfolio_value": float(total_value)
        }

    def get_trade_log(self):
        return [vars(t) for t in self.trade_log]

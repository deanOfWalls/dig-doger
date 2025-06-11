import gym
import numpy as np
import pandas as pd
from decimal import Decimal
from gym import spaces


class DogeTradingEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, initial_fiat=10000.0):
        super(DogeTradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_fiat = Decimal(str(initial_fiat))

        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)

        self._reset_portfolio()
        self.current_step = 50  # Start after enough data for indicators

    def _reset_portfolio(self):
        self.fiat = self.initial_fiat
        self.doge = Decimal("0")
        self.avg_cost = Decimal("0")
        self.trades = []

    def reset(self):
        self._reset_portfolio()
        self.current_step = 50
        return self._get_obs()

    def _get_obs(self):
        if self.current_step >= len(self.df):
            return np.zeros(5, dtype=np.float32)

        window = self.df.iloc[self.current_step - 50:self.current_step]

        sma_10 = window['close'].rolling(10).mean().bfill()
        sma_50 = window['close'].rolling(50).mean().bfill()
        momentum = window['close'].iloc[-1] - window['close'].iloc[-5]
        volatility = window['close'].pct_change().rolling(10).std().iloc[-1]

        obs = np.array([
            window['close'].values[-1],
            sma_10.values[-1],
            sma_50.values[-1],
            momentum,
            volatility
        ], dtype=np.float32)

        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return obs

    def _get_price(self):
        return Decimal(str(self.df.iloc[self.current_step]['close']))

    def step(self, action):
        price = self._get_price()

        reward = Decimal("0")

        if action == 1 and self.fiat > Decimal("1.0"):  # Buy
            amount_to_spend = self.fiat * Decimal("0.10")
            qty = amount_to_spend / price
            self.fiat -= amount_to_spend
            self.avg_cost = ((self.avg_cost * self.doge) + (price * qty)) / (self.doge + qty)
            self.doge += qty
            self.trades.append(("buy", price, self.current_step))

        elif action == 2 and self.doge > Decimal("0"):  # Sell
            self.fiat += self.doge * price
            realized_gain = (price - self.avg_cost) * self.doge
            reward += realized_gain
            self.doge = Decimal("0")
            self.avg_cost = Decimal("0")
            self.trades.append(("sell", price, self.current_step))

        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        if done:
            obs = self._get_obs() if self.current_step < len(self.df) else np.zeros(5, dtype=np.float32)
        else:
            obs = self._get_obs()

        unrealized = (price - self.avg_cost) * self.doge if self.doge > 0 else Decimal("0")
        total_value = self.fiat + (self.doge * price)
        roi = (total_value - self.initial_fiat) / self.initial_fiat
        reward += unrealized * Decimal("0.1") + roi * Decimal("0.01")

        return obs, float(reward), done, {}

    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Fiat: {self.fiat:.2f}, DOGE: {self.doge:.4f}, Avg Cost: {self.avg_cost:.4f}")

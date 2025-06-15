# drl_env.py

import gym
import numpy as np
import pandas as pd
from gym import spaces

class DogeTradingEnv(gym.Env):
    def __init__(self, df, initial_balance=1000.0, max_doge=1_000_000, lookahead_steps=10, exploration_steps=5000):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.max_doge = max_doge
        self.lookahead_steps = lookahead_steps
        self.exploration_steps = exploration_steps

        self.action_space = spaces.Box(low=0.0, high=2.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(42,), dtype=np.float32
        )

        self._prepare_indicators()
        self.reset()

    def _prepare_indicators(self):
        for name, window in {"5min": 5, "15min": 15, "1h": 60, "1d": 1440}.items():
            self.df[f"ma_{name}"] = self.df["close"].rolling(window).mean()
            self.df[f"std_{name}"] = self.df["close"].rolling(window).std()
            self.df[f"mom_{name}"] = (self.df["close"] - self.df[f"ma_{name}"]) / self.df[f"ma_{name}"]

        self.df["range_pct"] = (self.df["high"] - self.df["low"]) / self.df["low"]

        delta = self.df["close"].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(gain).rolling(14).mean()
        roll_down = pd.Series(loss).rolling(14).mean()
        rs = roll_up / (roll_down + 1e-8)
        self.df["rsi_14"] = 100 - (100 / (1 + rs))

        self.df["ema_12"] = self.df["close"].ewm(span=12).mean()
        self.df["ema_26"] = self.df["close"].ewm(span=26).mean()
        self.df["macd_line"] = self.df["ema_12"] - self.df["ema_26"]
        self.df["macd_signal"] = self.df["macd_line"].ewm(span=9).mean()

        self.df.dropna(inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def reset(self):
        self.balance = self.initial_balance
        self.doge = 0.0
        self.net_worth = self.initial_balance
        self.trades = []
        self.current_step = 0
        return self._get_observation()

    def _get_observation(self):
        row = self.df.iloc[self.current_step]

        obs = np.array([
            row["open"], row["high"], row["low"], row["close"], row["volume"],
            row["ma_5min"], row["std_5min"], row["mom_5min"],
            row["ma_15min"], row["std_15min"], row["mom_15min"],
            row["ma_1h"], row["std_1h"], row["mom_1h"],
            row["ma_1d"], row["std_1d"], row["mom_1d"],
            row["range_pct"],
            row["rsi_14"], row["ema_12"], row["ema_26"],
            row["macd_line"], row["macd_signal"],
            self.balance, self.doge, self.net_worth
        ])

        return np.pad(obs, (0, self.observation_space.shape[0] - len(obs)), 'constant')

    def step(self, action):
        current_price = self.df.iloc[self.current_step]["close"]
        prev_worth = self.net_worth
        prev_balance = self.balance
        prev_doge = self.doge
        prev_price = current_price

        # 🔥 Epsilon-greedy exploration override
        if self.current_step < self.exploration_steps:
            epsilon = 1.0 - (self.current_step / self.exploration_steps)
            if np.random.rand() < epsilon:
                action = np.array([np.random.uniform(0.0, 2.0)], dtype=np.float32)
                print(f"[Exploration] Random action injected: {action[0]:.4f}")

        action = float(np.clip(action, 0.0, 2.0))
        print(f"[Step {self.current_step}] Action taken: {action:.4f}")

        if action < 1.0:
            sell_frac = 1.0 - action
            doge_to_sell = self.doge * sell_frac
            self.balance += doge_to_sell * current_price
            self.doge -= doge_to_sell
        elif action > 1.0:
            buy_frac = action - 1.0
            spend = self.balance * buy_frac
            doge_bought = spend / current_price
            self.balance -= spend
            self.doge += doge_bought

        self.net_worth = self.balance + self.doge * current_price
        self.trades.append(self.net_worth)

        reward = self._calculate_reward(prev_worth, prev_balance, prev_doge, prev_price)
        self.current_step += 1
        done = self.current_step >= len(self.df) - self.lookahead_steps - 1

        return self._get_observation(), reward, done, {}

    def _calculate_reward(self, prev_worth, prev_balance, prev_doge, prev_price):
        future_step = min(self.current_step + self.lookahead_steps, len(self.df) - 1)
        future_price = self.df.iloc[future_step]["close"]
        projected_worth = self.balance + self.doge * future_price
        long_term_gain = projected_worth - prev_worth

        immediate_gain = self.net_worth - prev_worth
        reward = 0.0

        reward += immediate_gain * 0.5
        reward += long_term_gain * 0.25

        if self.doge > prev_doge and future_price > prev_price:
            reward += 0.5
        if self.doge < prev_doge and future_price > prev_price:
            reward -= 0.5

        volatility = self.df.iloc[self.current_step]["range_pct"]
        if volatility > 0.05:
            reward -= abs(immediate_gain) * 0.1

        if abs(reward) < 0.001:
            reward -= 0.005

        return reward

    def render(self, mode="human"):
        print(f"Step: {self.current_step} | Net Worth: {self.net_worth:.2f} | Balance: {self.balance:.2f} | DOGE: {self.doge:.2f}")

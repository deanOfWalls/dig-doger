import gym
import numpy as np
from gym import spaces
from capital_manager import CapitalManager


class DogeTradingEnv(gym.Env):
    def __init__(self, df, initial_fiat=10000.0, fee_rate=0.001):
        super(DogeTradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.capital_manager = CapitalManager(initial_fiat=initial_fiat, fee_rate=fee_rate)
        self.current_step = 50  # Start after feature burn-in
        self.window_size = 50

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation: close prices + volume % change + SMA ratio
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, 4),
            dtype=np.float32
        )

        self.done = False

    def _get_obs(self):
        window = self.df.iloc[self.current_step - self.window_size:self.current_step]
        close = window['close'].pct_change().fillna(0).values
        volume = window['volume'].pct_change().fillna(0).values
        sma_10 = window['close'].rolling(10).mean().fillna(method='bfill')
        sma_50 = window['close'].rolling(50).mean().fillna(method='bfill')
        sma_ratio = (sma_10 / sma_50).fillna(1).values
        return np.vstack([close, volume, sma_ratio, sma_ratio]).T.astype(np.float32)

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, self.done, {}

        row = self.df.iloc[self.current_step]
        timestamp = row['timestamp']
        price = row['close']

        reward = 0.0

        if action == 1 and self.capital_manager.fiat >= 1.0:
            invest_amount = self.capital_manager.fiat * 0.10
            self.capital_manager.buy(invest_amount, price, timestamp)
        elif action == 2 and self.capital_manager.doge > 0:
            cost = self.capital_manager.get_portfolio_summary(price).get('avg_cost', price)
            if price > cost:
                self.capital_manager.sell(self.capital_manager.doge, price, timestamp)

        self.current_step += 1
        if self.current_step >= len(self.df):
            self.done = True

        # Reward = net ROI change, penalize loss
        summary = self.capital_manager.get_portfolio_summary(price)
        reward = float(summary['total_roi']) - 0.002 * abs(float(summary['unrealized_gain']))

        return self._get_obs(), reward, self.done, {}

    def reset(self):
        self.capital_manager = CapitalManager()
        self.current_step = 50
        self.done = False
        return self._get_obs()

    def render(self, mode='human'):
        summary = self.capital_manager.get_portfolio_summary(self.df.iloc[self.current_step]['close'])
        print(summary)

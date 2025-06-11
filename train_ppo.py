import pandas as pd
from drl_env import DogeTradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

def main():
    df = pd.read_csv("doge_1m_ohlcv.csv")
    df['timestamp'] = pd.to_datetime(df['date'])
    df.drop(columns=['date'], inplace=True)
    df = df.sort_values('timestamp').reset_index(drop=True)

    env = DummyVecEnv([lambda: DogeTradingEnv(df)])

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_doge_logs")

    model.learn(total_timesteps=200_000)

    # Save the model
    os.makedirs("models", exist_ok=True)
    model.save("models/ppo_doge")

    print("Training complete. Model saved to models/ppo_doge")

if __name__ == "__main__":
    main()

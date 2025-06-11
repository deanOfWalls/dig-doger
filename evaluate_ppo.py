import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from drl_env import DogeTradingEnv
import os

def main():
    df = pd.read_csv("doge_1m_ohlcv.csv")
    df['timestamp'] = pd.to_datetime(df['date'])

    print("Loading trained PPO model...")
    model = PPO.load("models/ppo_doge")

    env = DummyVecEnv([lambda: DogeTradingEnv(df)])

    obs = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        total_reward += reward[0]

    print(f"Total PPO evaluation reward: {total_reward:.2f}")

    os.makedirs("results", exist_ok=True)
    result_path = "results/ppo_evaluation.txt"
    with open(result_path, "w") as f:
        f.write(f"Total PPO evaluation reward: {total_reward:.2f}\n")
        f.write(f"Final env info: {info[0]}\n")

    print(f"Final env info: {info[0]}")
    print(f"Evaluation results saved to {result_path}")

if __name__ == "__main__":
    main()

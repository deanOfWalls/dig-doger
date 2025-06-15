# walk_foward_evaluation.py

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from drl_env import DogeTradingEnv
import os

WINDOW_SIZE = 100_000
STEP_FORWARD = 50_000
MODEL_PATH = "models/ppo_doge_late"
RESULTS_PATH = "results/walk_forward_results.csv"

os.makedirs("results", exist_ok=True)

def evaluate(model, df):
    env = DogeTradingEnv(df)
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
    final_value = env.net_worth
    roi = (final_value - env.initial_balance) / env.initial_balance
    sharpe = compute_sharpe(env.trades)
    max_dd = compute_max_drawdown(env.trades)
    return final_value, roi, sharpe, max_dd

def compute_sharpe(trades, risk_free_rate=0.0):
    returns = np.diff(trades) / trades[:-1]
    excess_returns = returns - risk_free_rate / 252
    if excess_returns.std() == 0:
        return 0.0
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)

def compute_max_drawdown(equity_curve):
    equity_curve = np.array(equity_curve)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def main():
    df = pd.read_csv("doge_1m_ohlcv.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    model = PPO.load(MODEL_PATH)

    results = []

    for start in range(0, len(df) - WINDOW_SIZE, STEP_FORWARD):
        end = start + WINDOW_SIZE
        df_slice = df.iloc[start:end].copy()
        print(f"Evaluating window: {start} → {end}")

        final_value, roi, sharpe, max_dd = evaluate(model, df_slice)

        results.append({
            "start_index": start,
            "end_index": end,
            "final_value": final_value,
            "roi": roi,
            "sharpe": sharpe,
            "max_drawdown": max_dd
        })

    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"✅ Walk-forward results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()

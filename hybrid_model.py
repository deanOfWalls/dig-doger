# hybrid_model.py

import pandas as pd
import numpy as np
import os
from sb3_contrib.ppo_recurrent import RecurrentPPO
from drl_env import DogeTradingEnv

RESULTS_PATH = "results/hybrid_results.csv"
MODEL_PATH = "models/ppo_doge_recurrent"
WINDOW_SIZE = 100_000
STEP_FORWARD = 50_000
INITIAL_BALANCE = 1000.0

def compute_rsi(close, window=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    return 100 - (100 / (1 + rs))

def compute_sharpe(equity_curve):
    returns = np.diff(equity_curve) / equity_curve[:-1]
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(252)

def compute_max_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    return ((equity_curve - peak) / peak).min()

def evaluate(model, df):
    df["rsi"] = compute_rsi(df["close"])
    df = df.dropna().reset_index(drop=True)

    env = DogeTradingEnv(df)
    obs = env.reset()
    lstm_state = None
    done = False

    while not done:
        rsi = df.iloc[env.current_step]["rsi"]
        price = df.iloc[env.current_step]["close"]

        if rsi < 30 and env.balance > 0:
            action = np.array([[2.0]])  # force buy
        elif rsi > 70 and env.doge > 0 and env.doge * price > INITIAL_BALANCE:
            action = np.array([[0.0]])  # force sell
        else:
            action, lstm_state = model.predict(
                obs,
                state=lstm_state,
                episode_start=np.array([done]),
                deterministic=True
            )

        obs, reward, done, _ = env.step(action)

    net_worths = np.array(env.trades)
    final_value = env.net_worth
    roi = (final_value - INITIAL_BALANCE) / INITIAL_BALANCE
    sharpe = compute_sharpe(net_worths)
    max_dd = compute_max_drawdown(net_worths)

    return final_value, roi, sharpe, max_dd

def main():
    os.makedirs("results", exist_ok=True)
    df = pd.read_csv("doge_1m_ohlcv.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    model = RecurrentPPO.load(MODEL_PATH)

    results = []

    for start in range(0, len(df) - WINDOW_SIZE, STEP_FORWARD):
        end = start + WINDOW_SIZE
        df_slice = df.iloc[start:end].copy()
        print(f"Evaluating window {start} → {end}")
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
    print(f"✅ Hybrid results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()

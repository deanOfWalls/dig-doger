# rules_based_model.py

import pandas as pd
import numpy as np
import os

RESULTS_PATH = "results/rules_based_results.csv"
WINDOW_SIZE = 100_000
STEP_FORWARD = 50_000
INITIAL_BALANCE = 1000.0

def compute_rsi(close, window=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_sharpe(equity_curve, risk_free_rate=0.0):
    returns = np.diff(equity_curve) / equity_curve[:-1]
    excess_returns = returns - risk_free_rate / 252
    return 0.0 if excess_returns.std() == 0 else (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)

def compute_max_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def simulate(df):
    df["rsi"] = compute_rsi(df["close"])
    df = df.dropna(subset=["rsi"]).reset_index(drop=True)

    balance = INITIAL_BALANCE
    doge = 0.0
    net_worths = []

    for i in range(len(df)):
        price = df.iloc[i]["close"]
        rsi = df.iloc[i]["rsi"]

        net_worth = balance + doge * price
        net_worths.append(net_worth)

        if rsi < 30 and balance > 0:
            doge = balance / price
            balance = 0
        elif rsi > 70 and doge > 0:
            potential = doge * price
            if potential > INITIAL_BALANCE:
                balance = potential
                doge = 0

    final_value = balance + doge * df.iloc[-1]["close"]
    roi = (final_value - INITIAL_BALANCE) / INITIAL_BALANCE
    sharpe = compute_sharpe(np.array(net_worths))
    max_dd = compute_max_drawdown(np.array(net_worths))

    return final_value, roi, sharpe, max_dd

def main():
    os.makedirs("results", exist_ok=True)
    df = pd.read_csv("doge_1m_ohlcv.csv", parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    results = []

    for start in range(0, len(df) - WINDOW_SIZE, STEP_FORWARD):
        end = start + WINDOW_SIZE
        df_slice = df.iloc[start:end].copy()
        print(f"Simulating window {start} → {end}")
        final_value, roi, sharpe, max_dd = simulate(df_slice)
        results.append({
            "start_index": start,
            "end_index": end,
            "final_value": final_value,
            "roi": roi,
            "sharpe": sharpe,
            "max_drawdown": max_dd
        })

    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"✅ Rules-based results saved to {RESULTS_PATH}")

if __name__ == "__main__":
    main()

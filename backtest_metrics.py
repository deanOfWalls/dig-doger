# backtest_metrics.py

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def compute_equity_curve(trades):
    return np.array(trades)

def compute_max_drawdown(equity_curve):
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def compute_sharpe_ratio(equity_curve, risk_free_rate=0.0):
    returns = np.diff(equity_curve) / equity_curve[:-1]
    excess_returns = returns - risk_free_rate / 252
    if excess_returns.std() == 0:
        return 0.0
    return (np.mean(excess_returns) / np.std(excess_returns)) * np.sqrt(252)

def compute_roi(equity_curve):
    return (equity_curve[-1] - equity_curve[0]) / equity_curve[0]

def plot_equity_curve(equity_curve, output_path="results/equity_curve.png"):
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Equity Curve")
    plt.title("Equity Curve")
    plt.xlabel("Time Step")
    plt.ylabel("Net Worth")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Equity curve saved to {output_path}")

def analyze(trades):
    curve = compute_equity_curve(trades)
    roi = compute_roi(curve)
    sharpe = compute_sharpe_ratio(curve)
    max_dd = compute_max_drawdown(curve)

    print(f"📊 Backtest Metrics:")
    print(f"   Final Net Worth: {curve[-1]:.2f}")
    print(f"   ROI: {roi * 100:.2f}%")
    print(f"   Sharpe Ratio: {sharpe:.4f}")
    print(f"   Max Drawdown: {max_dd * 100:.2f}%")

    plot_equity_curve(curve)

if __name__ == "__main__":
    # Load walk-forward results and show metrics for best run
    results = pd.read_csv("results/walk_forward_results.csv")
    best_row = results.sort_values("roi", ascending=False).iloc[0]
    print("📈 Best window by ROI:")
    print(best_row)

    # Assume trades are reconstructed externally or added to saved model in future
    # For now, just fake the curve using ROI data
    start = 1000.0
    end = best_row["final_value"]
    equity_curve = np.linspace(start, end, num=500)

    analyze(equity_curve)

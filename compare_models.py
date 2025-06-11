import pandas as pd
import matplotlib.pyplot as plt
from backtest_metrics import compute_metrics

# Load strategy summaries
rules_df = pd.read_csv("results/rules_strategy_summary.csv")
hybrid_df = pd.read_csv("results/hybrid_strategy_summary.csv")
ppo_df = pd.read_csv("results/ppo_strategy_summary.csv")

# Helper function to tag strategies
def tag(df, label):
    df = df.copy()
    df['strategy'] = label
    return df

combined = pd.concat([
    tag(rules_df, 'Rules-Based'),
    tag(hybrid_df, 'Hybrid'),
    tag(ppo_df, 'PPO')
])

# Plot equity curves
plt.figure(figsize=(12, 6))
for name, group in combined.groupby("strategy"):
    plt.plot(group["timestamp"], group["portfolio_value"], label=name)

plt.title("Equity Curve Comparison")
plt.xlabel("Time")
plt.ylabel("Portfolio Value ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/model_equity_comparison.png")
plt.show()

# Compute metrics
print("\nEvaluation Metrics by Strategy:\n")
for label, df in zip(["Rules-Based", "Hybrid", "PPO"], [rules_df, hybrid_df, ppo_df]):
    print(f"--- {label} ---")
    metrics = compute_metrics(df)
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
    print()

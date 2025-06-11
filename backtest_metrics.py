import numpy as np
import matplotlib.pyplot as plt


def calculate_equity_curve(trade_log, initial_fiat=10000.0):
    equity = initial_fiat
    timestamps = []
    values = []

    for trade in trade_log:
        timestamps.append(trade['timestamp'])
        if trade['trade_type'] == "BUY":
            equity -= float(trade['amount_doge']) * float(trade['price_per_coin']) + float(trade['fee'])
        elif trade['trade_type'] == "SELL":
            equity += float(trade['amount_doge']) * float(trade['price_per_coin']) - float(trade['fee'])
        values.append(equity)

    return timestamps, values


def calculate_drawdown(equity_curve):
    peaks = np.maximum.accumulate(equity_curve)
    drawdowns = (peaks - equity_curve) / peaks
    return np.max(drawdowns)


def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    if np.std(excess_returns) == 0:
        return 0.0
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252 * 24 * 60)  # per-minute to annualized


def evaluate_strategy(trade_log, final_portfolio_value, initial_fiat=10000.0, plot=True):
    timestamps, equity_curve = calculate_equity_curve(trade_log, initial_fiat)
    equity_curve = np.array(equity_curve)
    returns = np.diff(equity_curve) / equity_curve[:-1]

    max_drawdown = calculate_drawdown(equity_curve)
    sharpe = sharpe_ratio(returns)
    roi = (final_portfolio_value - initial_fiat) / initial_fiat

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(timestamps, equity_curve)
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Portfolio Value")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "Final Value": round(final_portfolio_value, 2),
        "ROI": round(roi * 100, 2),
        "Max Drawdown": round(max_drawdown * 100, 2),
        "Sharpe Ratio": round(sharpe, 2)
    }

# generate_mock_data.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# === CONFIG ===
OUTDIR = "mock_data"
DATASETS = 10
MINUTES = 60 * 24 * 180  # 6 months of 1-min data
START_PRICE = 0.01

np.random.seed(42)

# === Patterns ===
def generate_uptrend(length, start):
    drift = 0.00002
    vol = 0.002
    return simulate_prices(length, start, drift, vol)

def generate_downtrend(length, start):
    drift = -0.00002
    vol = 0.002
    return simulate_prices(length, start, drift, vol)

def generate_sideways(length, start):
    drift = 0.0
    vol = 0.001
    return simulate_prices(length, start, drift, vol)

def generate_choppy(length, start):
    drift = 0.0
    vol = 0.003
    return simulate_prices(length, start, drift, vol)

def generate_parabola_then_crash(length, start):
    third = length // 3
    rise = simulate_prices(third, start, 0.0002, 0.002)
    peak = rise[-1]
    flat = simulate_prices(third, peak, 0.0, 0.001)
    fall = simulate_prices(length - 2 * third, flat[-1], -0.0004, 0.004)
    return np.concatenate([rise, flat, fall])

def simulate_prices(n, start_price, drift, vol):
    returns = np.random.normal(loc=drift, scale=vol, size=n)
    log_prices = np.log(start_price) + np.cumsum(returns)
    return np.exp(log_prices)

def create_ohlcv(prices):
    ohlcv = []
    timestamp = datetime(2020, 1, 1)
    for price in prices:
        high = price * np.random.uniform(1.000, 1.002)
        low = price * np.random.uniform(0.998, 1.000)
        open_price = price * np.random.uniform(0.999, 1.001)
        close = price
        volume = np.random.randint(10000, 50000)
        ohlcv.append([timestamp, open_price, high, low, close, volume])
        timestamp += timedelta(minutes=1)
    return pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

def save_mock_csv(df, index):
    os.makedirs(OUTDIR, exist_ok=True)
    path = os.path.join(OUTDIR, f"mock_set_{index:02}.csv")
    df.to_csv(path, index=False)
    print(f"✅ Saved {path}")

# === MAIN ===
if __name__ == "__main__":
    generators = [
        generate_uptrend,
        generate_downtrend,
        generate_sideways,
        generate_choppy,
        generate_parabola_then_crash,
        generate_uptrend,
        generate_downtrend,
        generate_choppy,
        generate_parabola_then_crash,
        generate_sideways
    ]

    for i, gen in enumerate(generators):
        prices = gen(MINUTES, START_PRICE)
        df = create_ohlcv(prices)
        save_mock_csv(df, i + 1)

    print("\n🎉 All mock datasets generated.")

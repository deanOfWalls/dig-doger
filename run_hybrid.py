import pandas as pd
from capital_manager import CapitalManager
from hybrid_model import HybridStrategy
from ml_model import MLTradingModel

def main():
    # Load DOGE data
    df = pd.read_csv("doge_1m_ohlcv.csv")
    df['timestamp'] = pd.to_datetime(df['date'])
    df.drop(columns=['date'], inplace=True)
    df = df.sort_values('timestamp')

    # Initialize components
    capital_manager = CapitalManager(initial_fiat=10000.0)
    ml_model = MLTradingModel(window_size=5, threshold=0.005)
    strategy = HybridStrategy(ml_model=ml_model, prob_threshold=0.6, min_delay_minutes=15)

    # Run strategy
    summary = strategy.run(df, capital_manager)

    # Output results
    print("\nFinal Portfolio Summary:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    print("\nTrade Log:")
    for trade in capital_manager.get_trade_log():
        print(trade)

if __name__ == "__main__":
    main()

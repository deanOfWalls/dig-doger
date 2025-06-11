import pandas as pd
from capital_manager import CapitalManager
from rules_strategy import RulesBasedStrategy

def main():
    # Load DOGE data
    df = pd.read_csv("doge_1m_ohlcv.csv")
    df['timestamp'] = pd.to_datetime(df['date'])
    df.drop(columns=['date'], inplace=True)
    df = df.sort_values('timestamp')

    # Initialize components
    capital_manager = CapitalManager(initial_fiat=10000.0)
    strategy = RulesBasedStrategy(
        drop_threshold=0.05,
        rise_threshold=0.07,
        sma_period=50,
        min_delay_minutes=15
    )

    # Run the strategy
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

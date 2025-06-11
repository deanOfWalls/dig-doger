import pandas as pd
from capital_manager import CapitalManager
from rules_strategy import RulesBasedStrategy
from backtest_metrics import evaluate_strategy

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

    # Run strategy
    summary = strategy.run(df, capital_manager)
    trade_log = capital_manager.get_trade_log()

    # Evaluate performance
    metrics = evaluate_strategy(
        trade_log=trade_log,
        final_portfolio_value=summary['portfolio_value'],
        initial_fiat=10000.0,
        plot=True
    )

    print("\nEvaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()

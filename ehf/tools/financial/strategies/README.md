# Eagle Hill Fund - Trading Strategies Framework

This directory contains the core trading strategy implementations for Eagle Hill Fund, organized into distinct families of quantitative strategies. Each strategy family is designed to be modular, testable, and easily extensible.

## Strategy Architecture

The framework follows a clean folder/child-class setup that allows the codebase to grow with different families of strategies. Each strategy family contains:

- **Base Strategy Classes**: Common backtest hooks and indicator setup
- **Concrete Implementations**: Specific strategy algorithms
- **Testing & Validation**: Backtesting and performance analysis tools

## Strategy Families

### 1. **Trend-Following Strategies** (`trend/`)
- Moving average crossovers (SMA, EMA, MACD)
- Breakouts (price, volatility, Donchian channels)
- Momentum (relative strength, rate of change)

### 2. **Mean-Reversion Strategies** (`mean_reversion/`)
- Pairs/statistical arbitrage (cointegration, spread z-score)
- Bollinger Band reversions
- RSI-based contrarian entries

### 3. **Volatility-Based Strategies** (`volatility/`)
- Straddles/strangles simulation (for backtesting options proxies)
- ATR breakouts or compressions
- Volatility regime switching (high/low vol filters)

### 4. **Seasonality & Cyclical Strategies** (`seasonality/`)
- Day-of-week effects
- Month-end/turn-of-month effects
- Quarterly earnings drift

### 5. **Event-Driven Strategies** (`event_driven/`)
- Earnings announcements
- Dividends / splits
- Macro events (FOMC, payrolls, CPI releases)

### 6. **Factor/Smart Beta Strategies** (`smart_beta/`)
- Value (P/E, P/B screens)
- Quality (ROE, debt ratios)
- Size (small-cap tilt)
- Momentum factors

### 7. **Machine Learning / Data-Driven Strategies** (`machine_learning/`)
- Classification (up/down prediction)
- Regression (expected return forecasting)
- Reinforcement learning agents

### 8. **Portfolio & Risk-Based Strategies** (`portfolio_and_risk/`)
- Risk parity
- Minimum variance portfolio
- Kelly criterion / fractional Kelly
- Equal weight vs market cap

### 9. **Execution / Microstructure Strategies** (`execution_and_microstructure/`)
- VWAP/TWAP backtests
- Liquidity-sensitive execution
- Slippage impact models

## Usage

Each strategy family follows a consistent interface:

```python
from eagle_hill_fund.server.tools.financial.strategies.trend import MovingAverageCrossover
from eagle_hill_fund.server.tools.financial.strategies.mean_reversion import PairsTrading

# Initialize and run strategies
strategy = MovingAverageCrossover(fast_period=20, slow_period=50)
results = strategy.backtest(data)
```

## Development Guidelines

1. **Inheritance**: All strategies inherit from `BaseStrategy`
2. **Naming**: Use descriptive class names (e.g., `MovingAverageCrossover`, `PairsTradingZScore`)
3. **Testing**: Include comprehensive backtesting and validation
4. **Documentation**: Each strategy should have clear docstrings and examples
5. **Modularity**: Keep strategies focused and single-purpose

## Contributing

When adding new strategies:
1. Choose the appropriate strategy family directory
2. Create a new Python file with your strategy implementation
3. Follow the existing naming conventions
4. Add comprehensive tests and documentation
5. Update the relevant README.md file

## Performance & Risk Management

All strategies include:
- Risk management controls
- Performance attribution
- Drawdown analysis
- Transaction cost modeling
- Slippage considerations

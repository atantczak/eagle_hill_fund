# Mean-Reversion Strategies

This directory contains mean-reversion trading strategies that capitalize on the tendency of asset prices to return to their historical average or mean over time.

## Strategy Philosophy

Mean-reversion strategies are based on the assumption that asset prices will eventually return to their long-term average or mean. These strategies identify when prices have deviated significantly from their mean and take positions expecting a reversion.

## Strategy Types

### 1. **Pairs Trading / Statistical Arbitrage**
- **Cointegration-Based**: Uses statistical cointegration to identify related assets
- **Spread Z-Score**: Monitors the spread between two assets for mean reversion
- **Correlation-Based**: Identifies temporarily diverging correlated assets

### 2. **Bollinger Band Strategies**
- **Bollinger Band Reversions**: Trades when prices touch or exceed band boundaries
- **Bollinger Band Squeeze**: Identifies periods of low volatility before expansion
- **Multiple Timeframe Bands**: Uses bands across different timeframes

### 3. **RSI-Based Contrarian Strategies**
- **RSI Oversold/Overbought**: Trades against extreme RSI readings
- **RSI Divergence**: Identifies divergences between price and RSI
- **RSI Mean Reversion**: Uses RSI's tendency to revert to 50

### 4. **Price-Based Mean Reversion**
- **Support/Resistance Levels**: Trades bounces off key price levels
- **Moving Average Reversions**: Trades deviations from moving averages
- **Price Channels**: Uses price channels to identify reversion opportunities

## Key Characteristics

- **Mean Identification**: Statistical methods to identify the "true" mean
- **Deviation Measurement**: Quantifies how far prices have deviated
- **Reversion Probability**: Estimates likelihood of mean reversion
- **Timing**: Critical for entry and exit points

## Implementation Examples

```python
# Pairs Trading Strategy
class PairsTradingZScore(BaseStrategy):
    def __init__(self, lookback_period=252, entry_threshold=2.0, exit_threshold=0.5):
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
    
    def generate_signals(self, data):
        # Implementation here
        pass

# Bollinger Band Mean Reversion
class BollingerBandReversion(BaseStrategy):
    def __init__(self, period=20, std_dev=2.0):
        self.period = period
        self.std_dev = std_dev
    
    def generate_signals(self, data):
        # Implementation here
        pass
```

## Performance Considerations

- **Market Regimes**: Works best in ranging or sideways markets
- **Trending Markets**: Can underperform during strong trends
- **Volatility**: Performance sensitive to volatility levels
- **Correlation Stability**: Pairs trading requires stable correlations

## Risk Management

- **Stop Losses**: Essential to limit losses if mean reversion doesn't occur
- **Position Sizing**: Based on deviation magnitude and volatility
- **Correlation Monitoring**: Watch for breakdown in asset relationships
- **Market Regime Filters**: Avoid trading during strong trends

## Statistical Requirements

- **Cointegration Testing**: For pairs trading strategies
- **Stationarity**: Ensure time series are stationary
- **Normality**: Many strategies assume normal distribution of returns
- **Autocorrelation**: Check for serial correlation in residuals

## Backtesting Considerations

- **Look-Ahead Bias**: Avoid using future information
- **Survivorship Bias**: Include delisted or merged companies
- **Transaction Costs**: Include realistic trading costs
- **Slippage**: Account for execution delays
- **Out-of-Sample Testing**: Validate on unseen data

## Common Pitfalls

- **False Signals**: Not all deviations result in mean reversion
- **Parameter Overfitting**: Avoid over-optimizing parameters
- **Market Regime Changes**: Strategies may fail in new market conditions
- **Liquidity Issues**: Some assets may not be liquid enough for pairs trading

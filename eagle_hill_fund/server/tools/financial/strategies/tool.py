import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import warnings

from eagle_hill_fund.server.tools.financial.portfolio.tool import PortfolioTool, OrderSide, OrderType
from eagle_hill_fund.server.tools.data.parallelization.tool import ParallelizationTool


@dataclass
class Signal:
    """Represents a trading signal."""
    symbol: str
    timestamp: datetime
    signal_type: str  # 'buy', 'sell', 'hold'
    strength: float = 1.0  # Signal strength (0-1)
    price: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class StrategyConfig:
    """Configuration for strategy execution."""
    name: str = "BaseStrategy"
    symbols: List[str] = field(default_factory=list)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 1000000.0
    commission_per_trade: float = 0.0
    position_size_pct: float = 0.1  # 10% of portfolio per position
    max_positions: int = 10
    rebalance_frequency: str = "daily"  # daily, weekly, monthly
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    lookback_period: int = 20
    min_volume: float = 1000.0
    min_price: float = 1.0
    max_price: float = 10000.0


class BaseStrategyTool(ABC):
    """
    Base class for strategy execution and backtesting.
    
    This class provides a framework for implementing trading strategies
    that can be backtested on historical data.
    
    Expected DataFrame columns:
    - symbol: Stock symbol
    - date: Date/timestamp
    - open: Opening price
    - high: High price
    - low: Low price
    - close: Closing price
    - volume: Trading volume
    - change: Price change
    - changePercent: Price change percentage
    - vwap: Volume weighted average price
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the strategy.
        
        Args:
            config: Strategy configuration
        """
        self.config = config
        self.portfolio = PortfolioTool(
            initial_cash=config.initial_capital,
            commission_per_trade=config.commission_per_trade,
            name=config.name
        )
        
        # Data storage
        self.data: Optional[pd.DataFrame] = None
        self.signals: List[Signal] = []
        self.indicators: Dict[str, pd.DataFrame] = {}
        
        # Performance tracking
        self.backtest_results: Optional[Dict] = None
        self.execution_log: List[Dict] = []
        
        # Strategy state
        self.current_date: Optional[datetime] = None
        self.current_prices: Dict[str, float] = {}
        self.position_targets: Dict[str, float] = {}
        
    def set_data(self, data: pd.DataFrame) -> None:
        """
        Set the historical data for backtesting.
        
        Args:
            data: DataFrame with historical price data
        """
        # Validate required columns
        required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume', 'change', 'changePercent', 'vwap']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
        
        # Sort by symbol and date
        self.data = data.sort_values(['symbol', 'date']).reset_index(drop=True)
        
        # Filter by date range if specified
        if self.config.start_date:
            self.data = self.data[self.data['date'] >= self.config.start_date]
        if self.config.end_date:
            self.data = self.data[self.data['date'] <= self.config.end_date]
        
        # Filter by symbols if specified
        if self.config.symbols:
            self.data = self.data[self.data['symbol'].isin(self.config.symbols)]
        
        print(f"Data loaded: {len(self.data)} rows, {self.data['symbol'].nunique()} symbols")
        print(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")
    
    def calculate_indicators(self) -> None:
        """
        Calculate technical indicators for all symbols using parallel processing.
        Override this method to implement custom indicators.
        """
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        # Get unique symbols
        symbols = self.data['symbol'].unique().tolist()
        
        # Initialize parallelization tool
        parallel_tool = ParallelizationTool()
        
        # Define the function to calculate indicators for a single symbol
        def calculate_symbol_indicators(symbol):
            symbol_data = self.data[self.data['symbol'] == symbol].copy()
            indicators = self._calculate_basic_indicators(symbol_data)
            return symbol, indicators
        
        # Calculate indicators in parallel
        print(f"Calculating indicators for {len(symbols)} symbols in parallel...")
        results = parallel_tool.smart_parallel_map(calculate_symbol_indicators, symbols)
        
        # Store results
        for symbol, indicators in results:
            self.indicators[symbol] = indicators
        
        print(f"Indicators calculated for {len(self.indicators)} symbols")
    
    def _calculate_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate basic technical indicators.
        
        Args:
            data: DataFrame with price data for a single symbol
            
        Returns:
            DataFrame with calculated indicators
        """
        df = data.copy()
        
        # Price-based indicators
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Price momentum
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)
        
        # Volatility
        df['volatility_20'] = df['close'].rolling(window=20).std()
        df['atr'] = self._calculate_atr(df)
        
        return df
    
    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    @abstractmethod
    def generate_signals(self, date: datetime) -> List[Signal]:
        """
        Generate trading signals for a given date.
        This method must be implemented by subclasses.
        
        Args:
            date: Current date for signal generation
            
        Returns:
            List of trading signals
        """
        pass
    
    def calculate_position_size(self, symbol: str, price: float, signal_strength: float = 1.0) -> float:
        """
        Calculate position size based on signal strength and risk parameters.
        
        Args:
            symbol: Stock symbol
            price: Current price
            signal_strength: Strength of the signal (0-1)
            
        Returns:
            Number of shares to trade
        """
        # Base position size as percentage of portfolio
        portfolio_value = self.portfolio.get_total_value(self.current_prices)
        base_position_value = portfolio_value * self.config.position_size_pct
        
        # Adjust for signal strength
        position_value = base_position_value * signal_strength
        
        # Adjust for number of positions (diversification)
        current_positions = len(self.portfolio.get_positions())
        if current_positions > 0:
            max_positions = min(self.config.max_positions, len(self.config.symbols) if self.config.symbols else 10)
            position_value *= (1.0 / max_positions)
        
        # Calculate number of shares
        shares = position_value / price
        
        return max(0, shares)
    
    def execute_signals(self, signals: List[Signal]) -> List[Dict]:
        """
        Execute trading signals.
        
        Args:
            signals: List of trading signals to execute
            
        Returns:
            List of execution results
        """
        executions = []
        
        for signal in signals:
            try:
                # Check if symbol meets basic criteria
                if not self._validate_symbol(signal.symbol, signal.price):
                    continue
                
                # Calculate position size
                if signal.signal_type == 'buy':
                    quantity = self.calculate_position_size(signal.symbol, signal.price, signal.strength)
                    if quantity > 0:
                        trade = self.portfolio.execute_trade(
                            symbol=signal.symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            price=signal.price,
                            timestamp=self.current_date
                        )
                        executions.append({
                            'timestamp': self.current_date,
                            'symbol': signal.symbol,
                            'action': 'buy',
                            'quantity': quantity,
                            'price': signal.price,
                            'trade_id': trade.trade_id
                        })
                
                elif signal.signal_type == 'sell':
                    position = self.portfolio.get_position(signal.symbol)
                    if position and position.quantity > 0:
                        trade = self.portfolio.execute_trade(
                            symbol=signal.symbol,
                            side=OrderSide.SELL,
                            quantity=position.quantity,
                            price=signal.price,
                            timestamp=self.current_date
                        )
                        executions.append({
                            'timestamp': self.current_date,
                            'symbol': signal.symbol,
                            'action': 'sell',
                            'quantity': position.quantity,
                            'price': signal.price,
                            'trade_id': trade.trade_id
                        })
                
            except Exception as e:
                print(f"Error executing signal for {signal.symbol}: {e}")
                executions.append({
                    'timestamp': self.current_date,
                    'symbol': signal.symbol,
                    'action': 'error',
                    'error': str(e)
                })
        
        return executions
    
    def _validate_symbol(self, symbol: str, price: float) -> bool:
        """
        Validate if a symbol meets basic trading criteria.
        
        Args:
            symbol: Stock symbol
            price: Current price
            
        Returns:
            True if symbol is valid for trading
        """
        # Check price range
        if price < self.config.min_price or price > self.config.max_price:
            return False
        
        # Check if we have data for this symbol
        if symbol not in self.indicators:
            return False
        
        return True
    
    def run_backtest(self) -> Dict:
        """
        Run the backtest strategy with parallel processing for signal generation.
        
        Returns:
            Dictionary containing backtest results
        """
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")
        
        print(f"Starting backtest for {self.config.name}")
        print(f"Period: {self.data['date'].min()} to {self.data['date'].max()}")
        print(f"Symbols: {self.data['symbol'].nunique()}")
        
        # Calculate indicators (already parallelized)
        self.calculate_indicators()
        
        # Get unique dates
        dates = sorted(self.data['date'].unique())
        
        # Initialize portfolio
        self.portfolio.set_current_timestamp(dates[0])
        
        # Initialize parallelization tool
        parallel_tool = ParallelizationTool()
        
        # Run backtest with parallel signal generation
        for i, date in enumerate(dates):
            self.current_date = date
            
            # Get current prices for all symbols
            current_data = self.data[self.data['date'] == date]
            self.current_prices = dict(zip(current_data['symbol'], current_data['close']))
            
            # Generate signals (parallelized for large symbol counts)
            if len(self.config.symbols) > 100:
                signals = self._generate_signals_parallel(date, parallel_tool)
            else:
                signals = self.generate_signals(date)
            
            # Execute signals
            executions = self.execute_signals(signals)
            self.execution_log.extend(executions)
            
            # Take portfolio snapshot
            self.portfolio.take_snapshot(date, self.current_prices)
            
            # Progress update
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(dates)} dates ({date})")
        
        # Generate results
        self.backtest_results = self.portfolio.get_backtest_results()
        self.backtest_results['strategy_name'] = self.config.name
        self.backtest_results['execution_log'] = self.execution_log
        
        print("Backtest completed!")
        return self.backtest_results
    
    def _generate_signals_parallel(self, date: datetime, parallel_tool: ParallelizationTool) -> List[Signal]:
        """
        Generate signals in parallel for large symbol counts.
        This is a fallback method when there are many symbols to process.
        """
        # Get data for the specified date
        current_data = self.data[self.data['date'] == date]
        if current_data.empty:
            return []
        
        # Create a dictionary for easy lookup by symbol
        data_dict = current_data.set_index('symbol')
        
        # Filter symbols that have data for this date
        available_symbols = [symbol for symbol in self.config.symbols if symbol in data_dict.index]
        
        # Define function to generate signals for a batch of symbols
        def generate_signals_batch(symbol_batch):
            batch_signals = []
            for symbol in symbol_batch:
                if symbol not in data_dict.index:
                    continue
                
                # Get basic data
                row = data_dict.loc[symbol]
                
                # Get indicators for this symbol
                if symbol not in self.indicators:
                    continue
                    
                symbol_indicators = self.indicators[symbol]
                symbol_data_for_date = symbol_indicators[symbol_indicators['date'] == date]
                
                if symbol_data_for_date.empty:
                    continue
                    
                indicator_row = symbol_data_for_date.iloc[0]
                
                # Generate signals for this symbol (simplified version)
                # This is a basic implementation - subclasses should override for full functionality
                try:
                    signals = self._generate_single_symbol_signals(symbol, row, indicator_row, date)
                    batch_signals.extend(signals)
                except Exception as e:
                    # Skip symbols that cause errors
                    continue
            
            return batch_signals
        
        # Split symbols into batches for parallel processing
        batch_size = max(1, len(available_symbols) // parallel_tool.max_workers)
        symbol_batches = [available_symbols[i:i + batch_size] for i in range(0, len(available_symbols), batch_size)]
        
        # Process batches in parallel
        batch_results = parallel_tool.parallel_map(generate_signals_batch, symbol_batches)
        
        # Flatten results
        all_signals = []
        for batch_signals in batch_results:
            all_signals.extend(batch_signals)
        
        return all_signals
    
    def _generate_single_symbol_signals(self, symbol: str, row: pd.Series, indicator_row: pd.Series, date: datetime) -> List[Signal]:
        """
        Generate signals for a single symbol. This is a simplified version
        that subclasses can override for more complex logic.
        """
        signals = []
        
        # Check for position exits first
        position = self.portfolio.get_position(symbol)
        if position and position.quantity != 0:
            # Basic exit logic - subclasses should override
            if hasattr(self, '_should_exit_position'):
                if self._should_exit_position(symbol, indicator_row):
                    signals.append(Signal(
                        symbol=symbol,
                        timestamp=date,
                        signal_type='sell' if position.quantity > 0 else 'buy',
                        strength=1.0,
                        price=row['close'],
                        metadata={'reason': 'exit'}
                    ))
        
        # Check for new entries
        current_positions = len(self.portfolio.get_positions())
        if current_positions < self.config.max_positions:
            # Basic entry logic - subclasses should override
            if hasattr(self, '_is_valid_entry') and hasattr(self, '_should_enter_position'):
                if self._is_valid_entry(indicator_row, 'long') and self._should_enter_position(symbol, indicator_row):
                    signals.append(Signal(
                        symbol=symbol,
                        timestamp=date,
                        signal_type='buy',
                        strength=1.0,
                        price=row['close'],
                        metadata={'reason': 'entry'}
                    ))
        
        return signals
    
    def get_performance_summary(self) -> Dict:
        """
        Get a summary of strategy performance.
        
        Returns:
            Dictionary containing performance summary
        """
        if self.backtest_results is None:
            return {"error": "No backtest results available"}
        
        performance = self.backtest_results.get('performance', {})
        summary = self.backtest_results.get('summary', {})
        
        return {
            "strategy_name": self.config.name,
            "total_return": performance.get('total_return', 0),
            "annualized_return": performance.get('annualized_return', 0),
            "volatility": performance.get('volatility', 0),
            "sharpe_ratio": performance.get('sharpe_ratio', 0),
            "max_drawdown": performance.get('max_drawdown', 0),
            "win_rate": performance.get('win_rate', 0),
            "total_trades": summary.get('num_trades', 0),
            "final_value": summary.get('total_value', 0),
            "initial_capital": self.config.initial_capital
        }
    
    def export_results(self, filepath: str) -> None:
        """
        Export backtest results to file.
        
        Args:
            filepath: Path to save results
        """
        if self.backtest_results is None:
            raise ValueError("No backtest results to export")
        
        import json
        
        # Convert datetime objects to strings for JSON serialization
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        # Create exportable results
        export_results = {}
        for key, value in self.backtest_results.items():
            if key == 'time_series':
                export_results[key] = [
                    {k: convert_datetime(v) if isinstance(v, datetime) else v 
                     for k, v in item.items()} 
                    for item in value
                ]
            else:
                export_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(export_results, f, indent=2, default=convert_datetime)
        
        print(f"Results exported to {filepath}")

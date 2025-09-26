import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Trade:
    """Represents a single trade execution."""
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    timestamp: datetime
    order_type: OrderType = OrderType.MARKET
    commission: float = 0.0
    trade_id: str = field(default_factory=lambda: f"trade_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}")
    
    @property
    def notional_value(self) -> float:
        """Calculate the notional value of the trade."""
        return self.quantity * self.price
    
    @property
    def total_cost(self) -> float:
        """Calculate total cost including commission."""
        return self.notional_value + self.commission


@dataclass
class Position:
    """Represents a position in a security."""
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    first_trade_date: Optional[datetime] = None
    last_trade_date: Optional[datetime] = None
    
    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.average_price
    
    @property
    def total_pnl(self) -> float:
        """Total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class PortfolioSnapshot:
    """Snapshot of portfolio state at a point in time."""
    timestamp: datetime
    total_value: float
    cash_balance: float
    positions_value: float
    total_pnl: float
    daily_return: float = 0.0
    positions: Dict[str, Position] = field(default_factory=dict)


class PortfolioTool:
    """
    A comprehensive portfolio management tool for backtesting and live trading.
    
    Features:
    - Trade execution and tracking
    - Position management
    - Cash balance tracking
    - Performance metrics calculation
    - Risk management
    - Backtesting capabilities
    - Portfolio snapshots and history
    """
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_per_trade: float = 0.0,
        commission_per_share: float = 0.0,
        min_commission: float = 0.0,
        max_commission: float = float('inf'),
        name: str = "Portfolio"
    ):
        """
        Initialize the portfolio.
        
        Args:
            initial_cash: Starting cash balance
            commission_per_trade: Fixed commission per trade
            commission_per_share: Commission per share traded
            min_commission: Minimum commission per trade
            max_commission: Maximum commission per trade
            name: Portfolio name
        """
        self.name = name
        self.initial_cash = initial_cash
        self.cash_balance = initial_cash
        self.commission_per_trade = commission_per_trade
        self.commission_per_share = commission_per_share
        self.min_commission = min_commission
        self.max_commission = max_commission
        
        # Data storage
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.snapshots: List[PortfolioSnapshot] = []
        self.current_timestamp: Optional[datetime] = None
        
        # Performance tracking
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        
        # Risk management
        self.max_position_size: Optional[float] = None
        self.max_portfolio_risk: Optional[float] = None
        self.stop_loss_pct: Optional[float] = None
        
    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a trade."""
        commission = self.commission_per_trade + (self.commission_per_share * quantity)
        return max(self.min_commission, min(commission, self.max_commission))
    
    def _update_position(self, trade: Trade) -> None:
        """Update position after a trade execution."""
        symbol = trade.symbol
        
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
        
        position = self.positions[symbol]
        
        if trade.side == OrderSide.BUY:
            # Buying - increase position
            if position.quantity == 0:
                position.average_price = trade.price
                position.quantity = trade.quantity
                position.first_trade_date = trade.timestamp
            else:
                # Calculate new average price
                total_cost = (position.quantity * position.average_price) + trade.notional_value
                total_quantity = position.quantity + trade.quantity
                position.average_price = total_cost / total_quantity
                position.quantity = total_quantity
            
            position.last_trade_date = trade.timestamp
            position.total_commission += trade.commission
            
        else:  # SELL
            # Selling - decrease position
            if position.quantity < trade.quantity:
                raise ValueError(f"Insufficient position for {symbol}. Have {position.quantity}, trying to sell {trade.quantity}")
            
            # Calculate realized P&L
            realized_pnl = (trade.price - position.average_price) * trade.quantity
            position.realized_pnl += realized_pnl
            position.quantity -= trade.quantity
            position.total_commission += trade.commission
            position.last_trade_date = trade.timestamp
            
            # Remove position if fully closed
            if position.quantity == 0:
                del self.positions[symbol]
    
    def execute_trade(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        timestamp: Optional[datetime] = None,
        order_type: OrderType = OrderType.MARKET,
        commission: Optional[float] = None
    ) -> Trade:
        """
        Execute a trade and update portfolio state.
        
        Args:
            symbol: Security symbol
            side: Buy or sell
            quantity: Number of shares
            price: Execution price
            timestamp: Trade timestamp (defaults to current time)
            order_type: Type of order
            commission: Override commission calculation
            
        Returns:
            Trade object representing the execution
        """
        if timestamp is None:
            timestamp = self.current_timestamp or datetime.now()
        
        if commission is None:
            commission = self._calculate_commission(quantity, price)
        
        # Check if we have enough cash for buy orders
        if side == OrderSide.BUY:
            total_cost = (quantity * price) + commission
            if total_cost > self.cash_balance:
                raise ValueError(f"Insufficient cash. Need ${total_cost:.2f}, have ${self.cash_balance:.2f}")
        
        # Check if we have enough shares for sell orders
        if side == OrderSide.SELL:
            if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                available = self.positions.get(symbol, Position(symbol)).quantity
                raise ValueError(f"Insufficient shares. Need {quantity}, have {available}")
        
        # Create and execute trade
        trade = Trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            timestamp=timestamp,
            order_type=order_type,
            commission=commission
        )
        
        # Update cash balance
        if side == OrderSide.BUY:
            self.cash_balance -= trade.total_cost
        else:  # SELL
            self.cash_balance += trade.notional_value - commission
        
        # Update position
        self._update_position(trade)
        
        # Record trade
        self.trades.append(trade)
        
        return trade
    
    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Trade]:
        """
        Get trade history with optional filtering.
        
        Args:
            symbol: Filter by symbol
            start_date: Filter trades after this date
            end_date: Filter trades before this date
            
        Returns:
            List of trades matching criteria
        """
        trades = self.trades
        
        if symbol:
            trades = [t for t in trades if t.symbol == symbol]
        
        if start_date:
            trades = [t for t in trades if t.timestamp >= start_date]
        
        if end_date:
            trades = [t for t in trades if t.timestamp <= end_date]
        
        return sorted(trades, key=lambda x: x.timestamp)
    
    def get_positions(self) -> Dict[str, Position]:
        """Get current positions."""
        return self.positions.copy()
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol."""
        return self.positions.get(symbol)
    
    def close_position(self, symbol: str, timestamp: Optional[datetime] = None) -> Optional[Trade]:
        """
        Close entire position for a symbol.
        
        Args:
            symbol: Symbol to close
            timestamp: Trade timestamp
            
        Returns:
            Trade object if position was closed, None if no position exists
        """
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        if position.quantity == 0:
            return None
        
        # For backtesting, we need current price - this would be provided by market data
        # For now, we'll use the average price as a placeholder
        current_price = position.average_price
        
        return self.execute_trade(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=position.quantity,
            price=current_price,
            timestamp=timestamp
        )
    
    def get_total_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            current_prices: Current market prices for positions
            
        Returns:
            Total portfolio value (cash + positions)
        """
        positions_value = 0.0
        
        for symbol, position in self.positions.items():
            if current_prices and symbol in current_prices:
                positions_value += position.quantity * current_prices[symbol]
            else:
                positions_value += position.market_value
        
        return self.cash_balance + positions_value
    
    def get_positions_value(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate total value of all positions.
        
        Args:
            current_prices: Current market prices for positions
            
        Returns:
            Total positions value
        """
        positions_value = 0.0
        
        for symbol, position in self.positions.items():
            if current_prices and symbol in current_prices:
                positions_value += position.quantity * current_prices[symbol]
            else:
                positions_value += position.market_value
        
        return positions_value
    
    def get_cash_balance(self) -> float:
        """Get current cash balance."""
        return self.cash_balance
    
    def get_position_weights(self, current_prices: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Calculate position weights as percentage of total portfolio.
        
        Args:
            current_prices: Current market prices for positions
            
        Returns:
            Dictionary of symbol -> weight percentage
        """
        total_value = self.get_total_value(current_prices)
        weights = {}
        
        for symbol, position in self.positions.items():
            if current_prices and symbol in current_prices:
                position_value = position.quantity * current_prices[symbol]
            else:
                position_value = position.market_value
            
            weights[symbol] = position_value / total_value if total_value > 0 else 0.0
        
        return weights
    
    def get_cash_weight(self, current_prices: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate cash weight as percentage of total portfolio.
        
        Args:
            current_prices: Current market prices for positions
            
        Returns:
            Cash weight percentage
        """
        total_value = self.get_total_value(current_prices)
        return self.cash_balance / total_value if total_value > 0 else 1.0
    
    def update_prices(self, current_prices: Dict[str, float]) -> None:
        """
        Update unrealized P&L for all positions based on current prices.
        
        Args:
            current_prices: Current market prices for positions
        """
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position.unrealized_pnl = (current_price - position.average_price) * position.quantity
    
    def take_snapshot(self, timestamp: Optional[datetime] = None, current_prices: Optional[Dict[str, float]] = None) -> PortfolioSnapshot:
        """
        Take a snapshot of the current portfolio state.
        
        Args:
            timestamp: Snapshot timestamp
            current_prices: Current market prices for positions
            
        Returns:
            PortfolioSnapshot object
        """
        if timestamp is None:
            timestamp = self.current_timestamp or datetime.now()
        
        # Update unrealized P&L if prices provided
        if current_prices:
            self.update_prices(current_prices)
        
        total_value = self.get_total_value(current_prices)
        positions_value = self.get_positions_value(current_prices)
        
        # Calculate total P&L
        total_pnl = 0.0
        for position in self.positions.values():
            total_pnl += position.total_pnl
        
        # Calculate daily return if we have previous snapshot
        daily_return = 0.0
        if self.snapshots:
            prev_snapshot = self.snapshots[-1]
            if prev_snapshot.total_value > 0:
                daily_return = (total_value - prev_snapshot.total_value) / prev_snapshot.total_value
        
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            total_value=total_value,
            cash_balance=self.cash_balance,
            positions_value=positions_value,
            total_pnl=total_pnl,
            daily_return=daily_return,
            positions=self.positions.copy()
        )
        
        self.snapshots.append(snapshot)
        return snapshot
    
    def get_snapshots(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[PortfolioSnapshot]:
        """
        Get portfolio snapshots with optional date filtering.
        
        Args:
            start_date: Filter snapshots after this date
            end_date: Filter snapshots before this date
            
        Returns:
            List of snapshots matching criteria
        """
        snapshots = self.snapshots
        
        if start_date:
            snapshots = [s for s in snapshots if s.timestamp >= start_date]
        
        if end_date:
            snapshots = [s for s in snapshots if s.timestamp <= end_date]
        
        return sorted(snapshots, key=lambda x: x.timestamp)
    
    def get_portfolio_summary(self, current_prices: Optional[Dict[str, float]] = None) -> Dict:
        """
        Get a comprehensive portfolio summary.
        
        Args:
            current_prices: Current market prices for positions
            
        Returns:
            Dictionary containing portfolio summary
        """
        total_value = self.get_total_value(current_prices)
        positions_value = self.get_positions_value(current_prices)
        cash_weight = self.get_cash_weight(current_prices)
        position_weights = self.get_position_weights(current_prices)
        
        # Calculate total P&L
        total_pnl = 0.0
        for position in self.positions.values():
            total_pnl += position.total_pnl
        
        # Calculate total commission paid
        total_commission = sum(trade.commission for trade in self.trades)
        
        return {
            "name": self.name,
            "total_value": total_value,
            "cash_balance": self.cash_balance,
            "positions_value": positions_value,
            "cash_weight": cash_weight,
            "position_weights": position_weights,
            "total_pnl": total_pnl,
            "total_commission": total_commission,
            "num_positions": len(self.positions),
            "num_trades": len(self.trades),
            "num_snapshots": len(self.snapshots),
            "start_date": self.start_date,
            "end_date": self.end_date
        }
    
    def calculate_returns(self, benchmark_returns: Optional[pd.Series] = None) -> Dict:
        """
        Calculate portfolio performance metrics.
        
        Args:
            benchmark_returns: Benchmark returns for comparison
            
        Returns:
            Dictionary containing performance metrics
        """
        if not self.snapshots:
            return {"error": "No snapshots available for performance calculation"}
        
        # Convert snapshots to DataFrame
        snapshot_data = []
        for snapshot in self.snapshots:
            snapshot_data.append({
                'timestamp': snapshot.timestamp,
                'total_value': snapshot.total_value,
                'daily_return': snapshot.daily_return
            })
        
        df = pd.DataFrame(snapshot_data)
        df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        returns = df['daily_return'].dropna()
        
        if len(returns) == 0:
            return {"error": "No valid returns data"}
        
        # Basic metrics
        total_return = (df['total_value'].iloc[-1] / df['total_value'].iloc[0]) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        
        # Risk metrics
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Additional metrics
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        # Benchmark comparison
        benchmark_metrics = {}
        if benchmark_returns is not None:
            # Align dates
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            if len(aligned_returns) > 0:
                excess_returns = aligned_returns - aligned_benchmark
                tracking_error = excess_returns.std() * np.sqrt(252)
                information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
                beta = aligned_returns.cov(aligned_benchmark) / aligned_benchmark.var() if aligned_benchmark.var() > 0 else 0
                alpha = annualized_return - (beta * aligned_benchmark.mean() * 252)
                
                benchmark_metrics = {
                    "tracking_error": tracking_error,
                    "information_ratio": information_ratio,
                    "beta": beta,
                    "alpha": alpha
                }
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "num_trading_days": len(returns),
            **benchmark_metrics
        }
    
    def get_performance_attribution(self) -> Dict:
        """
        Analyze performance attribution by position.
        
        Returns:
            Dictionary containing performance attribution analysis
        """
        if not self.trades:
            return {"error": "No trades available for attribution analysis"}
        
        # Group trades by symbol
        symbol_trades = {}
        for trade in self.trades:
            if trade.symbol not in symbol_trades:
                symbol_trades[trade.symbol] = []
            symbol_trades[trade.symbol].append(trade)
        
        attribution = {}
        for symbol, trades in symbol_trades.items():
            # Calculate realized P&L
            realized_pnl = 0.0
            total_commission = 0.0
            total_volume = 0.0
            
            for trade in trades:
                if trade.side == OrderSide.SELL:
                    # Find corresponding buy trades for P&L calculation
                    buy_trades = [t for t in trades if t.side == OrderSide.BUY and t.timestamp <= trade.timestamp]
                    if buy_trades:
                        avg_buy_price = sum(t.price * t.quantity for t in buy_trades) / sum(t.quantity for t in buy_trades)
                        realized_pnl += (trade.price - avg_buy_price) * trade.quantity
                
                total_commission += trade.commission
                total_volume += trade.notional_value
            
            # Get current position info
            position = self.positions.get(symbol)
            unrealized_pnl = position.unrealized_pnl if position else 0.0
            
            attribution[symbol] = {
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": realized_pnl + unrealized_pnl,
                "total_commission": total_commission,
                "total_volume": total_volume,
                "num_trades": len(trades),
                "current_position": position.quantity if position else 0.0
            }
        
        return attribution
    
    def get_trade_analysis(self) -> Dict:
        """
        Analyze trading patterns and statistics.
        
        Returns:
            Dictionary containing trade analysis
        """
        if not self.trades:
            return {"error": "No trades available for analysis"}
        
        # Convert trades to DataFrame for analysis
        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side.value,
                'quantity': trade.quantity,
                'price': trade.price,
                'notional_value': trade.notional_value,
                'commission': trade.commission
            })
        
        df = pd.DataFrame(trade_data)
        
        # Basic statistics
        total_trades = len(df)
        buy_trades = len(df[df['side'] == 'buy'])
        sell_trades = len(df[df['side'] == 'sell'])
        
        # Volume analysis
        total_volume = df['notional_value'].sum()
        avg_trade_size = df['notional_value'].mean()
        median_trade_size = df['notional_value'].median()
        
        # Commission analysis
        total_commission = df['commission'].sum()
        avg_commission = df['commission'].mean()
        commission_rate = total_commission / total_volume if total_volume > 0 else 0
        
        # Symbol analysis
        symbol_stats = df.groupby('symbol').agg({
            'notional_value': ['count', 'sum', 'mean'],
            'commission': 'sum'
        }).round(2)
        
        # Time analysis
        df['date'] = df['timestamp'].dt.date
        daily_trades = df.groupby('date').size()
        avg_daily_trades = daily_trades.mean()
        max_daily_trades = daily_trades.max()
        
        return {
            "total_trades": total_trades,
            "buy_trades": buy_trades,
            "sell_trades": sell_trades,
            "total_volume": total_volume,
            "avg_trade_size": avg_trade_size,
            "median_trade_size": median_trade_size,
            "total_commission": total_commission,
            "avg_commission": avg_commission,
            "commission_rate": commission_rate,
            "avg_daily_trades": avg_daily_trades,
            "max_daily_trades": max_daily_trades,
            "symbol_stats": symbol_stats.to_dict(),
            "unique_symbols": df['symbol'].nunique()
        }
    
    def set_current_timestamp(self, timestamp: datetime) -> None:
        """
        Set the current timestamp for backtesting.
        
        Args:
            timestamp: Current timestamp in the backtest
        """
        self.current_timestamp = timestamp
        
        if self.start_date is None:
            self.start_date = timestamp
        self.end_date = timestamp
    
    def backtest_step(
        self,
        timestamp: datetime,
        current_prices: Dict[str, float],
        take_snapshot: bool = True
    ) -> Optional[PortfolioSnapshot]:
        """
        Execute a single step in backtesting.
        
        Args:
            timestamp: Current timestamp
            current_prices: Current market prices
            take_snapshot: Whether to take a portfolio snapshot
            
        Returns:
            PortfolioSnapshot if take_snapshot is True, None otherwise
        """
        self.set_current_timestamp(timestamp)
        
        # Update unrealized P&L
        self.update_prices(current_prices)
        
        # Take snapshot if requested
        if take_snapshot:
            return self.take_snapshot(timestamp, current_prices)
        
        return None
    
    def backtest_with_data(
        self,
        price_data: pd.DataFrame,
        symbol_column: str = 'symbol',
        timestamp_column: str = 'timestamp',
        price_column: str = 'close',
        take_daily_snapshots: bool = True
    ) -> List[PortfolioSnapshot]:
        """
        Run backtest with historical price data.
        
        Args:
            price_data: DataFrame with historical price data
            symbol_column: Column name for symbols
            timestamp_column: Column name for timestamps
            price_column: Column name for prices
            take_daily_snapshots: Whether to take daily snapshots
            
        Returns:
            List of portfolio snapshots
        """
        # Ensure timestamp column is datetime
        if not pd.api.types.is_datetime64_any_dtype(price_data[timestamp_column]):
            price_data[timestamp_column] = pd.to_datetime(price_data[timestamp_column])
        
        # Sort by timestamp
        price_data = price_data.sort_values(timestamp_column)
        
        # Group by timestamp to get current prices for all symbols
        snapshots = []
        for timestamp, group in price_data.groupby(timestamp_column):
            current_prices = dict(zip(group[symbol_column], group[price_column]))
            
            snapshot = self.backtest_step(
                timestamp=timestamp,
                current_prices=current_prices,
                take_snapshot=take_daily_snapshots
            )
            
            if snapshot:
                snapshots.append(snapshot)
        
        return snapshots
    
    def get_backtest_results(self) -> Dict:
        """
        Get comprehensive backtest results.
        
        Returns:
            Dictionary containing backtest results
        """
        if not self.snapshots:
            return {"error": "No backtest data available"}
        
        # Performance metrics
        performance = self.calculate_returns()
        
        # Trade analysis
        trade_analysis = self.get_trade_analysis()
        
        # Performance attribution
        attribution = self.get_performance_attribution()
        
        # Portfolio summary
        summary = self.get_portfolio_summary()
        
        # Time series data
        time_series = []
        for snapshot in self.snapshots:
            time_series.append({
                'timestamp': snapshot.timestamp,
                'total_value': snapshot.total_value,
                'cash_balance': snapshot.cash_balance,
                'positions_value': snapshot.positions_value,
                'total_pnl': snapshot.total_pnl,
                'daily_return': snapshot.daily_return
            })
        
        return {
            "performance": performance,
            "trade_analysis": trade_analysis,
            "attribution": attribution,
            "summary": summary,
            "time_series": time_series,
            "backtest_period": {
                "start_date": self.start_date,
                "end_date": self.end_date,
                "duration_days": (self.end_date - self.start_date).days if self.start_date and self.end_date else 0
            }
        }
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """
        Export portfolio snapshots to DataFrame.
        
        Returns:
            DataFrame with portfolio time series
        """
        if not self.snapshots:
            return pd.DataFrame()
        
        data = []
        for snapshot in self.snapshots:
            row = {
                'timestamp': snapshot.timestamp,
                'total_value': snapshot.total_value,
                'cash_balance': snapshot.cash_balance,
                'positions_value': snapshot.positions_value,
                'total_pnl': snapshot.total_pnl,
                'daily_return': snapshot.daily_return
            }
            
            # Add position data
            for symbol, position in snapshot.positions.items():
                row[f'position_{symbol}'] = position.quantity
                row[f'price_{symbol}'] = position.average_price
                row[f'pnl_{symbol}'] = position.total_pnl
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_state(self, filepath: str) -> None:
        """
        Save portfolio state to file.
        
        Args:
            filepath: Path to save the state
        """
        state = {
            "name": self.name,
            "initial_cash": self.initial_cash,
            "cash_balance": self.cash_balance,
            "commission_per_trade": self.commission_per_trade,
            "commission_per_share": self.commission_per_share,
            "min_commission": self.min_commission,
            "max_commission": self.max_commission,
            "current_timestamp": self.current_timestamp.isoformat() if self.current_timestamp else None,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "positions": {
                symbol: {
                    "symbol": pos.symbol,
                    "quantity": pos.quantity,
                    "average_price": pos.average_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "realized_pnl": pos.realized_pnl,
                    "total_commission": pos.total_commission,
                    "first_trade_date": pos.first_trade_date.isoformat() if pos.first_trade_date else None,
                    "last_trade_date": pos.last_trade_date.isoformat() if pos.last_trade_date else None
                }
                for symbol, pos in self.positions.items()
            },
            "trades": [
                {
                    "symbol": trade.symbol,
                    "side": trade.side.value,
                    "quantity": trade.quantity,
                    "price": trade.price,
                    "timestamp": trade.timestamp.isoformat(),
                    "order_type": trade.order_type.value,
                    "commission": trade.commission,
                    "trade_id": trade.trade_id
                }
                for trade in self.trades
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self, filepath: str) -> None:
        """
        Load portfolio state from file.
        
        Args:
            filepath: Path to load the state from
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore basic properties
        self.name = state["name"]
        self.initial_cash = state["initial_cash"]
        self.cash_balance = state["cash_balance"]
        self.commission_per_trade = state["commission_per_trade"]
        self.commission_per_share = state["commission_per_share"]
        self.min_commission = state["min_commission"]
        self.max_commission = state["max_commission"]
        
        # Restore timestamps
        self.current_timestamp = datetime.fromisoformat(state["current_timestamp"]) if state["current_timestamp"] else None
        self.start_date = datetime.fromisoformat(state["start_date"]) if state["start_date"] else None
        self.end_date = datetime.fromisoformat(state["end_date"]) if state["end_date"] else None
        
        # Restore positions
        self.positions = {}
        for symbol, pos_data in state["positions"].items():
            position = Position(
                symbol=pos_data["symbol"],
                quantity=pos_data["quantity"],
                average_price=pos_data["average_price"],
                unrealized_pnl=pos_data["unrealized_pnl"],
                realized_pnl=pos_data["realized_pnl"],
                total_commission=pos_data["total_commission"],
                first_trade_date=datetime.fromisoformat(pos_data["first_trade_date"]) if pos_data["first_trade_date"] else None,
                last_trade_date=datetime.fromisoformat(pos_data["last_trade_date"]) if pos_data["last_trade_date"] else None
            )
            self.positions[symbol] = position
        
        # Restore trades
        self.trades = []
        for trade_data in state["trades"]:
            trade = Trade(
                symbol=trade_data["symbol"],
                side=OrderSide(trade_data["side"]),
                quantity=trade_data["quantity"],
                price=trade_data["price"],
                timestamp=datetime.fromisoformat(trade_data["timestamp"]),
                order_type=OrderType(trade_data["order_type"]),
                commission=trade_data["commission"],
                trade_id=trade_data["trade_id"]
            )
            self.trades.append(trade)
    
    def set_risk_parameters(
        self,
        max_position_size: Optional[float] = None,
        max_portfolio_risk: Optional[float] = None,
        stop_loss_pct: Optional[float] = None
    ) -> None:
        """
        Set risk management parameters.
        
        Args:
            max_position_size: Maximum position size as percentage of portfolio
            max_portfolio_risk: Maximum portfolio risk (VaR or similar)
            stop_loss_pct: Stop loss percentage for positions
        """
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_pct = stop_loss_pct
    
    def check_position_size_limit(self, symbol: str, quantity: float, price: float) -> bool:
        """
        Check if position size is within limits.
        
        Args:
            symbol: Security symbol
            quantity: Number of shares
            price: Price per share
            
        Returns:
            True if within limits, False otherwise
        """
        if self.max_position_size is None:
            return True
        
        position_value = quantity * price
        total_value = self.get_total_value()
        
        if total_value == 0:
            return True
        
        position_weight = position_value / total_value
        return position_weight <= self.max_position_size
    
    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        target_weight: float,
        current_prices: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate position size based on target weight.
        
        Args:
            symbol: Security symbol
            price: Current price
            target_weight: Target weight as percentage of portfolio
            current_prices: Current market prices
            
        Returns:
            Number of shares to buy/sell
        """
        total_value = self.get_total_value(current_prices)
        target_value = total_value * target_weight
        
        # Check if we already have a position
        current_position = self.positions.get(symbol)
        if current_position:
            current_value = current_position.quantity * price
            target_value -= current_value
        
        return target_value / price
    
    def check_stop_loss(self, symbol: str, current_price: float) -> bool:
        """
        Check if stop loss should be triggered.
        
        Args:
            symbol: Security symbol
            current_price: Current market price
            
        Returns:
            True if stop loss should be triggered, False otherwise
        """
        if self.stop_loss_pct is None or symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        if position.quantity == 0:
            return False
        
        # Calculate loss percentage
        loss_pct = (position.average_price - current_price) / position.average_price
        
        return loss_pct >= self.stop_loss_pct
    
    def execute_stop_loss(self, symbol: str, current_price: float) -> Optional[Trade]:
        """
        Execute stop loss for a position.
        
        Args:
            symbol: Security symbol
            current_price: Current market price
            
        Returns:
            Trade object if stop loss was executed, None otherwise
        """
        if not self.check_stop_loss(symbol, current_price):
            return None
        
        position = self.positions[symbol]
        return self.execute_trade(
            symbol=symbol,
            side=OrderSide.SELL,
            quantity=position.quantity,
            price=current_price,
            timestamp=self.current_timestamp
        )
    
    def rebalance_portfolio(
        self,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        rebalance_threshold: float = 0.05
    ) -> List[Trade]:
        """
        Rebalance portfolio to target weights.
        
        Args:
            target_weights: Target weights for each symbol
            current_prices: Current market prices
            rebalance_threshold: Minimum deviation to trigger rebalancing
            
        Returns:
            List of trades executed for rebalancing
        """
        trades = []
        current_weights = self.get_position_weights(current_prices)
        
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0.0)
            weight_diff = target_weight - current_weight
            
            # Only rebalance if difference exceeds threshold
            if abs(weight_diff) > rebalance_threshold:
                price = current_prices.get(symbol)
                if price is None:
                    continue
                
                # Calculate position size
                quantity = self.calculate_position_size(symbol, price, target_weight, current_prices)
                
                # Check position size limits
                if not self.check_position_size_limit(symbol, abs(quantity), price):
                    continue
                
                # Execute trade
                if quantity > 0:
                    # Buy
                    try:
                        trade = self.execute_trade(
                            symbol=symbol,
                            side=OrderSide.BUY,
                            quantity=quantity,
                            price=price,
                            timestamp=self.current_timestamp
                        )
                        trades.append(trade)
                    except ValueError as e:
                        print(f"Failed to buy {symbol}: {e}")
                
                elif quantity < 0:
                    # Sell
                    try:
                        trade = self.execute_trade(
                            symbol=symbol,
                            side=OrderSide.SELL,
                            quantity=abs(quantity),
                            price=price,
                            timestamp=self.current_timestamp
                        )
                        trades.append(trade)
                    except ValueError as e:
                        print(f"Failed to sell {symbol}: {e}")
        
        return trades
    
    def calculate_var(
        self,
        confidence_level: float = 0.05,
        lookback_days: int = 252
    ) -> float:
        """
        Calculate Value at Risk (VaR) for the portfolio.
        
        Args:
            confidence_level: Confidence level (e.g., 0.05 for 95% VaR)
            lookback_days: Number of days to look back for calculation
            
        Returns:
            VaR value
        """
        if not self.snapshots or len(self.snapshots) < 2:
            return 0.0
        
        # Get recent returns
        recent_snapshots = self.snapshots[-lookback_days:] if len(self.snapshots) > lookback_days else self.snapshots
        returns = [snapshot.daily_return for snapshot in recent_snapshots if snapshot.daily_return is not None]
        
        if len(returns) < 2:
            return 0.0
        
        # Calculate VaR
        returns_array = np.array(returns)
        var = np.percentile(returns_array, confidence_level * 100)
        
        return var
    
    def calculate_expected_shortfall(
        self,
        confidence_level: float = 0.05,
        lookback_days: int = 252
    ) -> float:
        """
        Calculate Expected Shortfall (Conditional VaR) for the portfolio.
        
        Args:
            confidence_level: Confidence level (e.g., 0.05 for 95% ES)
            lookback_days: Number of days to look back for calculation
            
        Returns:
            Expected Shortfall value
        """
        if not self.snapshots or len(self.snapshots) < 2:
            return 0.0
        
        # Get recent returns
        recent_snapshots = self.snapshots[-lookback_days:] if len(self.snapshots) > lookback_days else self.snapshots
        returns = [snapshot.daily_return for snapshot in recent_snapshots if snapshot.daily_return is not None]
        
        if len(returns) < 2:
            return 0.0
        
        # Calculate Expected Shortfall
        returns_array = np.array(returns)
        var_threshold = np.percentile(returns_array, confidence_level * 100)
        tail_returns = returns_array[returns_array <= var_threshold]
        
        if len(tail_returns) == 0:
            return 0.0
        
        return tail_returns.mean()
    
    def get_risk_metrics(self) -> Dict:
        """
        Get comprehensive risk metrics for the portfolio.
        
        Returns:
            Dictionary containing risk metrics
        """
        if not self.snapshots:
            return {"error": "No snapshots available for risk calculation"}
        
        # Calculate VaR and Expected Shortfall
        var_95 = self.calculate_var(confidence_level=0.05)
        var_99 = self.calculate_var(confidence_level=0.01)
        es_95 = self.calculate_expected_shortfall(confidence_level=0.05)
        es_99 = self.calculate_expected_shortfall(confidence_level=0.01)
        
        # Calculate position concentration
        current_weights = self.get_position_weights()
        max_position_weight = max(current_weights.values()) if current_weights else 0.0
        position_concentration = sum(w**2 for w in current_weights.values())  # Herfindahl index
        
        # Calculate leverage (if applicable)
        total_value = self.get_total_value()
        leverage = total_value / self.cash_balance if self.cash_balance > 0 else 1.0
        
        return {
            "var_95": var_95,
            "var_99": var_99,
            "expected_shortfall_95": es_95,
            "expected_shortfall_99": es_99,
            "max_position_weight": max_position_weight,
            "position_concentration": position_concentration,
            "leverage": leverage,
            "num_positions": len(self.positions),
            "cash_weight": self.get_cash_weight()
        }

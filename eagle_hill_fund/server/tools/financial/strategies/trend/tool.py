import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from eagle_hill_fund.server.tools.financial.strategies.tool import BaseStrategyTool, StrategyConfig, Signal


@dataclass
class DipBuyingConfig(StrategyConfig):
    """Configuration for Dip Buying strategy."""
    name: str = "DipBuyingStrategy"
    
    # Dip buying parameters
    dip_threshold: float = 0.05  # Buy when stock dips 5% from recent high
    lookback_period: int = 20  # Look back 20 days for recent high
    
    # Exit parameters
    profit_target: float = 0.10  # Sell when up 10% from buy price
    stop_loss: float = 0.08  # Sell when down 8% from buy price
    
    # Additional filters
    min_volume_ratio: float = 1.0  # Minimum volume ratio (1.0 = average volume)
    max_price: float = 1000.0  # Maximum stock price to consider
    min_price: float = 5.0  # Minimum stock price to consider
    
    # Risk management
    max_holding_days: int = 30  # Maximum holding period
    position_size_pct: float = 0.05  # 5% of portfolio per position


class DipBuyingTool(BaseStrategyTool):
    """
    Dip Buying Strategy
    
    This strategy buys stocks when they dip a certain percentage from recent highs,
    and sells based on profit targets and stop losses.
    """
    
    def __init__(self, config: DipBuyingConfig):
        super().__init__(config)
        self.config = config
        self.position_entry_prices = {}  # Track entry prices for each position
        
    def _calculate_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators specific to dip buying strategy.
        """
        df = super()._calculate_basic_indicators(data)
        
        # Calculate recent high over lookback period
        # Find the most recent local high (i.e., a high that is higher than both its neighbors)
        df['recent_high'] = df['high'][(df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])]
        df['recent_high'] = df['recent_high'].ffill()
        
        # Calculate dip percentage from recent high
        df['dip_pct'] = (df['close'] - df['recent_high']) / df['recent_high']
        
        # Calculate price momentum
        df['price_momentum_5'] = df['close'].pct_change(5)
        df['price_momentum_10'] = df['close'].pct_change(10)
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        return df
    
    def _is_valid_dip_entry(self, row: pd.Series) -> bool:
        """Check if conditions are met for a dip entry."""
        # Check if stock has dipped enough
        if row['dip_pct'] > -self.config.dip_threshold:
            return False
            
        # Check volume filter
        if row['volume_ratio'] < self.config.min_volume_ratio:
            return False
            
        # Check price range
        if row['close'] < self.config.min_price or row['close'] > self.config.max_price:
            return False
            
        # Check if we have valid data
        if pd.isna(row['dip_pct']) or pd.isna(row['recent_high']):
            return False
            
        return True
    
    def _should_exit_position(self, symbol: str, current_price: float, entry_price: float) -> bool:
        """Check if position should be exited based on profit/loss targets."""
        # Calculate return from entry price
        return_pct = (current_price - entry_price) / entry_price
        
        # Profit target
        if return_pct >= self.config.profit_target:
            return True
            
        # Stop loss
        if return_pct <= -self.config.stop_loss:
            return True
            
        return False
    
    def _should_enter_position(self, symbol: str, row: pd.Series) -> bool:
        """Check if we should enter a new position."""
        # Check if we already have a position
        position = self.portfolio.get_position(symbol)
        if position and position.quantity != 0:
            return False
            
        # Check if we have max positions
        current_positions = len(self.portfolio.get_positions())
        if current_positions >= self.config.max_positions:
            return False
            
        # Check dip entry conditions
        return self._is_valid_dip_entry(row)
    
    def generate_signals(self, date: datetime) -> List[Signal]:
        """
        Generate trading signals based on dip buying logic.
        """
        signals = []
        
        # Get data for the specified date
        current_data = self.data[self.data['date'] == date]
        if current_data.empty:
            return signals
            
        # Create a dictionary for easy lookup by symbol
        data_dict = current_data.set_index('symbol')
        
        for symbol in self.config.symbols:
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
            
            # Check for position exits first
            position = self.portfolio.get_position(symbol)
            if position and position.quantity != 0:
                # Get entry price for this position
                entry_price = self.position_entry_prices.get(symbol, position.average_price)
                
                if self._should_exit_position(symbol, row['close'], entry_price):
                    # Calculate return for metadata
                    return_pct = (row['close'] - entry_price) / entry_price
                    
                    signals.append(Signal(
                        symbol=symbol,
                        timestamp=date,
                        signal_type='sell',
                        strength=1.0,
                        price=row['close'],
                        metadata={
                            'reason': 'profit_target' if return_pct > 0 else 'stop_loss',
                            'entry_price': entry_price,
                            'return_pct': return_pct,
                            'dip_pct': indicator_row['dip_pct'],
                            'volume_ratio': indicator_row['volume_ratio']
                        }
                    ))
                    
                    # Remove from entry price tracking
                    if symbol in self.position_entry_prices:
                        del self.position_entry_prices[symbol]
                continue
            
            # Check for new entries
            if self._should_enter_position(symbol, indicator_row):
                signals.append(Signal(
                    symbol=symbol,
                    timestamp=date,
                    signal_type='buy',
                    strength=abs(indicator_row['dip_pct']),  # Stronger signal for bigger dips
                    price=row['close'],
                    metadata={
                        'reason': 'dip_entry',
                        'dip_pct': indicator_row['dip_pct'],
                        'recent_high': indicator_row['recent_high'],
                        'volume_ratio': indicator_row['volume_ratio'],
                        'price_momentum_5': indicator_row['price_momentum_5'],
                        'price_momentum_10': indicator_row['price_momentum_10']
                    }
                ))
                
                # Track entry price
                self.position_entry_prices[symbol] = row['close']
        
        return signals
    
    def _generate_single_symbol_signals(self, symbol: str, row: pd.Series, indicator_row: pd.Series, date: datetime) -> List[Signal]:
        """
        Generate dip buying signals for a single symbol.
        Override the base class method with dip buying specific logic.
        """
        signals = []
        
        # Check for position exits first
        position = self.portfolio.get_position(symbol)
        if position and position.quantity != 0:
            # Get entry price for this position
            entry_price = self.position_entry_prices.get(symbol, position.average_price)
            
            if self._should_exit_position(symbol, row['close'], entry_price):
                # Calculate return for metadata
                return_pct = (row['close'] - entry_price) / entry_price
                
                signals.append(Signal(
                    symbol=symbol,
                    timestamp=date,
                    signal_type='sell',
                    strength=1.0,
                    price=row['close'],
                    metadata={
                        'reason': 'profit_target' if return_pct > 0 else 'stop_loss',
                        'entry_price': entry_price,
                        'return_pct': return_pct,
                        'dip_pct': indicator_row['dip_pct'],
                        'volume_ratio': indicator_row['volume_ratio']
                    }
                ))
                
                # Remove from entry price tracking
                if symbol in self.position_entry_prices:
                    del self.position_entry_prices[symbol]
        
        # Check for new entries
        if self._should_enter_position(symbol, indicator_row):
            signals.append(Signal(
                symbol=symbol,
                timestamp=date,
                signal_type='buy',
                strength=abs(indicator_row['dip_pct']),  # Stronger signal for bigger dips
                price=row['close'],
                metadata={
                    'reason': 'dip_entry',
                    'dip_pct': indicator_row['dip_pct'],
                    'recent_high': indicator_row['recent_high'],
                    'volume_ratio': indicator_row['volume_ratio'],
                    'price_momentum_5': indicator_row['price_momentum_5'],
                    'price_momentum_10': indicator_row['price_momentum_10']
                }
            ))
            
            # Track entry price
            self.position_entry_prices[symbol] = row['close']
        
        return signals
    
    def get_strategy_info(self) -> Dict:
        """Return strategy information and current state."""
        return {
            'name': self.config.name,
            'type': 'dip_buying',
            'parameters': {
                'dip_threshold': self.config.dip_threshold,
                'lookback_period': self.config.lookback_period,
                'profit_target': self.config.profit_target,
                'stop_loss': self.config.stop_loss,
                'min_volume_ratio': self.config.min_volume_ratio,
                'max_price': self.config.max_price,
                'min_price': self.config.min_price,
                'max_holding_days': self.config.max_holding_days,
                'position_size_pct': self.config.position_size_pct
            },
            'current_positions': len(self.portfolio.get_positions()),
            'max_positions': self.config.max_positions,
            'tracked_entry_prices': len(self.position_entry_prices)
        }
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from eagle_hill_fund.server.tools.financial.strategies.tool import BaseStrategyTool, StrategyConfig, Signal


@dataclass
class BollingerBandConfig(StrategyConfig):
    """Configuration for Bollinger Band mean reversion strategy."""
    name: str = "BollingerBandMeanReversion"
    
    # Bollinger Band parameters
    bb_period: int = 20  # Period for moving average
    bb_std_dev: float = 2.0  # Standard deviation multiplier
    
    # Entry/Exit thresholds
    oversold_threshold: float = 0.1  # Buy when bb_position <= 0.1
    overbought_threshold: float = 0.9  # Sell when bb_position >= 0.9
    exit_threshold: float = 0.5  # Exit when bb_position crosses 0.5
    
    # Additional filters
    min_bb_width: float = 0.02  # Minimum band width (2% of price)
    max_bb_width: float = 0.15  # Maximum band width (15% of price)
    volume_filter: bool = True  # Require above-average volume
    volume_threshold: float = 1.2  # Volume must be 1.2x average
    
    # Risk management
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.03  # 3% take profit
    max_holding_days: int = 10  # Maximum holding period


class BollingerBandMeanReversionTool(BaseStrategyTool):
    """
    Bollinger Band Mean Reversion Strategy
    
    This strategy identifies oversold and overbought conditions using Bollinger Bands
    and takes contrarian positions expecting mean reversion.
    """
    
    def __init__(self, config: BollingerBandConfig):
        super().__init__(config)
        self.config = config
        
    def _calculate_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate indicators with custom Bollinger Band parameters.
        Override base class method to use configurable parameters.
        """
        df = super()._calculate_basic_indicators(data)
        
        # Recalculate Bollinger Bands with custom parameters
        df['bb_middle'] = df['close'].rolling(window=self.config.bb_period).mean()
        bb_std = df['close'].rolling(window=self.config.bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * self.config.bb_std_dev)
        df['bb_lower'] = df['bb_middle'] - (bb_std * self.config.bb_std_dev)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Additional mean reversion indicators
        df['bb_squeeze'] = df['bb_width'] < self.config.min_bb_width
        df['bb_expansion'] = df['bb_width'] > self.config.max_bb_width
        
        return df
    
    def _is_valid_entry(self, row: pd.Series, signal_type: str) -> bool:
        """Check if entry conditions are met."""
        # Basic data quality checks
        if pd.isna(row['bb_position']) or pd.isna(row['bb_width']):
            return False
            
        # Volume filter
        if self.config.volume_filter and row['volume_ratio'] < self.config.volume_threshold:
            return False
            
        # Band width filter (avoid trading in very tight or very wide bands)
        if row['bb_width'] < self.config.min_bb_width or row['bb_width'] > self.config.max_bb_width:
            return False
            
        # Price and volume filters from base config
        if row['close'] < self.config.min_price or row['close'] > self.config.max_price:
            return False
            
        if row['volume'] < self.config.min_volume:
            return False
            
        return True
    
    def _should_exit_position(self, symbol: str, current_data: pd.Series) -> bool:
        """Check if position should be exited."""
        position = self.portfolio.get_position(symbol)
        if not position or position.quantity == 0:
            return False
            
        # Time-based exit (if we track entry date in portfolio)
        # Note: This would require extending the portfolio to track entry dates
        # For now, we'll rely on the base class stop_loss/take_profit
        
        # Bollinger Band position exit
        if position.quantity > 0:  # Long position
            if current_data['bb_position'] >= self.config.exit_threshold:
                return True
        elif position.quantity < 0:  # Short position
            if current_data['bb_position'] <= self.config.exit_threshold:
                return True
                
        return False
    
    def generate_signals(self, date: datetime) -> List[Signal]:
        """
        Generate trading signals based on Bollinger Band mean reversion logic.
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
                if self._should_exit_position(symbol, indicator_row):
                    signals.append(Signal(
                        symbol=symbol,
                        timestamp=date,
                        signal_type='sell' if position.quantity > 0 else 'buy',
                        strength=1.0,
                        price=row['close'],
                        metadata={
                            'reason': 'bollinger_exit',
                            'bb_position': indicator_row['bb_position'],
                            'bb_width': indicator_row['bb_width']
                        }
                    ))
                continue
            
            # Check for new entries (only if we don't have max positions)
            current_positions = len(self.portfolio.get_positions())
            if current_positions >= self.config.max_positions:
                continue
                
            # Long entry: oversold condition
            if (indicator_row['bb_position'] <= self.config.oversold_threshold and 
                self._is_valid_entry(indicator_row, 'long')):
                
                signals.append(Signal(
                    symbol=symbol,
                    timestamp=date,
                    signal_type='buy',
                    strength=1.0 - indicator_row['bb_position'],  # Stronger signal when more oversold
                    price=row['close'],
                    metadata={
                        'reason': 'bollinger_oversold',
                        'bb_position': indicator_row['bb_position'],
                        'bb_width': indicator_row['bb_width'],
                        'volume_ratio': indicator_row['volume_ratio']
                    }
                ))
                
                # Position will be tracked by the portfolio after execution
            
            # Short entry: overbought condition
            elif (indicator_row['bb_position'] >= self.config.overbought_threshold and 
                  self._is_valid_entry(indicator_row, 'short')):
                
                signals.append(Signal(
                    symbol=symbol,
                    timestamp=date,
                    signal_type='sell',
                    strength=indicator_row['bb_position'],  # Stronger signal when more overbought
                    price=row['close'],
                    metadata={
                        'reason': 'bollinger_overbought',
                        'bb_position': indicator_row['bb_position'],
                        'bb_width': indicator_row['bb_width'],
                        'volume_ratio': indicator_row['volume_ratio']
                    }
                ))
                
                # Position will be tracked by the portfolio after execution
        
        return signals
    
    def _generate_single_symbol_signals(self, symbol: str, row: pd.Series, indicator_row: pd.Series, date: datetime) -> List[Signal]:
        """
        Generate Bollinger Band signals for a single symbol.
        Override the base class method with Bollinger Band specific logic.
        """
        signals = []
        
        # Check for position exits first
        position = self.portfolio.get_position(symbol)
        if position and position.quantity != 0:
            if self._should_exit_position(symbol, indicator_row):
                signals.append(Signal(
                    symbol=symbol,
                    timestamp=date,
                    signal_type='sell' if position.quantity > 0 else 'buy',
                    strength=1.0,
                    price=row['close'],
                    metadata={
                        'reason': 'bollinger_exit',
                        'bb_position': indicator_row['bb_position'],
                        'bb_width': indicator_row['bb_width']
                    }
                ))
        
        # Check for new entries (only if we don't have max positions)
        current_positions = len(self.portfolio.get_positions())
        if current_positions < self.config.max_positions:
            # Long entry: oversold condition
            if (indicator_row['bb_position'] <= self.config.oversold_threshold and 
                self._is_valid_entry(indicator_row, 'long')):
                
                signals.append(Signal(
                    symbol=symbol,
                    timestamp=date,
                    signal_type='buy',
                    strength=1.0 - indicator_row['bb_position'],  # Stronger signal when more oversold
                    price=row['close'],
                    metadata={
                        'reason': 'bollinger_oversold',
                        'bb_position': indicator_row['bb_position'],
                        'bb_width': indicator_row['bb_width'],
                        'volume_ratio': indicator_row['volume_ratio']
                    }
                ))
            
            # Short entry: overbought condition
            elif (indicator_row['bb_position'] >= self.config.overbought_threshold and 
                  self._is_valid_entry(indicator_row, 'short')):
                
                signals.append(Signal(
                    symbol=symbol,
                    timestamp=date,
                    signal_type='sell',
                    strength=indicator_row['bb_position'],  # Stronger signal when more overbought
                    price=row['close'],
                    metadata={
                        'reason': 'bollinger_overbought',
                        'bb_position': indicator_row['bb_position'],
                        'bb_width': indicator_row['bb_width'],
                        'volume_ratio': indicator_row['volume_ratio']
                    }
                ))
        
        return signals
    
    def get_strategy_info(self) -> Dict:
        """Return strategy information and current state."""
        return {
            'name': self.config.name,
            'type': 'mean_reversion',
            'parameters': {
                'bb_period': self.config.bb_period,
                'bb_std_dev': self.config.bb_std_dev,
                'oversold_threshold': self.config.oversold_threshold,
                'overbought_threshold': self.config.overbought_threshold,
                'exit_threshold': self.config.exit_threshold,
                'min_bb_width': self.config.min_bb_width,
                'max_bb_width': self.config.max_bb_width,
                'volume_filter': self.config.volume_filter,
                'volume_threshold': self.config.volume_threshold,
                'stop_loss_pct': self.config.stop_loss_pct,
                'take_profit_pct': self.config.take_profit_pct,
                'max_holding_days': self.config.max_holding_days
            },
            'current_positions': len(self.portfolio.get_positions()),
            'max_positions': self.config.max_positions
        }
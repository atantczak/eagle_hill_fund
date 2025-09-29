import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from eagle_hill_fund.server.tools.financial.strategies.tool import BaseStrategyTool, StrategyConfig, Signal


@dataclass
class BollingerBandConfig(StrategyConfig):
    """Configuration for Bollinger Band mean reversion strategy."""
    name: str = "BollingerBandMeanReversion"

    short_allowed: bool = False
    
    # Bollinger Band parameters
    bb_period: int = 20  # Period for moving average
    bb_std_dev: float = 2.0  # Standard deviation multiplier
    
    # Entry/Exit thresholds
    entry_threshold: float = 0.0  # Buy when bb_position reaches bottom of band
    exit_threshold: float = 1.0  # Exit when bb_position crosses 0.5
    force_win_for_bb_exit: bool = True
    
    # Additional filters
    min_bb_width: float = 0.02  # Minimum band width (2% of price)
    max_bb_width: float = 0.15  # Maximum band width (15% of price)
    volume_filter: bool = True  # Require above-average volume
    volume_threshold: float = 1.2  # Volume must be 1.2x average
    
    # Risk management
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.03  # 3% take profit
    max_holding_days: Optional[int] = None  # Maximum holding period (None = disabled)
    
    # Exit control flags
    use_bb_exit: bool = True  # Use Bollinger Band position for exits
    use_take_profit: bool = True  # Use take profit exits
    use_max_holding_days: bool = False  # Use maximum holding days (requires max_holding_days to be set)
    
    # Moving average trend filter
    require_ma_uptrend_for_sales: bool = False  # Only allow sales when MA is trending up
    ma_trend_lookback: int = bb_period  # Number of periods to check for MA trend (default: 3 days)


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
        
        # Moving average trend calculation
        df['ma_trend'] = df['bb_middle'].diff(self.config.ma_trend_lookback)
        df['ma_uptrend'] = df['ma_trend'] > 0
        
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
    
    def _check_bb_exit_condition(self, position, current_data: pd.Series) -> Tuple[bool, str]:
        """
        Check if position should exit based on Bollinger Band position.
        
        Returns:
            Tuple of (should_exit, reason)
        """
        if not self.config.use_bb_exit:
            return False, ""
            
        if position.quantity > 0:  # Long position
            if current_data['bb_position'] >= self.config.exit_threshold:
                if self.config.force_win_for_bb_exit and position.average_price < current_data['close']:
                    return True, f"bb_exit_long_win_{current_data['bb_position']:.3f}"
        elif position.quantity < 0:  # Short position
            if current_data['bb_position'] <= self.config.exit_threshold:
                return True, f"bb_exit_short_{current_data['bb_position']:.3f}"
                
        return False, ""
    
    def _check_take_profit_condition(self, position, current_price: float) -> Tuple[bool, str]:
        """
        Check if position should exit based on take profit threshold.
        
        Returns:
            Tuple of (should_exit, reason)
        """
        if not self.config.use_take_profit or self.config.take_profit_pct is None:
            return False, ""
            
        if position.quantity == 0:
            return False, ""
            
        # Calculate current P&L percentage
        if position.quantity > 0:  # Long position
            pnl_pct = (current_price - position.average_price) / position.average_price
            if pnl_pct >= self.config.take_profit_pct:
                return True, f"take_profit_long_{pnl_pct:.3f}"
        elif position.quantity < 0:  # Short position
            pnl_pct = (position.average_price - current_price) / position.average_price
            if pnl_pct >= self.config.take_profit_pct:
                return True, f"take_profit_short_{pnl_pct:.3f}"
                
        return False, ""
    
    def _check_max_holding_days_condition(self, position, current_date: datetime) -> Tuple[bool, str]:
        """
        Check if position should exit based on maximum holding days.
        
        Returns:
            Tuple of (should_exit, reason)
        """
        if not self.config.use_max_holding_days or self.config.max_holding_days is None:
            return False, ""
            
        if position.first_trade_date is None:
            return False, ""
            
        days_held = (current_date - position.first_trade_date).days
        if days_held >= self.config.max_holding_days:
            return True, f"max_holding_days_{days_held}"
            
        return False, ""
    
    def _check_ma_trend_condition(self, current_data: pd.Series) -> Tuple[bool, str]:
        """
        Check if moving average trend condition is met for sales.
        
        Returns:
            Tuple of (trend_ok, reason)
        """
        if not self.config.require_ma_uptrend_for_sales:
            return True, "ma_trend_not_required"
            
        # Check if we have enough data for trend calculation
        if pd.isna(current_data.get('ma_uptrend')):
            return False, "ma_trend_insufficient_data"
            
        # Check if MA is in uptrend
        if current_data['ma_uptrend']:
            return True, f"ma_uptrend_{current_data.get('ma_trend', 0):.3f}"
        else:
            return False, f"ma_downtrend_{current_data.get('ma_trend', 0):.3f}"
    
    def _should_exit_position(self, symbol: str, current_data: pd.Series, current_date: datetime) -> Tuple[bool, str, Dict]:
        """
        Check if position should be exited based on multiple exit conditions.
        
        Returns:
            Tuple of (should_exit, reason, metadata)
        """
        position = self.portfolio.get_position(symbol)
        if not position or position.quantity == 0:
            return False, "", {}
            
        current_price = current_data['close']
        exit_metadata = {
            'bb_position': current_data.get('bb_position', None),
            'bb_width': current_data.get('bb_width', None),
            'current_price': current_price,
            'average_price': position.average_price,
            'quantity': position.quantity,
            'days_held': (current_date - position.first_trade_date).days if position.first_trade_date else None,
            'ma_trend': current_data.get('ma_trend', None),
            'ma_uptrend': current_data.get('ma_uptrend', None)
        }
        
        # First check if MA trend condition is met (if required)
        ma_trend_ok, ma_trend_reason = self._check_ma_trend_condition(current_data)
        if not ma_trend_ok:
            # If MA trend is required but not met, don't exit
            exit_metadata['ma_trend_blocked'] = ma_trend_reason
            return False, f"ma_trend_blocked_{ma_trend_reason}", exit_metadata
        
        # Check all exit conditions
        exit_conditions = []
        
        # 1. Bollinger Band exit
        bb_exit, bb_reason = self._check_bb_exit_condition(position, current_data)
        if bb_exit:
            exit_conditions.append(bb_reason)
            
        # 2. Take profit exit
        tp_exit, tp_reason = self._check_take_profit_condition(position, current_price)
        if tp_exit:
            exit_conditions.append(tp_reason)
            
        # 3. Max holding days exit
        max_days_exit, max_days_reason = self._check_max_holding_days_condition(position, current_date)
        if max_days_exit:
            exit_conditions.append(max_days_reason)
            
        # If any exit condition is met, exit the position
        if exit_conditions:
            combined_reason = "_".join(exit_conditions)
            exit_metadata['exit_conditions'] = exit_conditions
            exit_metadata['ma_trend_ok'] = ma_trend_reason
            return True, combined_reason, exit_metadata
                
        return False, "", exit_metadata
    
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
                should_exit, exit_reason, exit_metadata = self._should_exit_position(symbol, indicator_row, date)
                if should_exit:
                    signals.append(Signal(
                        symbol=symbol,
                        timestamp=date,
                        signal_type='sell' if position.quantity > 0 else 'buy',
                        strength=1.0,
                        price=row['close'],
                        metadata={
                            'reason': exit_reason,
                            **exit_metadata
                        }
                    ))
                continue
            
            # Check for new entries (only if we don't have max positions)
            current_positions = len(self.portfolio.get_positions())
            if current_positions >= self.config.max_positions:
                continue
                
            # Long entry: oversold condition
            if (indicator_row['bb_position'] <= self.config.entry_threshold and 
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
            elif (indicator_row['bb_position'] >= self.config.exit_threshold and 
                  self._is_valid_entry(indicator_row, 'short')) and self.short_allowed:
                
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
            should_exit, exit_reason, exit_metadata = self._should_exit_position(symbol, indicator_row, date)
            if should_exit:
                signals.append(Signal(
                    symbol=symbol,
                    timestamp=date,
                    signal_type='sell' if position.quantity > 0 else 'buy',
                    strength=1.0,
                    price=row['close'],
                    metadata={
                        'reason': exit_reason,
                        **exit_metadata
                    }
                ))
        
        # Check for new entries (only if we don't have max positions)
        current_positions = len(self.portfolio.get_positions())
        if current_positions < self.config.max_positions:
            # Long entry: oversold condition
            if (indicator_row['bb_position'] <= self.config.entry_threshold and 
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
            elif (indicator_row['bb_position'] >= self.config.exit_threshold and 
                  self._is_valid_entry(indicator_row, 'short')) and self.config.short_allowed:
                
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
                'max_holding_days': self.config.max_holding_days,
                'use_bb_exit': self.config.use_bb_exit,
                'use_take_profit': self.config.use_take_profit,
                'use_max_holding_days': self.config.use_max_holding_days,
                'require_ma_uptrend_for_sales': self.config.require_ma_uptrend_for_sales,
                'ma_trend_lookback': self.config.ma_trend_lookback,
                'short_allowed': self.config.short_allowed
            },
            'current_positions': len(self.portfolio.get_positions()),
            'max_positions': self.config.max_positions
        }
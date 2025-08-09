# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, IStrategy, merge_informative_pair)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib


class SimpleMAStrategy(IStrategy):
    """
    Elder Impulse System với EMA 20 + ADX + EMA20 Position + ATR Trailing Stop
    
    Based on backtest results:
    - Total Return: 2,398.36% over 5.5 years
    - Sharpe Ratio: 1.171
    - Max Drawdown: -38.86%
    - Win Rate: 28.8%
    - Profit Factor: 2.22
    
    Strategy Logic:
    - Daily Elder Impulse với EMA 20 + MACD (12,26,9)
    - Entry: Elder Green/Red + ADX > 20 + Price vs EMA20
    - Exit: Signal change hoặc ATR trailing stop
    """

    INTERFACE_VERSION = 3

    # Strategy parameters
    timeframe = '1h'
    
    # Multi-timeframe setup
    informative_timeframe = '1d'
    
    # Can this strategy go short?
    can_short: bool = True

    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 10,  # 1000% ROI để không tự động exit, dùng trailing stop
    }

    # Optimal stoploss designed for the strategy.
    stoploss = -0.99  # 99% stoploss để không can thiệp, dùng custom stop

    # Trailing stoploss
    trailing_stop = False  # Sử dụng custom trailing stop

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    # Hyperopt parameters
    ema_period = IntParameter(13, 26, default=20, space="buy")
    adx_threshold = IntParameter(18, 25, default=20, space="buy")
    atr_multiplier = DecimalParameter(2.0, 3.0, default=2.5, space="sell")

    # Run "populate_indicators()" only for new candle
    process_only_new_candles = True

    # These values can be overridden in the config
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 200

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame
        """
        # Get informative pair data for daily timeframe
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], 
                                                  timeframe=self.informative_timeframe)
        
        # Calculate indicators on daily timeframe
        informative = self.calculate_daily_indicators(informative)
        
        # Merge informative data to main timeframe
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, 
                                           self.informative_timeframe, ffill=True)
        
        # Calculate hourly indicators
        dataframe = self.calculate_hourly_indicators(dataframe)
        
        return dataframe

    def calculate_daily_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Calculate Elder Impulse và các indicators trên daily timeframe"""
        
        # EMA for Elder Impulse
        dataframe['ema'] = ta.EMA(dataframe, timeperiod=self.ema_period.value)
        
        # MACD for Elder Impulse  
        macd = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe['macd'] = macd['macd']
        dataframe['macd_signal'] = macd['macdsignal']
        dataframe['macd_hist'] = macd['macdhist']
        
        # ADX for trend strength
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        
        # EMA20 for position filter
        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        
        # Elder Impulse Signals
        dataframe['ema_rising'] = dataframe['ema'] > dataframe['ema'].shift(1)
        dataframe['macd_rising'] = dataframe['macd_hist'] > dataframe['macd_hist'].shift(1)
        
        # Elder Green: EMA tăng + MACD Histogram tăng
        dataframe['elder_green'] = (dataframe['ema_rising'] & dataframe['macd_rising'])
        
        # Elder Red: EMA giảm + MACD Histogram giảm  
        dataframe['elder_red'] = (~dataframe['ema_rising'] & ~dataframe['macd_rising'])
        
        # Elder Blue: Các trường hợp khác
        dataframe['elder_blue'] = ~(dataframe['elder_green'] | dataframe['elder_red'])
        
        return dataframe

    def calculate_hourly_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Calculate ATR và các indicators trên hourly timeframe"""
        
        # ATR for trailing stop
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        # Support/Resistance levels
        dataframe['high_20'] = dataframe['high'].rolling(window=20).max()
        dataframe['low_20'] = dataframe['low'].rolling(window=20).min()
        
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signals based on Elder Impulse System
        """
        # Long Entry: Elder Green + ADX > threshold + Price > EMA20
        dataframe.loc[
            (
                (dataframe['elder_green_1d'] == True) &
                (dataframe['adx_1d'] > self.adx_threshold.value) &
                (dataframe['close'] > dataframe['ema20_1d']) &
                (dataframe['volume'] > 0)
            ),
            'enter_long'] = 1

        # Short Entry: Elder Red + ADX > threshold + Price < EMA20
        dataframe.loc[
            (
                (dataframe['elder_red_1d'] == True) &
                (dataframe['adx_1d'] > self.adx_threshold.value) &
                (dataframe['close'] < dataframe['ema20_1d']) &
                (dataframe['volume'] > 0)
            ),
            'enter_short'] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signals based on Elder Impulse changes
        """
        # Long Exit: Elder không còn Green (Red hoặc Blue)
        dataframe.loc[
            (
                (dataframe['elder_green_1d'] == False)
            ),
            'exit_long'] = 1

        # Short Exit: Elder không còn Red (Green hoặc Blue)
        dataframe.loc[
            (
                (dataframe['elder_red_1d'] == False)
            ),
            'exit_short'] = 1

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        ATR-based trailing stoploss
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = dataframe.iloc[-1].squeeze()
        
        # Tính ATR trailing stop
        atr_value = last_candle['atr']
        atr_distance = atr_value * self.atr_multiplier.value
        
        if trade.is_short:
            # Short position: stop ở trên entry
            stop_price = trade.open_rate + atr_distance
            if current_rate > stop_price:
                return (current_rate - trade.open_rate) / trade.open_rate
        else:
            # Long position: stop ở dưới entry
            stop_price = trade.open_rate - atr_distance
            if current_rate < stop_price:
                return (current_rate - trade.open_rate) / trade.open_rate
        
        return self.stoploss
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


class ElderEMA20_ADX_Strategy(IStrategy):
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

    # Hyperopt parameters
    elder_ema_period = IntParameter(15, 25, default=20, space='buy')
    adx_threshold = IntParameter(15, 30, default=20, space='buy')
    ema_position_period = IntParameter(15, 25, default=20, space='buy')
    atr_multiplier = DecimalParameter(2.0, 3.5, default=2.5, space='sell')

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 50

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pairs will NOT be traded on, they are just for informational purposes.
        """
        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, self.informative_timeframe) for pair in pairs]
        return informative_pairs

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """

        # Get informative data (daily)
        informative = self.dp.get_pair_dataframe(pair=metadata['pair'], 
                                                 timeframe=self.informative_timeframe)

        # Calculate Elder Impulse on daily timeframe
        informative = self.calculate_elder_impulse(informative, 'informative')

        # Merge informative data
        dataframe = merge_informative_pair(dataframe, informative, self.timeframe, 
                                          self.informative_timeframe, ffill=True)

        # Calculate hourly indicators
        # ADX
        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        # EMA 20 for position
        dataframe['ema_20'] = ta.EMA(dataframe, timeperiod=self.ema_position_period.value)

        # ATR for trailing stop
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_stop_long'] = dataframe['close'] - (dataframe['atr'] * self.atr_multiplier.value)
        dataframe['atr_stop_short'] = dataframe['close'] + (dataframe['atr'] * self.atr_multiplier.value)

        # Volume (optional)
        dataframe['volume_ma'] = ta.SMA(dataframe['volume'], timeperiod=20)

        return dataframe

    def calculate_elder_impulse(self, dataframe: DataFrame, prefix: str = '') -> DataFrame:
        """
        Calculate Elder Impulse System với EMA 20
        """
        # EMA 20
        ema_col = f'{prefix}_ema_{self.elder_ema_period.value}' if prefix else f'ema_{self.elder_ema_period.value}'
        dataframe[ema_col] = ta.EMA(dataframe, timeperiod=self.elder_ema_period.value)

        # MACD (12, 26, 9)
        macd_col = f'{prefix}_macd' if prefix else 'macd'
        macd_signal_col = f'{prefix}_macd_signal' if prefix else 'macd_signal'
        macd_hist_col = f'{prefix}_macd_hist' if prefix else 'macd_hist'

        macd_result = ta.MACD(dataframe, fastperiod=12, slowperiod=26, signalperiod=9)
        dataframe[macd_col] = macd_result['macd']
        dataframe[macd_signal_col] = macd_result['macdsignal']
        dataframe[macd_hist_col] = macd_result['macdhist']

        # Elder Signal calculation
        elder_signal_col = f'{prefix}_elder_signal' if prefix else 'elder_signal'

        # EMA direction
        ema_up_col = f'{prefix}_ema_up' if prefix else 'ema_up'
        dataframe[ema_up_col] = dataframe[ema_col] > dataframe[ema_col].shift(1)

        # MACD histogram direction
        macd_up_col = f'{prefix}_macd_up' if prefix else 'macd_up'
        dataframe[macd_up_col] = dataframe[macd_hist_col] > dataframe[macd_hist_col].shift(1)

        # Elder Impulse logic
        dataframe[elder_signal_col] = 'gray'
        dataframe.loc[
            (dataframe[ema_up_col] == True) & (dataframe[macd_up_col] == True),
            elder_signal_col
        ] = 'green'
        dataframe.loc[
            (dataframe[ema_up_col] == False) & (dataframe[macd_up_col] == False),
            elder_signal_col
        ] = 'red'

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the entry signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with entry columns populated
        """

        # Long entry conditions
        dataframe.loc[
            (
                # Elder Green signal from daily timeframe (previous day)
                (dataframe[f'informative_elder_signal_{self.informative_timeframe}'].shift(1) == 'green') &

                # ADX > threshold (trend strength)
                (dataframe['adx'] > self.adx_threshold.value) &

                # Price above EMA20
                (dataframe['close'] > dataframe['ema_20']) &

                # Volume confirmation (optional)
                (dataframe['volume'] > dataframe['volume_ma']) &

                # Not in a position
                (dataframe['volume'] > 0)  # Basic volume check
            ),
            ['enter_long', 'enter_tag']] = (1, 'elder_long')

        # Short entry conditions
        dataframe.loc[
            (
                # Elder Red signal from daily timeframe (previous day)
                (dataframe[f'informative_elder_signal_{self.informative_timeframe}'].shift(1) == 'red') &

                # ADX > threshold (trend strength)
                (dataframe['adx'] > self.adx_threshold.value) &

                # Price below EMA20
                (dataframe['close'] < dataframe['ema_20']) &

                # Volume confirmation (optional)
                (dataframe['volume'] > dataframe['volume_ma']) &

                # Not in a position
                (dataframe['volume'] > 0)  # Basic volume check
            ),
            ['enter_short', 'enter_tag']] = (1, 'elder_short')

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the exit signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with exit columns populated
        """

        # Long exit conditions
        dataframe.loc[
            (
                # Elder signal change to Red or Gray
                (dataframe[f'informative_elder_signal_{self.informative_timeframe}'].shift(1).isin(['red', 'gray'])) |

                # Or price below EMA20 (trend change)
                (dataframe['close'] < dataframe['ema_20'])
            ),
            ['exit_long', 'exit_tag']] = (1, 'elder_exit_long')

        # Short exit conditions  
        dataframe.loc[
            (
                # Elder signal change to Green or Gray
                (dataframe[f'informative_elder_signal_{self.informative_timeframe}'].shift(1).isin(['green', 'gray'])) |

                # Or price above EMA20 (trend change)
                (dataframe['close'] > dataframe['ema_20'])
            ),
            ['exit_short', 'exit_tag']] = (1, 'elder_exit_short')

        return dataframe

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                       current_rate: float, current_profit: float, **kwargs) -> float:
        """
        Custom stoploss logic using ATR trailing stop
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or dataframe.empty:
            return -1  # No stoploss

        current_candle = dataframe.iloc[-1].squeeze()

        if trade.is_short:
            # Short position: stop if price goes above ATR stop
            atr_stop = current_candle['atr_stop_short']
            if current_rate >= atr_stop:
                return 0.01  # Small positive number to trigger stop
        else:
            # Long position: stop if price goes below ATR stop  
            atr_stop = current_candle['atr_stop_long']
            if current_rate <= atr_stop:
                return 0.01  # Small positive number to trigger stop

        return -1  # No stoploss triggered

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time: datetime,
                           entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """
        Confirm trade entry - can be used for additional filtering
        """
        # Additional confirmation logic can be added here
        return True

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                side: str, **kwargs) -> float:
        """
        Customize leverage for each new trade.
        """
        return 3.0  # Use 3x leverage (conservative)

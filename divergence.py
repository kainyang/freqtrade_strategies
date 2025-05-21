from freqtrade.strategy import IStrategy, IntParameter,informative, CategoricalParameter
from pandas import DataFrame
import talib.abstract as ta
from functools import reduce # For combining conditions

class MultiTimeframeDivergence(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = '1m' # Primary timeframe

    # Stoploss
    stoploss = -0.30  # 30%

    # ROI table for Take Profit
    # For longs, 150% target. For shorts, this is problematic as discussed.
    minimal_roi = {
        "0": 1.50
    }

    # Custom exit for 30-candle rule
    use_custom_exit = True

    # Parameters
    rsi_period = 14
    # Lookback for divergence (how many candles back to compare for LL/HH in price and HL/LH in RSI)
    # This would ideally be optimizable, perhaps different for each timeframe
    div_lookback_1m = IntParameter(10, 30, default=15, space="buy sell")
    div_lookback_3m = IntParameter(10, 30, default=10, space="buy sell") # 3m candles cover more time
    div_lookback_5m = IntParameter(10, 30, default=8, space="buy sell") # 5m candles cover even more

    # --- Helper to detect simple divergence (current vs Nth candle ago) ---
    def _check_divergence(self, dataframe: DataFrame, price_col: str, osc_col: str, lookback: int, type: str):
        if type == 'bullish':
            # Price: current low < low N periods ago
            # RSI: current rsi > rsi N periods ago
            price_divergence = dataframe[price_col] < dataframe[price_col].shift(lookback)
            osc_divergence = dataframe[osc_col] > dataframe[osc_col].shift(lookback)
        elif type == 'bearish':
            # Price: current high > high N periods ago
            # RSI: current rsi < rsi N periods ago
            price_divergence = dataframe[price_col] > dataframe[price_col].shift(lookback)
            osc_divergence = dataframe[osc_col] < dataframe[osc_col].shift(lookback)
        else:
            return DataFrame(False, index=dataframe.index)
        return price_divergence & osc_divergence

    # --- Populate indicators for informative timeframes ---
    def informative_populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        for tf_info in ['3m', '5m']:
            inf_df = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=tf_info)
            inf_df[f'rsi_{tf_info}'] = ta.RSI(inf_df['close'], timeperiod=self.rsi_period)

            lookback_val = getattr(self, f'div_lookback_{tf_info}').value
            inf_df[f'bullish_div_{tf_info}'] = self._check_divergence(inf_df, 'low', f'rsi_{tf_info}', lookback_val, 'bullish')
            inf_df[f'bearish_div_{tf_info}'] = self._check_divergence(inf_df, 'high', f'rsi_{tf_info}', lookback_val, 'bearish')

            dataframe = merge_informative_pair(dataframe, inf_df, self.timeframe, tf_info, ffill=True,
                                               append_prefix=True) # append_prefix adds 'inf_tf_'
        return dataframe

    # --- Populate indicators for base timeframe (1m) ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['rsi_1m'] = ta.RSI(dataframe['close'], timeperiod=self.rsi_period)
        lookback_1m = self.div_lookback_1m.value
        dataframe['bullish_div_1m'] = self._check_divergence(dataframe, 'low', 'rsi_1m', lookback_1m, 'bullish')
        dataframe['bearish_div_1m'] = self._check_divergence(dataframe, 'high', 'rsi_1m', lookback_1m, 'bearish')
        return dataframe

    # --- Populate Entry Signals ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions_long = [
            dataframe['bullish_div_1m'],
            dataframe[f'inf_3m_bullish_div_3m'], # Column names might vary slightly with merge_informative_pair
            dataframe[f'inf_5m_bullish_div_5m'],
            dataframe['volume'] > 0
        ]
        dataframe.loc[reduce(lambda x, y: x & y, conditions_long), 'enter_long'] = 1

        conditions_short = [
            dataframe['bearish_div_1m'],
            dataframe[f'inf_3m_bearish_div_3m'],
            dataframe[f'inf_5m_bearish_div_5m'],
            dataframe['volume'] > 0
        ]
        dataframe.loc[reduce(lambda x, y: x & y, conditions_short), 'enter_short'] = 1
        return dataframe

    # --- Populate Exit Signals ---
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Exit long if bearish divergence on any TF (more reactive)
        exit_long_conditions = [
            dataframe['bearish_div_1m'],
            dataframe[f'inf_3m_bearish_div_3m'],
            dataframe[f'inf_5m_bearish_div_5m'],
        ]
        dataframe.loc[reduce(lambda x, y: x | y, exit_long_conditions), 'exit_long'] = 1

        # Exit short if bullish divergence on any TF
        exit_short_conditions = [
            dataframe['bullish_div_1m'],
            dataframe[f'inf_3m_bullish_div_3m'],
            dataframe[f'inf_5m_bullish_div_5m'],
        ]
        dataframe.loc[reduce(lambda x, y: x | y, exit_short_conditions), 'exit_short'] = 1
        return dataframe

    # --- Custom Exit for 30-candle rule ---
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        # timeframe_duration_in_seconds is not directly available here,
        # but we can calculate it or use trade.timeframe if available and matches self.timeframe
        trade_duration_candles = (current_time - trade.open_date_utc).total_seconds() // self.timeframe_to_seconds(self.timeframe)

        if trade_duration_candles >= 30:
            return 'time_exit_30_candles'
        return None

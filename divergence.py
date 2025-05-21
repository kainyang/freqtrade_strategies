# --- freqtrade strategy ---
from freqtrade.strategy import IStrategy, IntParameter, FloatParameter, CategoricalParameter
from freqtrade.exchange import timeframe_to_prev_date # Not strictly used in this version, but good for reference
from freqtrade.persistence import Trade # For type hinting in custom_exit
from pandas import DataFrame, Series
import talib.abstract as ta
from functools import reduce
from freqtrade.strategy import merge_informative_pair # Ensure this import is available

import logging
logger = logging.getLogger(__name__) # Optional: For custom logging

class MultiTimeframeDivergence(IStrategy): # Renamed class slightly for clarity if you have old versions
    INTERFACE_VERSION = 3 # Set to 3 for recent Freqtrade versions

    # Strategy timeframe, primary timeframe for calculations and candle exits
    timeframe = '1m'

    # ROI table:
    # Targets 150% profit based on a 1:5 risk/reward ratio with a 30% stoploss.
    # In futures markets, this ROI is potentially achievable for both long and short positions.
    minimal_roi = {"0": 1.50}

    # Stoploss:
    stoploss = -0.30  # 30% stoploss

    # Trailing stop: (Optional, can be enabled if desired)
    # trailing_stop = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.02
    # trailing_only_offset_is_reached = False

    # Custom exit for 30-candle rule
    use_custom_exit = True
    process_only_new_candles = True # Recommended for most strategies

    # --- Strategy Parameters ---
    # RSI period
    rsi_period = 14 # Default RSI period

    # Lookback periods for divergence detection on each timeframe
    div_lookback_1m = IntParameter(10, 30, default=15, space="buy sell")
    div_lookback_3m = IntParameter(8, 25, default=12, space="buy sell") # Typically shorter for longer TFs
    div_lookback_5m = IntParameter(6, 20, default=10, space="buy sell") # Typically shorter for longer TFs

    # RSI Buffer: Current RSI must be X points above/below the previous RSI point in the divergence
    rsi_buffer = FloatParameter(0.0, 3.0, default=0.5, decimals=1, space="buy sell")


    # --- Helper function to detect divergence ---
    def _check_divergence(self, dataframe: DataFrame, price_col_name: str, osc_col_name: str, lookback: int, divergence_type: str, rsi_bf: float) -> Series:
        """
        Checks for divergence between price and an oscillator.
        :param dataframe: DataFrame with price and oscillator data.
        :param price_col_name: Column name for price (e.g., 'low' for bullish, 'high' for bearish).
        :param osc_col_name: Column name for the oscillator (e.g., 'rsi_1m').
        :param lookback: Number of periods to look back for the previous price/oscillator point.
        :param divergence_type: 'bullish' or 'bearish'.
        :param rsi_bf: RSI buffer value.
        :return: Pandas Series with True where divergence is detected.
        """
        if lookback <= 0: # Basic validation
            return Series([False] * len(dataframe), index=dataframe.index)
        if price_col_name not in dataframe.columns or osc_col_name not in dataframe.columns:
            logger.warning(f"Missing required columns for divergence check: {price_col_name} or {osc_col_name} in _check_divergence. Columns: {dataframe.columns.tolist()}")
            return Series([False] * len(dataframe), index=dataframe.index)

        # Shift data to compare current with past
        price_shifted = dataframe[price_col_name].shift(lookback)
        osc_shifted = dataframe[osc_col_name].shift(lookback)

        if divergence_type == 'bullish':
            # Price: current low <= low N periods ago (relaxed condition)
            price_condition = dataframe[price_col_name] <= price_shifted
            # RSI: current rsi > (rsi N periods ago + buffer)
            osc_condition = dataframe[osc_col_name] > (osc_shifted + rsi_bf)
        elif divergence_type == 'bearish':
            # Price: current high >= high N periods ago (relaxed condition)
            price_condition = dataframe[price_col_name] >= price_shifted
            # RSI: current rsi < (rsi N periods ago - buffer)
            osc_condition = dataframe[osc_col_name] < (osc_shifted - rsi_bf)
        else:
            return Series([False] * len(dataframe), index=dataframe.index)

        return price_condition & osc_condition


    # --- Populate indicators for informative timeframes (3m, 5m) ---
    def informative_populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        current_rsi_buffer = self.rsi_buffer.value

        for tf_info_str in ['3m', '5m']:
            inf_df = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=tf_info_str)
            if inf_df.empty:
                logger.info(f"Informative dataframe for {metadata['pair']} timeframe {tf_info_str} is empty. Defining empty signal columns.")
                # Define empty columns to prevent merge errors later if this path is taken
                dataframe[f'inf_{tf_info_str}_bullish_div'] = False # Ensure prefixed name matches expectation
                dataframe[f'inf_{tf_info_str}_bearish_div'] = False
                continue

            # Calculate RSI for the informative timeframe
            inf_df[f'rsi_{tf_info_str}'] = ta.RSI(inf_df['close'], timeperiod=self.rsi_period)

            # Determine lookback based on the informative timeframe string
            lookback_val = 0
            if tf_info_str == '3m':
                lookback_val = self.div_lookback_3m.value
            elif tf_info_str == '5m':
                lookback_val = self.div_lookback_5m.value

            # Calculate divergence signals for the informative timeframe
            # These columns will be prefixed with 'inf_{timeframe}_' by merge_informative_pair
            inf_df[f'bullish_div'] = self._check_divergence(inf_df, 'low', f'rsi_{tf_info_str}', lookback_val, 'bullish', current_rsi_buffer)
            inf_df[f'bearish_div'] = self._check_divergence(inf_df, 'high', f'rsi_{tf_info_str}', lookback_val, 'bearish', current_rsi_buffer)

            columns_to_merge = [f'bullish_div', f'bearish_div']
            existing_cols_in_inf_df = [col for col in columns_to_merge if col in inf_df.columns]

            if existing_cols_in_inf_df:
                dataframe = merge_informative_pair(dataframe, inf_df[existing_cols_in_inf_df], self.timeframe, tf_info_str, ffill=True, append_prefix=True)
            else:
                 for col_base_name in columns_to_merge: # e.g., col_base_name is 'bullish_div'
                     dataframe[f'inf_{tf_info_str}_{col_base_name}'] = False # Manually add prefixed empty columns
        return dataframe


    # --- Populate indicators for base timeframe (1m) ---
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        current_rsi_buffer = self.rsi_buffer.value
        lookback_1m = self.div_lookback_1m.value

        dataframe['rsi_1m'] = ta.RSI(dataframe['close'], timeperiod=self.rsi_period)
        dataframe['bullish_div_1m'] = self._check_divergence(dataframe, 'low', 'rsi_1m', lookback_1m, 'bullish', current_rsi_buffer)
        dataframe['bearish_div_1m'] = self._check_divergence(dataframe, 'high', 'rsi_1m', lookback_1m, 'bearish', current_rsi_buffer)
        return dataframe


    # --- Populate Entry Signals ---
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Define expected column names after merge_informative_pair
        col_b_1m = 'bullish_div_1m'
        col_b_3m = 'inf_3m_bullish_div' # Original col in inf_df was 'bullish_div'
        col_b_5m = 'inf_5m_bullish_div' # Original col in inf_df was 'bullish_div'

        col_s_1m = 'bearish_div_1m'
        col_s_3m = 'inf_3m_bearish_div' # Original col in inf_df was 'bearish_div'
        col_s_5m = 'inf_5m_bearish_div' # Original col in inf_df was 'bearish_div'

        required_cols = [col_b_1m, col_b_3m, col_b_5m, col_s_1m, col_s_3m, col_s_5m]
        for col in required_cols:
            if col not in dataframe.columns:
                logger.warning(f"Column '{col}' not found for entry check on {metadata['pair']}. Filling with False. Available: {dataframe.columns.tolist()}")
                dataframe[col] = False

        bullish_cond_1m_3m = dataframe[col_b_1m] & dataframe[col_b_3m]
        bullish_cond_1m_5m = dataframe[col_b_1m] & dataframe[col_b_5m]

        dataframe.loc[
            (bullish_cond_1m_3m | bullish_cond_1m_5m) &
            (dataframe['volume'] > 0),
            'enter_long'] = 1

        bearish_cond_1m_3m = dataframe[col_s_1m] & dataframe[col_s_3m]
        bearish_cond_1m_5m = dataframe[col_s_1m] & dataframe[col_s_5m]

        dataframe.loc[
            (bearish_cond_1m_3m | bearish_cond_1m_5m) &
            (dataframe['volume'] > 0),
            'enter_short'] = 1

        return dataframe


    # --- Populate Exit Signals (UPDATED LOGIC) ---
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Define expected column names for 3-minute divergence signals
        # These names assume the default prefix 'inf_{timeframe}_' from merge_informative_pair
        # and that the original columns in the 3m informative dataframe were 'bullish_div' and 'bearish_div'.
        col_b_3m = 'inf_3m_bullish_div' # Column for 3-minute bullish divergence
        col_s_3m = 'inf_3m_bearish_div' # Column for 3-minute bearish divergence

        # Ensure the necessary 3-minute divergence columns exist in the dataframe.
        # If not, fill them with False to prevent KeyErrors and allow strategy to proceed.
        required_cols_for_exit = [col_b_3m, col_s_3m]
        for col in required_cols_for_exit:
            if col not in dataframe.columns:
                logger.warning(f"Exit signal column '{col}' for 3m timeframe not found for pair {metadata['pair']}. Filling with False. Available: {dataframe.columns.tolist()}")
                dataframe[col] = False

        # Exit a long position if bearish divergence is detected on the 3-minute timeframe.
        if col_s_3m in dataframe.columns and not dataframe[col_s_3m].empty: # Check if Series is not empty
            dataframe.loc[dataframe[col_s_3m], 'exit_long'] = 1
        else:
            # Ensure 'exit_long' column exists even if condition column is missing or empty
            if 'exit_long' not in dataframe.columns:
                 dataframe['exit_long'] = 0 # Or False, depending on expected type by Freqtrade internals
            # If col_s_3m was filled with False by the loop above, this path means no exit signal from it.

        # Exit a short position if bullish divergence is detected on the 3-minute timeframe.
        if col_b_3m in dataframe.columns and not dataframe[col_b_3m].empty: # Check if Series is not empty
            dataframe.loc[dataframe[col_b_3m], 'exit_short'] = 1
        else:
            if 'exit_short' not in dataframe.columns:
                 dataframe['exit_short'] = 0
            # If col_b_3m was filled with False by the loop above, this path means no exit signal from it.

        return dataframe


    # --- Custom Exit for 30-candle rule ---
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        """
        Custom exit signal logic.
        Called to check if a trade should be exited based on trade duration.
        """
        timeframe_seconds = self.timeframe_to_seconds(self.timeframe)

        if timeframe_seconds > 0:
            trade_duration_candles = (current_time - trade.open_date_utc).total_seconds() // timeframe_seconds
        else:
            trade_duration_candles = 0 # Should not happen with valid timeframe

        if trade_duration_candles >= 30:
            logger.info(f"Exiting {pair} (Trade ID: {trade.id}) due to trade duration ({trade_duration_candles} candles) exceeding 30 candles.")
            return 'time_exit_30_candles' # This string is a custom reason for the exit

        return None # Return None if no custom exit is triggered

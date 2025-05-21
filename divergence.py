# --- freqtrade strategy: MTFD (Updated to use DecimalParameter) ---

# Required imports
import logging

import talib.abstract as ta
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_prev_date
from freqtrade.persistence import Trade # For type hinting in custom_exit
# MODIFIED IMPORT: Replaced FloatParameter with DecimalParameter
from freqtrade.strategy import (IStrategy, IntParameter, DecimalParameter, merge_informative_pair)

logger = logging.getLogger(__name__)

class MTFD(IStrategy):
    INTERFACE_VERSION = 3

    # Strategy timeframe
    timeframe = '1m'

    # ROI table (Futures context)
    minimal_roi = {"0": 1.50}

    # Stoploss
    stoploss = -0.30

    use_custom_exit = True
    process_only_new_candles = True

    # --- Strategy Parameters ---
    rsi_period = 14

    div_lookback_1m = IntParameter(10, 30, default=15, space="buy sell")
    div_lookback_3m = IntParameter(8, 25, default=12, space="buy sell")
    div_lookback_5m = IntParameter(6, 20, default=10, space="buy sell")

    # MODIFIED PARAMETER DEFINITION: Using DecimalParameter instead of FloatParameter
    rsi_buffer = DecimalParameter(0.0, 3.0, default=0.5, decimals=1, space="buy sell")

    # --- Helper function to detect divergence ---
    def _check_divergence(self, dataframe: DataFrame, price_col_name: str, osc_col_name: str, lookback: int, divergence_type: str, rsi_bf: float) -> Series:
        if lookback <= 0:
            return Series([False] * len(dataframe), index=dataframe.index)
        if price_col_name not in dataframe.columns or osc_col_name not in dataframe.columns:
            logger.warning(f"Missing required columns for divergence check: {price_col_name} or {osc_col_name} in _check_divergence. Columns: {dataframe.columns.tolist()}")
            return Series([False] * len(dataframe), index=dataframe.index)

        price_shifted = dataframe[price_col_name].shift(lookback)
        osc_shifted = dataframe[osc_col_name].shift(lookback)

        if divergence_type == 'bullish':
            price_condition = dataframe[price_col_name] <= price_shifted
            osc_condition = dataframe[osc_col_name] > (osc_shifted + rsi_bf)
        elif divergence_type == 'bearish':
            price_condition = dataframe[price_col_name] >= price_shifted
            osc_condition = dataframe[osc_col_name] < (osc_shifted - rsi_bf)
        else:
            return Series([False] * len(dataframe), index=dataframe.index)
        return price_condition & osc_condition

    # --- Populate indicators for informative timeframes (3m, 5m) ---
    def informative_populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        current_rsi_buffer = self.rsi_buffer.value

        # Pre-initialize all expected informative columns at the very beginning
        # This ensures they exist if merges are skipped or if merge_informative_pair
        # doesn't create them and we don't catch it immediately after.
        for tf_info_str_prefix_init in ['3m', '5m']:
            for signal_suffix_init in ['bullish_div', 'bearish_div']:
                col_name_init = f'inf_{tf_info_str_prefix_init}_{signal_suffix_init}'
                if col_name_init not in dataframe.columns:
                    dataframe[col_name_init] = False

        for tf_info_str in ['3m', '5m']:
            inf_df = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=tf_info_str)
            if inf_df.empty:
                logger.info(f"Informative dataframe for {metadata['pair']} timeframe {tf_info_str} is empty. Pre-initialized columns will be used.")
                # Columns for this tf_info_str should already be pre-initialized to False.
                # No merge will happen, so we continue to the next timeframe.
                continue

            inf_df[f'rsi_{tf_info_str}'] = ta.RSI(inf_df['close'], timeperiod=self.rsi_period)

            lookback_val = 0
            if tf_info_str == '3m': lookback_val = self.div_lookback_3m.value
            elif tf_info_str == '5m': lookback_val = self.div_lookback_5m.value

            inf_df[f'bullish_div'] = self._check_divergence(inf_df, 'low', f'rsi_{tf_info_str}', lookback_val, 'bullish', current_rsi_buffer)
            inf_df[f'bearish_div'] = self._check_divergence(inf_df, 'high', f'rsi_{tf_info_str}', lookback_val, 'bearish', current_rsi_buffer)

            columns_to_merge = [f'bullish_div', f'bearish_div']
            existing_cols_in_inf_df = [
                col for col in columns_to_merge
                if col in inf_df.columns and not inf_df[col].empty # Check if series has data points
            ]

            if existing_cols_in_inf_df and not inf_df.empty: # ensure inf_df has rows and selected columns have data
                informative_data_to_merge = inf_df[existing_cols_in_inf_df]
                if not informative_data_to_merge.empty:
                     dataframe = merge_informative_pair(dataframe, informative_data_to_merge, self.timeframe, tf_info_str, ffill=True, append_prefix=True)

                     # After merge, explicitly ensure the target columns for THIS timeframe exist.
                     # This is because merge_informative_pair might return a new dataframe
                     # and might not create a column if its source in informative_data_to_merge was all False/NaN.
                     expected_signal_cols_for_tf = [f'inf_{tf_info_str}_bullish_div', f'inf_{tf_info_str}_bearish_div']
                     for col_name in expected_signal_cols_for_tf:
                         if col_name not in dataframe.columns:
                             logger.warning(
                                 f"Strategy Dev: Column '{col_name}' was not found in dataframe after merge for {tf_info_str} on {metadata['pair']}. "
                                 f"Adding it as False. This might indicate an issue with merge_informative_pair or source data."
                             )
                             dataframe[col_name] = False
                else:
                    logger.info(f"Informative data slice for {tf_info_str} on {metadata['pair']} became empty after selecting columns. Merge skipped. Pre-initialized columns used.")
            else:
                 logger.info(f"No valid data in inf_df for {tf_info_str} on {metadata['pair']} to merge or inf_df was empty. Merge skipped. Pre-initialized columns used.")
        
        # Final check before returning - this is mostly for sanity checking during development.
        # The loop above should handle individual TFs.
        for tf_final_check in ['3m', '5m']:
            for sig_final_check in ['bullish_div', 'bearish_div']:
                final_col_name = f'inf_{tf_final_check}_{sig_final_check}'
                if final_col_name not in dataframe.columns:
                    # This would be unexpected if the logic above is correct.
                    logger.error(f"CRITICAL STRATEGY ERROR: Column '{final_col_name}' is MISSING from dataframe for {metadata['pair']} just before returning from informative_populate_indicators. Setting to False.")
                    dataframe[final_col_name] = False
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
        col_b_1m = 'bullish_div_1m'
        col_b_3m = 'inf_3m_bullish_div'
        col_b_5m = 'inf_5m_bullish_div'

        col_s_1m = 'bearish_div_1m'
        col_s_3m = 'inf_3m_bearish_div'
        col_s_5m = 'inf_5m_bearish_div'

        required_cols = [col_b_1m, col_b_3m, col_b_5m, col_s_1m, col_s_3m, col_s_5m]
        for col in required_cols:
            if col not in dataframe.columns:
                logger.warning(f"Column '{col}' not found for entry check on {metadata['pair']}. Filling with False. Available: {dataframe.columns.tolist()}")
                dataframe[col] = False

        bullish_cond_1m_3m = dataframe[col_b_1m] & dataframe[col_b_3m]
        bullish_cond_1m_5m = dataframe[col_b_1m] & dataframe[col_b_5m] # Corrected typo from previous full version

        dataframe.loc[
            (bullish_cond_1m_3m | bullish_cond_1m_5m) & # Corrected typo used here
            (dataframe['volume'] > 0),
            'enter_long'] = 1

        bearish_cond_1m_3m = dataframe[col_s_1m] & dataframe[col_s_3m]
        bearish_cond_1m_5m = dataframe[col_s_1m] & dataframe[col_s_5m]

        dataframe.loc[
            (bearish_cond_1m_3m | bearish_cond_1m_5m) &
            (dataframe['volume'] > 0),
            'enter_short'] = 1
        return dataframe

    # --- Populate Exit Signals (Based on 3-min TF Divergence Only) ---
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        col_b_3m = 'inf_3m_bullish_div'
        col_s_3m = 'inf_3m_bearish_div'

        required_cols_for_exit = [col_b_3m, col_s_3m]
        for col in required_cols_for_exit:
            if col not in dataframe.columns:
                logger.warning(f"Exit signal column '{col}' for 3m timeframe not found for pair {metadata['pair']}. Filling with False. Available: {dataframe.columns.tolist()}")
                dataframe[col] = False

        if col_s_3m in dataframe.columns and not dataframe[col_s_3m].empty:
            dataframe.loc[dataframe[col_s_3m], 'exit_long'] = 1
        else:
            if 'exit_long' not in dataframe.columns: dataframe['exit_long'] = 0

        if col_b_3m in dataframe.columns and not dataframe[col_b_3m].empty:
            dataframe.loc[dataframe[col_b_3m], 'exit_short'] = 1
        else:
            if 'exit_short' not in dataframe.columns: dataframe['exit_short'] = 0
        return dataframe

    # --- Custom Exit for 30-candle rule ---
    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        timeframe_seconds = self.timeframe_to_seconds(self.timeframe)
        if timeframe_seconds > 0:
            trade_duration_candles = (current_time - trade.open_date_utc).total_seconds() // timeframe_seconds
        else:
            trade_duration_candles = 0

        if trade_duration_candles >= 30:
            logger.info(f"Exiting {pair} (Trade ID: {trade.id}) due to trade duration ({trade_duration_candles} candles) exceeding 30 candles.")
            return 'time_exit_30_candles'
        return None

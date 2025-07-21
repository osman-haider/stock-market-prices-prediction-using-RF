import pandas as pd
import numpy as np

class TechnicalIndicatorExtractor:
    """
    Calculates technical indicators from preprocessed stock data.
    Assumes the DataFrame has at least 'Date', 'Close_Smooth', 'MA_Smooth', 'RSI_Smooth', and 'Target'.
    All indicators use the 'Close_Smooth' column instead of the raw 'Close' price.
    """

    def __init__(self, rsi_window=14, stoch_window=14, macd_fast=12, macd_slow=26, macd_signal=9, proc_window=10):
        """
        Initializes window sizes for indicators.
        """
        self.rsi_window = rsi_window
        self.stoch_window = stoch_window
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal = macd_signal
        self.proc_window = proc_window

    def add_rsi(self, df):
        """
        Adds RSI using 'Close_Smooth'.
        """
        delta = df['Close_Smooth'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    def add_stochastic_oscillator(self, df):
        """
        Adds Stochastic Oscillator (%K) using 'Close_Smooth' for close and min/max of MA_Smooth.
        """
        # Use MA_Smooth as a proxy for high/low since only smoothed features are available
        lowest_low = df['MA_Smooth'].rolling(window=self.stoch_window).min()
        highest_high = df['MA_Smooth'].rolling(window=self.stoch_window).max()
        df['Stochastic_K'] = 100 * (df['Close_Smooth'] - lowest_low) / (highest_high - lowest_low)
        return df

    def add_williams_r(self, df):
        """
        Adds Williams %R using 'Close_Smooth' and min/max of MA_Smooth.
        """
        lowest_low = df['MA_Smooth'].rolling(window=self.stoch_window).min()
        highest_high = df['MA_Smooth'].rolling(window=self.stoch_window).max()
        df['Williams_%R'] = (highest_high - df['Close_Smooth']) / (highest_high - lowest_low) * -100
        return df

    def add_macd(self, df):
        """
        Adds MACD and Signal using 'Close_Smooth'.
        """
        ema_fast = df['Close_Smooth'].ewm(span=self.macd_fast, adjust=False).mean()
        ema_slow = df['Close_Smooth'].ewm(span=self.macd_slow, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=self.macd_signal, adjust=False).mean()
        return df

    def add_proc(self, df):
        """
        Adds Price Rate of Change (PROC) using 'Close_Smooth'.
        """
        df['PROC'] = (df['Close_Smooth'] - df['Close_Smooth'].shift(self.proc_window)) / df['Close_Smooth'].shift(self.proc_window)
        return df

    def add_obv(self, df):
        """
        Adds On Balance Volume (OBV) using 'Close_Smooth' and the difference direction, using 'Volume' from the original DataFrame if present.
        If 'Volume' column is missing, OBV will not be added.
        """
        if 'Volume' not in df.columns:
            return df  # skip if no volume
        obv = [0]
        for i in range(1, len(df)):
            if df['Close_Smooth'][i] > df['Close_Smooth'][i-1]:
                obv.append(obv[-1] + df['Volume'][i])
            elif df['Close_Smooth'][i] < df['Close_Smooth'][i-1]:
                obv.append(obv[-1] - df['Volume'][i])
            else:
                obv.append(obv[-1])
        df['OBV'] = obv
        return df

    def transform(self, df):
        """
        Adds all technical indicators to the preprocessed DataFrame.
        Drops rows with NaN values caused by rolling/ewm windows.
        """
        data = df.copy()
        data = self.add_rsi(data)
        data = self.add_stochastic_oscillator(data)
        data = self.add_williams_r(data)
        data = self.add_macd(data)
        data = self.add_proc(data)
        data = self.add_obv(data)
        data = data.dropna().reset_index(drop=True)
        return data

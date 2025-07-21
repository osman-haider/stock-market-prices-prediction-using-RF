import numpy as np
import pandas as pd


class Preprocessor:
    """
    A class to preprocess stock data for machine learning models.
    Steps included:
    1. Exponential smoothing of the Close price.
    2. Calculation of moving average and RSI from smoothed Close price.
    3. Creation of classification target labels (+1 or -1) for price direction after d days.
    4. Output: A dataframe containing selected features and the target label.
    """

    def __init__(self, alpha=0.2, ma_window=5, rsi_window=5, d=3):
        """
        Initializes the preprocessor with smoothing and feature parameters.

        Parameters:
            alpha (float): Smoothing factor for exponential smoothing (0 < alpha < 1).
            ma_window (int): Window size for moving average.
            rsi_window (int): Window size for RSI calculation.
            d (int): Number of days ahead to compare for target creation.
        """
        self.alpha = alpha
        self.ma_window = ma_window
        self.rsi_window = rsi_window
        self.d = d

    def exponential_smoothing(self, series):
        """
        Applies exponential smoothing to a pandas Series.

        Parameters:
            series (pd.Series): Input time series (e.g., Close prices).

        Returns:
            pd.Series: Exponentially smoothed values.
        """
        result = [series.iloc[0]]  # Start with the first value
        for value in series.iloc[1:]:
            # S_t = alpha * Y_t + (1 - alpha) * S_{t-1}
            result.append(self.alpha * value + (1 - self.alpha) * result[-1])
        return pd.Series(result, index=series.index)

    def moving_average(self, series):
        """
        Calculates the moving average for a pandas Series.

        Parameters:
            series (pd.Series): Input time series.

        Returns:
            pd.Series: Moving average values.
        """
        return series.rolling(window=self.ma_window).mean()

    def rsi(self, series):
        """
        Calculates the Relative Strength Index (RSI) for a pandas Series.

        Parameters:
            series (pd.Series): Input time series.

        Returns:
            pd.Series: RSI values.
        """
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def create_target(self, series):
        """
        Creates target labels for classification (+1 for up, -1 for down) after d days.

        Parameters:
            series (pd.Series): Input time series (e.g., Close prices).

        Returns:
            pd.Series: Target labels.
        """
        future = series.shift(-self.d)
        diff = future - series
        return np.sign(diff).replace(0, -1)

    def transform(self, df):
        """
        Full preprocessing pipeline:
        - Exponential smoothing
        - Moving average and RSI calculation
        - Target creation
        - Drop rows with missing values due to rolling calculations

        Parameters:
            df (pd.DataFrame): Stock dataframe with at least a 'Close' column.

        Returns:
            pd.DataFrame: Preprocessed dataframe with features and target.
        """
        data = df.copy()

        # 1. Exponential Smoothing on 'Close'
        data['Close_Smooth'] = self.exponential_smoothing(data['Close'])

        # 2. Technical Indicators
        data['MA_Smooth'] = self.moving_average(data['Close_Smooth'])
        data['RSI_Smooth'] = self.rsi(data['Close_Smooth'])

        # 3. Target creation: +1 if price rises after d days, -1 if it falls
        data['Target'] = self.create_target(data['Close_Smooth'])

        # 4. Drop rows with NaN values (from rolling windows)
        data = data.dropna().reset_index(drop=True)

        # 5. Return only relevant columns
        feature_cols = ['Close_Smooth', 'MA_Smooth', 'RSI_Smooth']
        return data[['Date'] + feature_cols + ['Target']]

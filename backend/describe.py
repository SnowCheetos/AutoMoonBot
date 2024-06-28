import numpy as np
import pandas as pd

from typing import List, Dict, Tuple
from scipy.stats import zscore, skew, kurtosis, differential_entropy


class Descriptor:
    def __init__(
            self,
            config: Dict[str, List[str | int] | str]) -> None:
        '''
        config: {
            "columns": ["Open", "High", "Low", "Close", "Volume"],
            "windows": [8, 10, 12, 14, 16, 18, 20, ...],
        }
        '''
        self._columns = config['columns']
        self._windows = config['windows']

    @property
    def feature_dim(self) -> int:
        return len(self._windows) * 27

    def compute(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df = self._setup_data(df)
        df = self.compute_zscores(df)
        df = self.compute_skews(df)
        df = self.compute_kurtosis(df)
        df = self.compute_log_detrend_mean(df)
        df = self.compute_log_detrend_differential_entropy(df)
        return df

    def _setup_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data[self._columns].dropna()
        result = []

        for window in self._windows:
            df = data.copy()

            # Compute the moving average for the specified window
            sma = df.rolling(window=window).mean()

            # Calculate the differences
            delta = df.diff()
            gain  = delta.clip(lower=0)
            loss  = -delta.clip(upper=0)

            # Calculate average gains and losses
            avg_gain = gain.rolling(window=window).mean()
            avg_loss = loss.rolling(window=window).mean()
    
            # Calculate RS and RSI
            rs  = avg_gain / avg_loss
            rsi = 0.5 - (1 / (1 + rs))
    
            # Set the first 'window' rows to NaN (already handled by rolling().mean())
            df.loc[df.index[:window], self._columns] = float('nan')
    
            # Create a new MultiIndex for the columns with the window size as an additional level
            new_tuples = [(ticker, 'Price', price, f'Window={window}') for price, ticker in df.columns]
            sma_tuples = [(ticker, 'SMA', price, f'Window={window}') for price, ticker in sma.columns]
            rsi_tuples = [(ticker, 'RSI', price, f'Window={window}') for price, ticker in rsi.columns]

            # Combine the original data and moving average data with new MultiIndex
            combined_tuples       = new_tuples + sma_tuples + rsi_tuples
            combined_data         = pd.concat([df, sma, rsi], axis=1)
            multi_index           = pd.MultiIndex.from_tuples(combined_tuples, names=['Ticker', 'Type', 'Price', 'Window'])
            combined_data.columns = multi_index
            result.append(combined_data)

        # Combine the windowed DataFrames along the columns
        return pd.concat(result, axis=1)
    
    @staticmethod
    def compute_zscores(df: pd.DataFrame) -> pd.DataFrame:
        dt          = df.loc[:, df.columns.get_level_values('Type') == 'Price']
        dt          = dt.apply(zscore, nan_policy='omit')
        cols        = [(ticker, 'ZScore', price, window) for ticker, _, price, window in dt.columns]
        res         = pd.concat([df, dt], axis=1)
        res.columns = pd.MultiIndex.from_tuples(list(df.columns) + cols, names=['Ticker', 'Type', 'Price', 'Window'])
        return res # [n x 5]
    
    @staticmethod
    def compute_skews(df: pd.DataFrame) -> pd.DataFrame:
        idx         = [df.index[-1]]
        c1          = df.columns.get_level_values('Price') != 'Volume'
        c2          = df.columns.get_level_values('Type') == 'Price'
        dt          = df.loc[:, (c1) & (c2)]
        dt          = dt.apply(skew, nan_policy='omit', keepdims=True)
        dt.index    = idx
        cols        = [(ticker, 'Skews', price, window) for ticker, _, price, window in dt.columns]
        res         = pd.concat([df, dt], axis=1)
        res.columns = pd.MultiIndex.from_tuples(list(df.columns) + cols, names=['Ticker', 'Type', 'Price', 'Window'])
        return res # [n x 4]
    
    @staticmethod
    def compute_kurtosis(df: pd.DataFrame) -> pd.DataFrame:
        idx         = [df.index[-1]]
        dt          = df.loc[:, df.columns.get_level_values('Type') == 'Price']
        dt          = dt.apply(kurtosis, nan_policy='omit', keepdims=True)
        dt.index    = idx
        cols        = [(ticker, 'Kurtosis', price, window) for ticker, _, price, window in dt.columns]
        res         = pd.concat([df, dt], axis=1)
        res.columns = pd.MultiIndex.from_tuples(list(df.columns) + cols, names=['Ticker', 'Type', 'Price', 'Window'])
        return res # [n x 5]
    
    @staticmethod
    def compute_log_detrend_mean(df: pd.DataFrame) -> pd.DataFrame | None:
        idx       = [df.index[-1]]
        dt        = df.loc[:, df.columns.get_level_values('Price') != 'Volume']
        detrended = np.log(dt.xs('Price', level='Type', axis=1)) - np.log(dt.xs('SMA', level='Type', axis=1))
        detrended = detrended.dropna()
        if detrended.empty:
            return None
        
        dt          = pd.DataFrame([detrended.mean(axis=0)], index=idx)
        cols        = [(ticker, 'DetrendMean', price, window) for ticker, price, window in dt.columns]
        res         = pd.concat([df, dt], axis=1)
        res.columns = pd.MultiIndex.from_tuples(list(df.columns) + cols, names=['Ticker', 'Type', 'Price', 'Window'])
        return res # [n x 4]
    
    @staticmethod
    def compute_log_detrend_differential_entropy(df: pd.DataFrame) -> pd.DataFrame | None:
        idx       = [df.index[-1]]
        dt        = df.loc[:, df.columns.get_level_values('Price') != 'Volume']
        detrended = np.log(dt.xs('Price', level='Type', axis=1)) - np.log(dt.xs('SMA', level='Type', axis=1))
        detrended = detrended.dropna()
        if detrended.empty:
            return None
        dt          = detrended.apply(differential_entropy, nan_policy='omit', keepdims=True)
        dt.index    = idx
        cols        = [(ticker, 'DetrendDiffEntropy', price, window) for ticker, price, window in dt.columns]
        res         = pd.concat([df, dt], axis=1)
        res.columns = pd.MultiIndex.from_tuples(list(df.columns) + cols, names=['Ticker', 'Type', 'Price', 'Window'])
        return res # [n x 4]
    
    @staticmethod
    def compute_correlation_matrix(df: pd.DataFrame, column: List[str]=['Close']) -> pd.DataFrame:
        df         = df[column]
        cols       = [(ticker, price) for price, ticker in df.columns]
        df.columns = pd.MultiIndex.from_tuples(cols, names=['Ticker', 'Price'])
        return df.corr(method='spearman')
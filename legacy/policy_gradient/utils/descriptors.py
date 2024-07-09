import datetime
import numpy as np
from typing import List, Tuple, Dict

def compute_sharpe_ratio(
        returns:        List[float], 
        risk_free_rate: float) -> float:
    if len(returns) == 0:
        return 0

    returns = np.asarray(returns)

    excess_returns = returns - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    if std_excess_return == 0:
        return 0
    
    return mean_excess_return / std_excess_return

class Descriptors:
    def __init__(self, config: Dict[str, List[int]] | None=None) -> None:
        """
        config : {
            'sma' : [2, 4, 6, 8],
            'ema' : [2, 4, 6, 8],
            'rsi' : [2, 4, 6, 8],
            ...
        }
        """
        self._config = config

        self.f = {
            "sma": self.compute_sma,
            "ema": self.compute_ema,
            "rsi": self.compute_rsi,
            "sto": self.compute_stochastic_np,
            "zsc": self.compute_z_score,
            "nrm": self.compute_normalized_price,
            "grd": self.compute_normalized_grad,
            "cov": self.compute_coef_of_var,
            "cdl": self.compute_candle,
            "nts": self.normalize_time_of_day,
            "ntw": self.normalize_day_of_week
        }

    def __getitem__(self, key):
        return self.f.get(key)

    def compute(
            self,
            data: np.ndarray) -> np.ndarray:
        
        ts, open, high, low, close = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
        features = []
        for k in list(self._config.keys()):
            func = self.f[k]
            if k == "sto":
                params = self._config[k]
                windows, ks, ds = params["window"], params["k"], params["d"]
                for i in range(len(windows)):
                    _k, _d = func(close, high, low, windows[i], ks[i], ds[i])
                    if len(_k) > 0 and len(_d) > 0:
                        features += [_k[-1], _d[-1]]
                    else:
                        return np.array([])
            else:
                for p in self._config[k]:
                        if k == "cdl":
                            f = func(open, high, low, close, p)
                            if len(f) > 0: 
                                features += f
                            else:
                                return np.array([])
                        elif k == "nts" or k == "ntw":
                            f = func(ts[-1])
                            features += f
                        else:
                            f = func(close, p)
                            if len(f) > 0: 
                                features += [f[-1]]
                            else:
                                return np.array([])

        if len(features) == 0: 
            return np.array([])
        
        return np.asarray(features)[None, :]

    @staticmethod
    def normalize_time_of_day(unix_timestamp):
        # Convert Unix timestamp to datetime object
        dt = datetime.datetime.fromtimestamp(unix_timestamp)

        # Extract hours, minutes, and seconds
        hours = dt.hour
        minutes = dt.minute
        seconds = dt.second

        # Calculate the total seconds since the start of the day
        total_seconds = hours * 3600 + minutes * 60 + seconds

        # There are 86400 seconds in a day (24 * 3600)
        seconds_in_day = 24 * 3600

        # Normalize the total seconds to a range between -1 and 1
        normalized_time = (2 * total_seconds / seconds_in_day) - 1

        return [normalized_time]

    @staticmethod
    def normalize_day_of_week(unix_timestamp):
        # Convert Unix timestamp to datetime object
        dt = datetime.datetime.fromtimestamp(unix_timestamp)

        # Extract the day of the week (0 = Monday, ..., 6 = Sunday)
        day_of_week = dt.weekday()

        # Normalize the day of the week to a range between -1 and 1
        # day_of_week ranges from 0 to 6, we need to map this to -1 to 1
        normalized_day = (2 * day_of_week / 6) - 1

        return [normalized_day]

    @staticmethod
    def compute_candle(
            opens:  np.ndarray,
            highs:  np.ndarray,
            lows:   np.ndarray,
            closes: np.ndarray,
            window: int=4) -> np.ndarray:
        if len(closes) < window:
            return np.array([])

        prices = np.stack([opens, highs, lows, closes]).T[-window:]
        open   = opens[0]
        high   = highs.max()
        low    = lows.min()
        close  = closes[-1]
        mean   = prices.mean()
        candle = np.array([open, high, low, close])
        return (candle / mean - 1).tolist()

    @staticmethod
    def compute_coef_of_var(
            prices: np.ndarray, 
            window: int=64) -> np.ndarray:
        if len(prices) < window:
            return np.array([])
        
        arr  = prices[-window:]
        mean = arr.mean()
        std  = arr.std()
        return [std / mean]

    @staticmethod
    def compute_normalized_grad(
            prices: np.ndarray, 
            window: int=14) -> np.ndarray:
        if len(prices) < window:
            return np.array([])
        
        arr  = prices[-window:]
        mean = arr.mean()
        diff = arr[-1] - arr[0]
        return [diff / mean]

    @staticmethod
    def compute_normalized_price(
            prices: np.ndarray, 
            window: int=64) -> np.ndarray:
        if len(prices) < window:
            return np.array([])
        
        arr  = prices[-window:]
        mean = arr.mean()
        return arr / mean - 1

    @staticmethod
    def compute_z_score(
            prices: np.ndarray, 
            window: int=64) -> np.ndarray:
        if len(prices) < window:
            return np.array([])
        
        arr  = prices[-window:]
        mean = arr.mean()
        std  = arr.std()
        return (arr - mean) / std

    @staticmethod
    def compute_sma(
            prices: np.ndarray, 
            window: int=64) -> np.ndarray:
        
        if len(prices) < window:
            return np.array([])
        
        sma = np.convolve(prices, np.ones(window), 'valid') / window
        sma = np.concatenate((np.full(window-1, np.nan), sma)) / prices
        return sma - 1

    @staticmethod
    def compute_ema(
            prices: np.ndarray, 
            window: int=64) -> np.ndarray:
        
        if len(prices) < window:
            return np.array([])
        
        ema = np.zeros_like(prices)
        alpha = 2 / (window + 1)
        ema[0] = prices[0]
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
        ema = ema / prices
        return ema - 1

    @staticmethod
    def compute_rsi(
            prices: np.ndarray, 
            window: int=14) -> np.ndarray:
        if len(prices) < window + 1:
            return np.array([])

        deltas = np.diff(prices)
        seed = deltas[:window]
        up = seed[seed >= 0].sum() / window
        down = -seed[seed < 0].sum() / window
        rs = up / (down + 1e-9)
        rsi = np.zeros_like(prices)
        rsi[:window] = 100 - 100 / (1. + rs)

        for i in range(window, len(prices)):
            delta = deltas[i - 1]

            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (window - 1) + upval) / window
            down = (down * (window - 1) + downval) / window

            rs = up / (down + 1e-9)
            rsi[i] = 100 - 100 / (1. + rs)

        centered_rsi = 2 * (rsi / 100) - 1
        return centered_rsi

    @staticmethod
    def compute_stochastic_np(
            prices:   np.ndarray, 
            highs:    np.ndarray, 
            lows:     np.ndarray, 
            window:   int=14, 
            smooth_k: int=3, 
            smooth_d: int=3) -> Tuple[np.ndarray]:
        
        if len(prices) < window:
            return np.array([]), np.array([])
        
        if len(highs) != len(lows) != len(prices):
            return np.array([]), np.array([])
        
        k_values = []
        for i in range(window - 1, len(prices)):
            current_close = prices[i]
            lowest_low = np.min(lows[i - window + 1:i + 1])
            highest_high = np.max(highs[i - window + 1:i + 1])
            
            k_value = (current_close - lowest_low) / (highest_high - lowest_low)
            k_values.append(k_value)
        
        k_values = np.array(k_values)
        
        k_smooth = np.convolve(k_values, np.ones(smooth_k) / smooth_k, mode='valid')
        d_values = np.convolve(k_smooth, np.ones(smooth_d) / smooth_d, mode='valid')
        
        k_smooth = np.concatenate((np.full(window+1, np.nan), k_smooth)) - 0.5
        d_values = np.concatenate((np.full(window+3, np.nan), d_values)) - 0.5
        
        return k_smooth, d_values
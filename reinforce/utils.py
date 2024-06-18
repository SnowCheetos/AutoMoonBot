import numpy as np
from enum import Enum
from typing import List, Tuple


class Position(Enum):
    Cash  = 0
    Asset = 1


class Action(Enum):
    Buy  = 0
    Hold = 1
    Sell = 2


class Signal(Enum):
    Buy  = 0
    Idle = 1
    Sell = 2


class Status:
    def __init__(
            self, 
            max_risk: float, 
            alpha:    float) -> None:
        
        self._signal      = Signal.Idle
        self._take_profit = float("nan")
        self._stop_loss   = float("nan")
        self._max_risk    = max_risk
        self._alpha       = alpha

    @property
    def signal(self):
        return self._signal

    @property
    def take_profit(self):
        return self._take_profit
    
    @property
    def stop_loss(self):
        return self._stop_loss
    
    def reset(self):
        self._take_profit = float("nan")
        self._stop_loss   = float("nan")
        self._signal      = Signal.Idle

    @signal.setter
    def signal(self, action: int):
        signal = Signal(action)
        if abs(signal.value - self._signal.value) == 2:
            self.reset()
            return
        else:
            self._signal = signal
            return

    @take_profit.setter
    def take_profit(self, close: float):
        if self._signal == Signal.Buy:
            self._take_profit = close * (1 - self._alpha * self._max_risk)
        elif self._signal == Signal.Sell:
            self._take_profit = close * (1 + self._max_risk)

    @stop_loss.setter
    def stop_loss(self, close: float):
        if self._signal == Signal.Buy:
            self._stop_loss = close * (1 + self._max_risk)
        elif self._signal == Signal.Sell:
            self._stop_loss = close * (1 - self._alpha * self._max_risk)

    def confirm_buy(self, close: float) -> bool:
        if self._signal != Signal.Buy:
            return False
        if close >= self._stop_loss or close <= self._take_profit:
            return True
        return False
    
    def confirm_sell(self, close: float) -> bool:
        if self._signal != Signal.Sell:
            return False
        if close <= self._stop_loss or close >= self._take_profit:
            return True
        return False

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
    def __init__(self) -> None:
        self.f = {
            "sma": self.compute_sma,
            "ema": self.compute_ema,
            "rsi": self.compute_rsi,
            "sto": self.compute_stochastic_np,
            "zsc": self.compute_z_score,
            "nrm": self.compute_normalized_price,
        }

    def __getitem__(self, key):
        return self.f.get(key)

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
        rsi[:window] = 0.5 - 0.5 / (1. + rs)

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
            rsi[i] = 0.5 - 0.5 / (1. + rs)

        return rsi

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
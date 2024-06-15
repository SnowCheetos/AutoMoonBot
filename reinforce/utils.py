import numpy as np
from enum import Enum
from typing import List, Tuple

class Position(Enum):
    Cash  = 0
    Asset = 1

def compute_sharpe_ratio(returns: List[float], risk_free_rate: float) -> float:
    returns = np.asarray(returns)
    excess_returns = returns - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    if std_excess_return == 0:
        return 0
    
    return mean_excess_return / std_excess_return

def compute_sma(prices: np.ndarray, window: int=64) -> np.ndarray | None:
    if len(prices) < window:
        return None
    
    sma = np.convolve(prices, np.ones(window), 'valid') / window
    sma = np.concatenate((np.full(window-1, np.nan), sma)) / prices
    return sma - 1

def compute_ema(prices: np.ndarray, window: int=64) -> np.ndarray | None:
    if len(prices) < window:
        return None
    
    ema = np.zeros_like(prices)
    alpha = 2 / (window + 1)
    ema[0] = prices[0]
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i - 1]
    ema = ema / prices
    return ema - 1

def compute_rsi(prices: np.ndarray, window: int=14) -> np.ndarray | None:
    if len(prices) < window + 1:
        return None

    deltas = np.diff(prices)
    seed = deltas[:window]
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    rs = up / down
    rsi = np.zeros_like(prices)
    rsi[:window] = 100. - 100. / (1. + rs)

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

        rs = up / down
        rsi[i] = 0.5 - 1 / (1. + rs)

    return rsi

def compute_stochastic_np(prices: np.ndarray, highs: np.ndarray, lows: np.ndarray, window:int=14, smooth_k:int=3, smooth_d:int=3) -> Tuple[np.ndarray | None]:
    if len(prices) < window:
        return None, None
    
    if len(highs) != len(lows) != len(prices):
        return None, None
    
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
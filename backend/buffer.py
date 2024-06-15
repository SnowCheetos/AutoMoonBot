import numpy as np
import yfinance as yf

from typing import Dict, List
from collections import deque

from reinforce.utils import *


class DataBuffer:
    def __init__(
            self,
            ticker:         str,
            period:         str,
            interval:       str,
            queue_size:     int,
            feature_params: Dict[str, List[int] | Dict[str, List[int]]]) -> None:
        
        self._ticker   = ticker
        self._period   = period
        self._interval = interval
        self._queue = deque(maxlen=queue_size)
        self._feature_params = feature_params

        self._feature_funcs = {
            "sma": compute_sma,
            "ema": compute_ema,
            "rsi": compute_rsi,
            "sto": compute_stochastic_np
        }

        self._fill_queue()

    def _fill_queue(self) -> None:
        data = yf.download(
            tickers=self._ticker, 
            period=self._period,
            interval=self._interval)
        
        for idx in data.index:
            row = data.loc[idx]
            self._queue.append([
                idx.timestamp(),
                row.Open,
                row.High,
                row.Low,
                row.Close,
                row.Volume
            ])

    def update_queue(self) -> None:
        data = self.last_tohlcv()
        self._queue.append(list(data.values()))

    def last_tohlcv(self) -> Dict[str, float]:
        data = yf.download(
            tickers=self._ticker, 
            period="1d",
            interval=self._interval)
        
        last = data.loc[data.index[-1]]
        return {
            "timestamp": data.index[-1].timestamp(),
            "open":      last.Open,
            "high":      last.High,
            "low":       last.Low,
            "close":     last.Close,
            "volume":    last.Volume
        }
    
    def fetch_state(self, update: bool=True) -> np.ndarray:
        if update: self.update_queue()

        return self._construct_state()

    def _construct_state(self) -> np.ndarray:
        data = np.asarray(list(self._queue))
        high, low, close = data[:, 2], data[:, 3], data[:, 4]

        features = []
        for k in list(self._feature_params.keys()):
            func = self._feature_funcs[k]
            if k == "sto":
                params = self._feature_params[k]
                windows, ks, ds = params["window"], params["k"], params["d"]
                for i in range(len(windows)):
                    _k, _d = func(close, high, low, windows[i], ks[i], ds[i])
                    if len(_k) > 0 and len(_d) > 0:
                        features += [_k[-1], _d[-1]]
                    else:
                        return np.array([])
            else:
                for p in self._feature_params[k]:
                    f = func(close, p)
                    if len(f) > 0: 
                        features += [f[-1]]
                    else:
                        return np.array([])

        if len(features) == 0: 
            return np.array([])
        
        return np.asarray(features)[None, :]
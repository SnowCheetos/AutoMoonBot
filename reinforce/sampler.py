import sqlite3
import numpy as np

from collections import deque
from typing import Tuple, Dict, List

from reinforce.utils import Descriptors


class DataSampler:
    def __init__(
            self, 
            db_path:           str, 
            queue_size:        int,
            feature_params:    Dict[str, List[int] | Dict[str, List[int]]],
            max_access:        int | None=None,
            max_training_data: int | None=None) -> None:
        
        self._connection     = sqlite3.connect(db_path, check_same_thread=False)
        self._cursor         = self._connection.cursor()
        self._feature_params = feature_params

        self._cursor.execute("SELECT COUNT(*) FROM data")
        self._rows       = self._cursor.fetchone()[0]
        self._queue_size = queue_size
        self._queue      = deque(maxlen=queue_size)
        self._counter    = 0

        if max_access is not None:
            max_access = min(self._rows-2, max_access)

        if max_training_data is not None:
            if max_training_data < self._rows:
                self._counter = max(0, max_access - max_training_data)

        self._max_access        = max_access
        self._max_training_data = max_training_data

        self._feature_funcs = Descriptors()

    @property
    def counter(self):
        return self._counter

    @property 
    def max_access(self):
        return self._max_access
    
    @max_access.setter
    def max_access(self, m: int):
        if m > self._rows-2:
            m = self._rows-2
        elif m < self._queue_size:
            m = self._queue_size+2
        self._max_access = m

    def reset(self) -> None:
        self._queue = deque(maxlen=self._queue_size)
        if self._max_training_data is not None:
            if self._max_training_data < self._rows:
                self._counter = max(0, self._max_access - self._max_training_data)
        else:
            self._counter = 0

    def sample_next(self) -> Tuple[bool, float, np.ndarray]:
        if self._max_access is None:
            done = self._rows-2 <= self._counter
        else:
            done = self._max_access <= self._counter
        
        row = self._fetch_row(self._counter)
        if not done: 
            self._counter += 1
        
        return done, row[4], self._construct_state()

    def _fetch_row(self, index: int) -> Tuple[float]:
        res = self._cursor.execute("SELECT * FROM data WHERE id = ?", (index+1,))
        row = res.fetchone()
        self._queue.append(list(row))
        return row

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
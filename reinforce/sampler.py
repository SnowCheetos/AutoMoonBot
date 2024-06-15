import sqlite3
import numpy as np

from collections import deque
from typing import Tuple


class DataSampler:
    def __init__(self, db_path: str, queue_size: int=64) -> None:
        self._connection = sqlite3.connect(db_path)
        self._cursor = self._connection.cursor()

        self._cursor.execute("SELECT COUNT(*) FROM data")
        self._rows = self._cursor.fetchone()[0]
        self._queue_size = queue_size
        self._queue = deque(maxlen=queue_size)
        self._counter = 0

    @property
    def counter(self):
        return self._counter

    def reset(self) -> None:
        self._counter = 0
        self._queue = deque(maxlen=self._queue_size)

    def sample_next(self) -> Tuple[bool, float, np.ndarray]:
        done = self._rows-2 <= self._counter
        row = self._fetch_row(self._counter)
        if not done: 
            self._counter += 1
        
        return done, row[4], self._construct_state()

    def _fetch_row(self, index: int) -> Tuple[float]:
        res = self._cursor.execute("SELECT * FROM data WHERE id = ?", (index+1,))
        row = res.fetchone()
        self._queue.append(list(row))
        return row

    def _construct_state(self) -> np.ndarray | None:
        data = np.asarray(list(self._queue))
        return data
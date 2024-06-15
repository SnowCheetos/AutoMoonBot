import sqlite3
import numpy as np

from typing import Tuple


class DataSampler:
    def __init__(self, db_path: str) -> None:
        self._connection = sqlite3.connect(db_path)
        self._cursor = self._connection.cursor()

        self._cursor.execute("SELECT COUNT(*) FROM data")
        self._rows = self._cursor.fetchone()[0]
        self._counter = 0

    @property
    def counter(self):
        return self._counter

    def reset(self) -> None:
        self._counter = 0

    def sample_next(self) -> Tuple[bool, float, np.ndarray]:
        row = self._fetch_row(self._counter)
        self._counter += 1
        return self._rows <= self._counter, row[4], self._construct_state(row)

    def _fetch_row(self, index: int) -> Tuple[float]:
        res = self._cursor.execute("SELECT * FROM data WHERE id = ?", (index+1,))
        return res.fetchone()

    def _construct_state(self, row: Tuple[float]) -> np.ndarray:
        return np.asarray(row)
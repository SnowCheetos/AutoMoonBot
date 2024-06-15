import sqlite3
import numpy as np

from typing import Tuple


class DataSampler:
    def __init__(self, db_path: str) -> None:
        self._connection = sqlite3.connect(db_path)
        self._cursor = self._connection.cursor()

        self._counter = 0

    def reset(self) -> None:
        self._counter = 0

    def sample_next(self) -> np.ndarray:
        row = self._fetch_row(self._counter)
        self._counter += 1
        return self._construct_state(row)

    def _fetch_row(self, index: int) -> Tuple[float]:
        res = self._cursor.execute("SELECT * FROM data WHERE id = ?", (index,))
        res.fetchone()
        return res

    def _construct_state(self, row: Tuple[float]) -> np.ndarray:
        return np.asarray(row)
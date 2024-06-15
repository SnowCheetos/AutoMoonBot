import logging
import sqlite3
import numpy as np
import yfinance as yf

from typing import Dict, List, Optional
from collections import deque

from reinforce.utils import *


class DataBuffer:
    def __init__(
            self,
            ticker:         str,
            period:         str,
            interval:       str,
            queue_size:     int,
            db_path:        str,
            feature_params: Dict[str, List[int] | Dict[str, List[int]]],
            logger:         Optional[logging.Logger] = None) -> None:
        
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
        
        self._ticker   = ticker
        self._period   = period
        self._interval = interval
        self._queue = deque(maxlen=queue_size)
        self._db_path = db_path
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

    def update_queue(self, write_to_db: bool=False) -> None:
        data = self.last_tohlcv()
        self._queue.append(list(data.values()))

        if write_to_db:
            self.write_last_row_to_db()

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
    
    def _write_last_row_to_db(self):
        con = sqlite3.connect(self._db_path, check_same_thread=True)
        cursor = con.cursor()

        row = self._queue[-1]
        cursor.execute("""
            INSERT INTO data (timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (row[0], row[1], row[2], row[3], row[4], row[5]))
        
        con.commit()
        con.close()

    def write_queue_to_db(self, flush: bool=True):
        con = sqlite3.connect(self._db_path, check_same_thread=True)
        cursor = con.cursor()

        if flush:
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
            tables = cursor.fetchall()

            for table in tables:
                table_name = table[0]
                cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
                con.commit()
                self._logger.info(f"Table {table_name} dropped successfully")

            self._logger.info("All user tables dropped successfully")

            query = """
            CREATE TABLE IF NOT EXISTS data (
                id         INTEGER PRIMARY KEY,
                timestamp  REAL,
                open       REAL,
                high       REAL,
                low        REAL,
                close      REAL,
                volume     REAL
            )
            """
            cursor.execute(query)

        for row in self._queue:
            cursor.execute("""
            INSERT INTO data (timestamp, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (row[0], row[1], row[2], row[3], row[4], row[5]))

        con.commit()
        con.close()
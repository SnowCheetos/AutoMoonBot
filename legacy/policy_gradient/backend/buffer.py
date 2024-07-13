import logging
import sqlite3
import numpy as np
import yfinance as yf

from collections import deque
from typing import Dict, List, Optional

from utils import Descriptors


class DataBuffer:
    def __init__(
            self,
            ticker:         str,
            period:         str,
            interval:       str,
            queue_size:     int,
            db_path:        str,
            feature_params: Dict[str, List[int] | Dict[str, List[int]]],
            logger:         Optional[logging.Logger] = None,
            live_data:      bool=False) -> None:
        
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)
        
        self._queue_size     = queue_size
        self._live_data      = live_data
        self._ticker         = ticker
        self._period         = period
        self._interval       = interval
        self._queue          = deque(maxlen=queue_size)
        self._db_path        = db_path
        self._feature_params = feature_params
        self._done           = False

        self._con     = None
        self._cursor  = None
        self._counter = queue_size + 2
        self._rows    = 0
        
        if not live_data:
            self._logger.info(f"not live data, connecting to db {self._db_path}")
            self._con = sqlite3.connect(db_path, check_same_thread=False)
            self._cursor = self._con.cursor()
            self._cursor.execute("SELECT COUNT(*) FROM data")
            self._rows = self._cursor.fetchone()[0]

        self._feature_funcs = Descriptors(feature_params)
        self._fill_queue()

    def reset(self) -> None:
        if self._live_data:
            logging.warning("live buffer cannot be reset")
            return
        
        self._done    = False
        self._counter = self._queue_size + 2
        self._queue   = deque(maxlen=self._queue_size)
        self._fill_queue()

    @property
    def coef_of_var(self) -> float:
        data  = np.asarray(list(self._queue))
        close = data[:, 4]
        cov   = self._feature_funcs["cov"](close, len(self._queue))
        return cov[0]

    @property
    def done(self) -> bool:
        return self._done

    @property
    def queue(self) -> Dict[str, Dict[str, float]]:
        data = {"data": []}
        for row in self._queue:
            data["data"] += [{
                "timestamp": row[0],
                "open":      row[1],
                "high":      row[2],
                "low":       row[3],
                "close":     row[4],
                "volume":    row[5]
            }]
        return data

    def _fill_queue(self) -> None:
        if self._live_data:
            data = yf.download(
                tickers=self._ticker, 
                period=self._period,
                interval=self._interval)
        else:
            while self._counter < self._queue_size*2+2:
                self.update_queue(False)
            return

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
        if not self._live_data:
            if self._cursor:
                if self._counter >= self._rows-2:
                    self._done = True
                    return
                
                res = self._cursor.execute("SELECT * FROM data WHERE id = ?", (self._counter+1,))
                row = res.fetchone()
                self._queue.append(list(row)[1:])
                self._counter += 1
                return
            else:
                self._logger.error("no connection to database")
                return
        
        data = self.last_tohlcv()
        self._queue.append(list(data.values()))

        if write_to_db:
            self.write_last_row_to_db()

    def last_tohlcv(self) -> Dict[str, float]:
        if self._live_data:
            data = yf.download(
                tickers=self._ticker, 
                period="1d",
                interval=self._interval)
        
        else:
            # self.update_queue(False)
            if self._cursor:
                item = self._queue[-1]
                return {
                    "timestamp": item[0],
                    "open":      item[1],
                    "high":      item[2],
                    "low":       item[3],
                    "close":     item[4],
                    "volume":    item[5]
                }
            else:
                self._logger.error("no connection to database")
                return {}

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
        return self._feature_funcs.compute(data)
    
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
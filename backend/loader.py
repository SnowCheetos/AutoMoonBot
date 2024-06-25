import os
import sqlite3
import logging
import pandas as pd
import yfinance as yf

from typing import List, Dict, Tuple
from describe import Descriptor


class DataLoader:
    def __init__(
            self,
            session_id:     str,
            tickers:        List[str],
            db_path:        str,
            interval:       str,
            buffer_size:    int,
            feature_config: Dict[str, List[str | int] | str]) -> None:
        
        self._buffer      = None
        self._tickers     = yf.Tickers(tickers)
        self._interval    = interval
        self._buffer_size = buffer_size
        self._descriptor  = Descriptor(feature_config)
        self._db_path     = os.path.join(db_path, session_id)
        
        os.makedirs(self._db_path, exist_ok=True)
        db_file_path = os.path.join(self._db_path, 'data.db')
        
        try:
            self._conn = sqlite3.connect(db_file_path)
        except sqlite3.Error as e:
            logging.error(f"An error occurred while connecting to the database: {e}")
            self._conn = None

    def __del__(self) -> None:
        self._conn.close()

    @property
    def last_timestamp(self) -> pd.Timestamp:
        return self._buffer.index[-1]
    
    @property
    def data(self) -> pd.DataFrame:
        return self._buffer

    @property
    def features(self) -> Tuple[pd.DataFrame]:
        f = self._descriptor.compute(self.data)
        c = self._descriptor.compute_correlation_matrix(self.data, ['Close'])
        return (f, c)

    def init_db(self) -> None:
        history = self._tickers.history(interval=self._interval, period='2y', threads=True)
        for ticker in self._tickers.symbols:
            data = history.xs(ticker, level='Ticker', axis=1).reset_index()
            data.to_sql(ticker, self._conn, if_exists='replace', index_label='Id')

    def update_db(self) -> bool:
        history = self._tickers.history(interval=self._interval, start=self.last_timestamp, threads=True)
        history = history[history.index > self.last_timestamp]
        if len(history) == 0:
            return False
        for ticker in self._tickers.symbols:
            data = history.xs(ticker, level='Ticker', axis=1).reset_index()
            data.to_sql(ticker, self._conn, if_exists='replace', index_label='Id')
        return True

    def load_db(self, start: int = 0) -> None:
        dfs = []
        for ticker in self._tickers.symbols:
            query = f'SELECT * FROM {ticker} WHERE Id BETWEEN {start} AND {start + self._buffer_size}'
            data = pd.read_sql(query, self._conn)
            data.set_index('Datetime', inplace=True)
            data.columns = pd.MultiIndex.from_product([data.columns, [ticker]])
            dfs.append(data)
        self._buffer = pd.concat(dfs, axis=1)

    def load_row(self, row: int) -> bool:
        new_data_list = []
        for ticker in self._tickers.symbols:
            query = f"SELECT * FROM {ticker} WHERE Id = {row}"
            new_data = pd.read_sql(query, self._conn)
            if not new_data.empty:
                new_data.set_index('Datetime', inplace=True)
                new_data.columns = pd.MultiIndex.from_product([new_data.columns, [ticker]])
                new_data_list.append(new_data)
        if new_data_list:
            new_data_df = pd.concat(new_data_list, axis=1)
            self._buffer = pd.concat([self._buffer, new_data_df]).sort_index()
            if len(self._buffer) > self._buffer_size:
                self._buffer = self._buffer.iloc[-self._buffer_size:]
            return True
        return False
import sqlite3
import pandas as pd
import datetime as dt
from pandas import DataFrame
from backend.data import Streamer
from typing import List


class DBStreamer(Streamer):
    def __init__(
        self,
        db_path: str,
        tables: List[str],
        index: str,
        datetime: dt.datetime,
        timestep: dt.timedelta,
        queue_size: int,
        **kwargs,
    ) -> None:
        super().__init__(queue_size, **kwargs)

        try:
            self._conn = sqlite3.connect(
                db_path,
                check_same_thread=False,
            )
        except sqlite3.Error as e:
            raise AttributeError(
                f"an error occurred while connecting to the database: {e}"
            )
        self._tables = tables
        self._datetime = datetime
        self._timestep = timestep
        self._index = index

    def get_data(self) -> DataFrame | None:
        start = self._datetime
        end = self._datetime + self._timestep
        data = self.load_between(self._index, start.timestamp(), end.timestamp())
        if data:
            self._datetime = end
            return data
        return None

    def load_one(
        self,
        key: str,
        value: str,
        table: str | None = None,
    ) -> DataFrame | None:
        if table:
            query = f"SELECT * FROM {table} WHERE ? = ?"
        else:
            query = " UNION ".join(
                [
                    f"SELECT * FROM {table} AS table WHERE ? = ?"
                    for table in self._tables
                ]
            )
        data = pd.read_sql(sql=query, con=self._conn, params=[key, value])
        if not data.empty:
            return data
        return None

    def load_between(
        self,
        key: str,
        start: str,
        end: str,
        table: str | None = None,
    ) -> DataFrame | None:
        if table:
            query = f"SELECT * FROM {table} WHERE {key} BETWEEN ? AND ?"
        else:
            query = " UNION ".join(
                [
                    f"SELECT * FROM {table} AS table WHERE {key} BETWEEN ? AND ?"
                    for table in self._tables
                ]
            )
        data = pd.read_sql(sql=query, con=self._conn, params=[start, end])
        if not data.empty:
            return data
        return None

import sqlite3
import pandas as pd
import datetime as dt
from pandas import DataFrame
from typing import List


class DBInterface:
    def __init__(
        self,
        database: str,
        index: str,
        tables: List[str],
        starts: dt.datetime | int,
        step: dt.timedelta | int,
        stops: dt.datetime | int | None = None,
    ) -> None:

        self._db = database
        self._index = index
        self._tables = tables
        self._starts = starts
        self._step = step
        self._stops = stops

        try:
            self._con = sqlite3.connect(
                database=database,
                check_same_thread=False,
            )
        except sqlite3.Error as e:
            raise AttributeError(
                f"an error occurred while connecting to the database: {e}"
            )

    @property
    def starts(self) -> dt.datetime | int:
        return self._starts

    @property
    def step(self) -> dt.datetime | int:
        return self._step

    @property
    def stops(self) -> dt.datetime | int | None:
        return self._stops

    @starts.setter
    def starts(self, value: dt.datetime | int) -> None:
        self._starts = value

    @step.setter
    def step(self, value: dt.datetime | int) -> None:
        self._step = value

    @stops.setter
    def stops(self, value: dt.datetime | int | None) -> None:
        self._stops = value

    def incr(self, steps: int = 1) -> None:
        start = steps * self.step
        if start >= self.stops:
            return
        self.starts = start

    def get_one(
        self,
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
        data = pd.read_sql(sql=query, con=self._con, params=[self._index, value])
        if not data.empty:
            return data
        return None

    def get_between(
        self,
        start: str,
        end: str,
        table: str | None = None,
    ) -> DataFrame | None:
        if table:
            query = f"SELECT * FROM {table} WHERE ? BETWEEN ? AND ?"
        else:
            query = " UNION ".join(
                [
                    f"SELECT * FROM {table} AS table WHERE ? BETWEEN ? AND ?"
                    for table in self._tables
                ]
            )
        data = pd.read_sql(sql=query, con=self._con, params=[self._index, start, end])
        if not data.empty:
            return data
        return None

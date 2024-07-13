import time
import threading
import datetime as dt
from pandas import DataFrame
from queue import Queue, Empty, Full
from typing import List, Any, Iterable

from automoonbot.data import DBInterface


class Streamer:
    done_policies = {"omit", "raise"}

    def __init__(
        self,
        queue_size: int,
        done: str = "omit",
        data: Iterable | None = None,
        **kwargs,
    ) -> None:
        self._running = False
        self._data = iter(data) if data is not None else None
        self._queue = Queue(maxsize=queue_size)
        self._lock = threading.Lock()
        self._thread = None
        self._kwargs = kwargs
        assert (
            done in self.__class__.done_policies
        ), f"invalid done policy, must be one of {self.__class__.done_policies}"
        self._done = done

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(
                target=self._fetch_loop, kwargs=self._kwargs
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
        self._thread.join()

    def prefill(self, **_) -> None:
        """
        Override this method for specific types of streamers
        """
        if not self._data:
            return
        while not self._queue.full():
            try:
                data = next(self._data)
            except StopIteration:
                return
            self._store(data)

    def get_data(self, sleep: int = 1) -> Any | None:
        """
        Override this method for specific types of streamers
        """
        if not self._data:
            return None
        try:
            data = next(self._data)
        except StopIteration:
            return None
        time.sleep(sleep)
        return data

    def _store(self, data: Any) -> None:
        if data is None:
            return
        try:
            self._queue.put_nowait(data)
        except Full:
            try:
                discarded = self._queue.get_nowait()
                del discarded
                self._queue.put(data)
            except Empty:
                return

    def _fetch_loop(self, **kwargs) -> None:
        self.prefill(**kwargs)
        while self.running:
            data = self.get_data(**kwargs)
            self._store(data)

    def __iter__(self) -> Iterable:
        return self

    def __next__(self) -> Any | None:
        try:
            return self._queue.get_nowait()
        except Empty:
            if self._done == "omit":
                return None
            elif self._done == "raise":
                raise StopIteration

    def __del__(self) -> None:
        self.stop()

    def __enter__(self) -> Any:
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()


class DBStreamer(Streamer, DBInterface):
    def __init__(
        self,
        queue_size: int,
        database: str,
        index: str,
        tables: List[str],
        starts: dt.datetime | int,
        step: dt.timedelta | int,
        stops: dt.datetime | int | None = None,
        **kwargs,
    ) -> None:
        Streamer.__init__(queue_size, **kwargs)
        DBInterface.__init__(database, index, tables, starts, step, stops, **kwargs)

    def prefill(self) -> None:
        pass

    def get_data(self) -> DataFrame | None:
        pass

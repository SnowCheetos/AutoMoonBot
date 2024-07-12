import time
import threading
import datetime as dt
from pandas import DataFrame
from queue import Queue, Empty, Full
from typing import List, Any, Iterable
from backend.data import DBInterface


class Streamer:
    def __init__(
        self,
        queue_size: int,
        data: Iterable | None = None,
        **kwargs,
    ) -> None:
        self._running = False
        self._data = iter(data) if data is not None else None
        self._queue = Queue(maxsize=queue_size)
        self._thread = threading.Thread(
            target=self._fetch_loop,
            kwargs=kwargs,
        )

    @property
    def running(self) -> bool:
        return self._running

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._thread.join()

    def prefill(self, **kwargs) -> None:
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

    def get_data(self, sleep: int=1) -> Any:
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
            disgard = self._queue.get_nowait()
            self._queue.put(data)
            del disgard

    def _fetch_loop(self, **kwargs) -> None:
        self.prefill(**kwargs)
        while self._running:
            data = self.get_data(**kwargs)
            self._store(data)

    def __iter__(self) -> Iterable:
        return self

    def __next__(self) -> Any:
        try:
            return self._queue.get_nowait()
        except Empty:
            return None


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

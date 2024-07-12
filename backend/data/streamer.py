import threading
from typing import Any, Iterable
from queue import Queue, Empty, Full


class Streamer:
    def __init__(self, queue_size: int, **kwargs) -> None:
        self._running = False
        self._queue = Queue(maxsize=queue_size)
        self._thread = threading.Thread(
            target=self._fetch_loop,
            kwargs=kwargs,
        )

    def start(self) -> None:
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._thread.join()

    def prefill(self) -> None:
        """
        Override this method for specific types of streamers
        """
        return

    def get_data(self) -> Any:
        """
        Override this method for specific types of streamers
        """
        return

    def _store(self, data: Any) -> None:
        if data is None:
            return
        try:
            self._queue.put(data)
        except Full:
            disgard = self._queue.get()
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
            return self._queue.get()
        except Empty:
            return None

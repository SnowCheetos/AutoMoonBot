import sqlite3
from automoonbot.moonpy.data import HeteroGraph


class DataInterface(HeteroGraph):
    def __init__(self) -> None:
        super().__init__()
        self._db = None

    def reset(self):
        pass

    def update(self):
        pass

    def test(self):
        pass
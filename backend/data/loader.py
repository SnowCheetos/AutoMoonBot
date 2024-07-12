import sqlite3
from backend.data import HeteroGraph

class DataLoader:
    def __init__(self) -> None:
        
        self._db = None
        self._graph = HeteroGraph()

    def reset(self):
        pass

    def update(self):
        pass

    def test(self):
        pass
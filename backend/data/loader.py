import sqlite3
from backend.data import Graph

class DataLoader:
    def __init__(self, prices, news, buffer_size) -> None:
        
        self._db = None
        self._graph = Graph(prices, news, buffer_size)

    def reset(self):
        pass

    def update(self):
        pass

    def test(self):
        pass
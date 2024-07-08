from enum import Enum, auto


class Node(Enum):
    Price = auto()
    News = auto()

class Edge(Enum):
    Authors = auto()
    Publisher = auto()
    Topics = auto()
    Tickers = auto()
    Reference = auto()
    Correlation = auto()
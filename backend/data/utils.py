import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any
from collections import namedtuple


class NodeType(Enum):
    Ticker = auto()
    News = auto()
    # TODO more node types
    # Event     # e.g. Earnings, SEC filings, Dividends, Splits ...
    # Economy   # e.g. GDP, CPI, Unemployment, Inflation ...
    # Currency  # e.g. CNY, JPY, EUR, GBP ...
    # Crypto    # e.g. BTC, ETH ...
    # Commodity # e.g. Oil, Gold, Copper, Wheat, Corn ...


@dataclass
class EdgeNodeType:
    value: int
    src: NodeType
    tgt: NodeType


_edge = namedtuple("edge", ["value", "src_type", "tgt_type"])


class EdgeType(Enum):
    Authors = _edge(0, NodeType.News, NodeType.News)
    Publisher = _edge(1, NodeType.News, NodeType.News)
    Topics = _edge(2, NodeType.News, NodeType.News)
    Tickers = _edge(3, NodeType.News, NodeType.News)
    Reference = _edge(4, NodeType.News, NodeType.Ticker)
    Influence = _edge(5, NodeType.Ticker, NodeType.News)
    Correlation = _edge(6, NodeType.Ticker, NodeType.Ticker)

    @property
    def src_type(self):
        return self.value.src_type

    @property
    def tgt_type(self):
        return self.value.tgt_type


@dataclass
class Edge:
    src_id: str
    tgt_id: str
    src_index: int
    tgt_index: int
    src_type: NodeType
    tgt_type: NodeType
    edge_type: EdgeType
    edge_attr: Dict[str, Any]


def compute_time_decay(
    start: float, end: float, shift: float = 7, alpha: float = 0.5
) -> float:
    if start > end:
        return 0

    sigmoid = lambda x, alpha: 1 / (1 + np.exp(-alpha * x))

    time_delta = max(end - start, 1e-3)
    log_time = np.log(time_delta)
    shifted = log_time - shift
    return 1 - sigmoid(shifted, alpha)

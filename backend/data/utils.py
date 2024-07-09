import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Set, Any
from collections import namedtuple

_node = namedtuple(typename="node", field_names=["value", "node_name"])
_edge = namedtuple(
    typename="edge", field_names=["value", "edge_name", "src_type", "tgt_type"]
)


class NodeType(Enum):
    Ticker = _node(value=0, node_name="tickers")
    News = _node(value=1, node_name="news")
    Event = _node(
        value=2, node_name="event"
    )  # e.g. Earnings, SEC filings, Dividends, Splits ...
    Economy = _node(
        value=3, node_name="economy"
    )  # e.g. GDP, CPI, Unemployment, Inflation ...
    Currency = _node(value=4, node_name="currency")  # e.g. CNY, JPY, EUR, GBP ...
    Crypto = _node(value=5, node_name="crypto")  # e.g. BTC, ETH ...
    Commodity = _node(
        value=6, node_name="commodity"
    )  # e.g. Oil, Gold, Copper, Wheat, Corn ...
    Options = _node(value=7, node_name="options")

    @property
    def S(self) -> str:
        """
        Short name to save time typing
        """
        return self.value.node_name

    @classmethod
    def names(cls) -> Set[str]:
        return {m.value.node_name for m in cls._member_map_.values()}


@dataclass
class EdgeNodeType:
    value: int
    src: NodeType
    tgt: NodeType


class EdgeType(Enum):
    Authors = _edge(
        value=0,
        edge_name="shared_authors",
        src_type=NodeType.News,
        tgt_type=NodeType.News,
    )
    Publisher = _edge(
        value=1,
        edge_name="shared_publisher",
        src_type=NodeType.News,
        tgt_type=NodeType.News,
    )
    Topics = _edge(
        value=2,
        edge_name="shared_topics",
        src_type=NodeType.News,
        tgt_type=NodeType.News,
    )
    Tickers = _edge(
        value=3,
        edge_name="shared_tickers",
        src_type=NodeType.News,
        tgt_type=NodeType.News,
    )
    Reference = _edge(
        value=4,
        edge_name="references",
        src_type=NodeType.News,
        tgt_type=NodeType.Ticker,
    )
    Influence = _edge(
        value=5,
        edge_name="influences",
        src_type=NodeType.Ticker,
        tgt_type=NodeType.News,
    )
    Correlation = _edge(
        value=6,
        edge_name="correlations",
        src_type=NodeType.Ticker,
        tgt_type=NodeType.Ticker,
    )

    @property
    def S(self) -> str:
        return self.value.edge_name

    @property
    def src_type(self) -> NodeType:
        return self.value.src_type

    @property
    def tgt_type(self) -> NodeType:
        return self.value.tgt_type
    
    @classmethod
    def names(cls) -> Set[str]:
        return {m.value.edge_name for m in cls._member_map_.values()}


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

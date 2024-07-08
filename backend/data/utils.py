from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any
from collections import namedtuple


class NodeType(Enum):
    Ticker = auto()
    News = auto()


@dataclass
class EdgeNodeType:
    value: int
    src: NodeType
    tgt: NodeType


_edge = namedtuple("edge", ["value", "src_type", "tgt_type"])


class EdgeType(Enum):
    Authors = _edge(0, NodeType.News, NodeType.News)
    Publisher = _edge(0, NodeType.News, NodeType.News)
    Topics = _edge(0, NodeType.News, NodeType.News)
    Tickers = _edge(0, NodeType.News, NodeType.News)
    Reference = _edge(0, NodeType.News, NodeType.Ticker)
    Correlation = _edge(0, NodeType.Ticker, NodeType.Ticker)

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

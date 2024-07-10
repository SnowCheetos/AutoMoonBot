import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Set, Any
from collections import namedtuple

_node = namedtuple(typename="node", field_names=["value", "n", "attr"])

_edge = namedtuple(typename="edge", field_names=["value", "n", "s", "t", "attr"])


class Nodes(Enum):
    Stock = _node(
        value=auto(),
        n="stock",
        attr={
            "symbol": str,
            "company": str,
            "time_updated": float,
            "prices": list,
            "interval": str,
        },
    )

    Company = _node(
        value=auto(),
        n="company",
        attr={
            "name": str,
            "symbol": str,
            "sector": str,
            "industry": str,
            "country": str,
            "currency": str,
            "balance_sht": dict,
            "income_stmt": dict,
            "cash_flow": dict,
            "earnings": dict,
            "dividends": list,
            "splits": list,
            "calendar": list,
        },
    )

    News = _node(
        value=auto(),
        n="news",
        attr={
            "title": str,
            "summary": str,
            "source": str,
            "category": str,
            "time_published": float,
            "sentiment": float,
            "authors": list,
            "topics": list,
        },
    )

    Author = _node(
        value=auto(), 
        n="author", 
        attr={
            "name": str,
        }
    )
    
    Topic = _node(
        value=auto(), 
        n="topic", 
        attr={

        }
    )
    
    Publisher = _node(
        value=auto(),
        n="publisher", 
        attr={

        }
    )

    Event = _node(value=auto(), n="event", attr={})  # e.g. Earnings, SEC filings ...
    Economy = _node(
        value=auto(), n="economy", attr={}
    )  # e.g. GDP, CPI, Unemployment, Inflation ...
    Currency = _node(value=auto(), n="currency", attr={})  # e.g. CNY, JPY, EUR, GBP ...
    Crypto = _node(value=auto(), n="crypto", attr={})  # e.g. BTC, ETH ...
    Commodity = _node(
        value=auto(), n="commodity", attr={}
    )  # e.g. Oil, Gold, Copper, Wheat, Corn ...
    Options = _node(value=auto(), n="options", attr={})

    @property
    def n(self) -> str:
        return self.value.n

    @classmethod
    def names(cls) -> Set[str]:
        return {m.value.n for m in cls._member_map_.values()}


class Edges(Enum):
    Issued = _edge(
        value=auto(),
        n="issued",
        s=Nodes.Company,
        t=Nodes.Stock,
        attr={

        },
    )

    Authored = _edge(
        value=auto(),
        n="authored",
        s=Nodes.Author,
        t=Nodes.News,
        attr={
            "time_decay": float,
        },
    )

    Published = _edge(
        value=auto(),
        n="published",
        s=Nodes.Publisher,
        t=Nodes.News,
        attr={"time_decay": float, "category": str},
    )

    Covered = _edge(
        value=auto(),
        n="covered",
        s=Nodes.News,
        t=Nodes.Topic,
        attr={"time_decay": float, "relevance": float},
    )

    Referenced = _edge(
        value=auto(),
        n="referenced",
        s=Nodes.News,
        t=Nodes.Ticker,
        attr={
            "time_decay": float,
            "relevance": float,
            "sentiment": float,
        },
    )

    Correlated = _edge(
        value=auto(),
        n="correlated",
        s=Nodes.Ticker,
        t=Nodes.Ticker,
        attr={"time_decay": float, "corr": float},
    )
    # TODO More...

    @property
    def n(self) -> str:
        return self.value.n

    @property
    def s(self) -> Nodes:
        return self.value.s

    @property
    def t(self) -> Nodes:
        return self.value.t

    @classmethod
    def names(cls) -> Set[str]:
        return {m.value.n for m in cls._member_map_.values()}


@dataclass
class Edge:
    src_id: str
    tgt_id: str
    src_index: int
    tgt_index: int
    s: Nodes
    t: Nodes
    edge_type: Edges
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

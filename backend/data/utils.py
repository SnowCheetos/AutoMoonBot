import numpy as np
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Set, Any
from collections import namedtuple


_node = namedtuple(typename="node", field_names=["value", "n", "attr"])

_edge = namedtuple(typename="edge", field_names=["value", "n", "s", "t", "q", "attr"])


class Node(Enum):
    Index = _node(
        value=auto(),
        n="index",  # For now this represents both index and ETF
        attr={
            "symbol": str,
            "sector": str,
            "timestamp": float,  # current environment time
            "prices": list,
            "interval": str,
        },
    )

    Equity = _node(
        value=auto(),
        n="equity",
        attr={
            "symbol": str,
            "company": str,
            "timestamp": float,
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
            "founded": float,
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
            "timestamp": float,
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
            "avg_sentiment": float,
        },
    )

    Exchange = _node(
        value=auto(),
        n="exchange",
        attr={
            "open": bool,
            "country": str,
            "region": str,
            "currency": str,
        },
    )

    Topic = _node(value=auto(), n="topic", attr={})

    Publisher = _node(value=auto(), n="publisher", attr={})

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

    @property
    def attr(self) -> Dict[str, Any]:
        return self.value.attr

    @classmethod
    def names(cls) -> Set[str]:
        return {m.value.n for m in cls._member_map_.values()}


class Edges(Enum):
    # I'm actually not so sure...
    Owns = _edge(
        value=auto(),
        n="owns",
        s=Node.Company,
        t=Node.Equity,
        q=None,
        attr={
            "amount": float,
        },
    )

    Correlates = _edge(
        value=auto(),
        n="correlates",
        s=Node.Equity,
        t=Node.Equity,
        q=None,
        attr={
            "corr": float,
        },
    )

    Represents = _edge(
        value=auto(),
        n="represents",
        s=Node.Equity,
        t=Node.Company,
        q=None,
        attr={
            "corr": float,
        },
    )

    Includes = _edge(
        value=auto(),
        n="includes",
        s=Node.Topic,
        t=Node.Company,
        q=None,
        attr={
            "corr": float,
        },
    )

    Hosts = _edge(
        value=auto(),
        n="hosts",
        s=Node.Exchange,
        t=Node.Equity,
        q=None,
        attr={
            "amount": float,
        },
    )

    Serves = _edge(
        value=auto(),
        n="serves",
        s=Node.Author,
        t=Node.Publisher,
        q=None,
        attr={...},
    )

    Drafted = _edge(
        value=auto(),
        n="drafted",
        s=Node.Author,
        t=Node.News,
        q=lambda t: float(t["overall_sentiment_score"]),
        attr={
            "time_decay": "t2c",  # From target to current
            "sentiment": "t->q",
        },
    )

    Published = _edge(
        value=auto(),
        n="published",
        s=Node.Publisher,
        t=Node.News,
        q=lambda t: t["category_within_source"],
        attr={
            "time_decay": "t2c",  # From target to current
            "category": "t->q",  # Publisher internal category
        },
    )

    Covered = _edge(
        value=auto(),
        n="covered",
        s=Node.News,
        t=Node.Topic,
        q=lambda s, t: (float(i["relevance_score"]) for i in t["topics"] if i["topic"] == s),
        attr={
            "time_decay": "s2c",  # From source to current
            "relevance": "s,t->q",
        },
    )

    Referenced = _edge(
        value=auto(),
        n="referenced",
        s=Node.News,
        t=Node.Equity,
        q=lambda s, t: (float(i["relevance_score"]) for i in t["ticker_sentiment"] if i["ticker"] == s),
        attr={
            "time_decay": "s2t",  # From source to target
            "relevance": "s,t->q",
            "sentiment": "s,t->q",
        },
    )

    Mentioned = _edge(
        value=auto(),
        n="mentioned",
        s=Node.News,
        t=Node.Company,
        q=lambda s, t: (float(i["relevance_score"]) for i in t["ticker_sentiment"] if i["ticker"] == s),
        attr={
            "time_decay": "s2c",  # From source to current
            "relevance": "s,t->q",
            "sentiment": "s,t->q",
        },
    )

    # TODO More...

    @property
    def n(self) -> str:
        return self.value.n

    @property
    def s(self) -> Node:
        return self.value.s

    @property
    def t(self) -> Node:
        return self.value.t

    @property
    def attr(self) -> Dict[str, Any]:
        return self.value.attr

    @classmethod
    def names(cls) -> Set[str]:
        return {m.value.n for m in cls._member_map_.values()}


@dataclass
class Edge:
    src_id: str
    tgt_id: str
    src_index: int
    tgt_index: int
    s: Node
    t: Node
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

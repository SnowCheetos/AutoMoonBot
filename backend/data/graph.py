import hashlib
import pandas as pd
import networkx as nx
from collections import deque
from typing import Dict, List

from backend.data import Edge, Node


class Graph:
    def __init__(
        self,
        prices: Dict[str, Dict[str, str | float]],
        news: List[Dict[str, str | List[str | Dict[str, str]]]],
        buffer_size: int,
    ) -> None:
        self.G = nx.Graph()
        self._buffer_size = buffer_size

        for ticker, price in prices.items():
            self.add_ticker(ticker, price, False)

        for content in news:
            self.add_news(content, False)

        self.compute_edges()

    @property
    def num_nodes(self) -> int:
        return len(self.G.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.G.edges)

    def to_pyg(self):
        pass

    def update_ticker(
        self,
        ticker: str,
        data: Dict[str, Dict[str, str | float]],
        compute_edges: bool = True,
    ) -> None:
        if not self.G.has_node(ticker):
            return

        node = self.G.nodes.get(ticker)
        prices = [
            {"0. time": ts, **{k: float(v) for k, v in p.items()}}
            for ts, p in reversed(data.items())
        ]
        for price in prices:
            node["prices"].append(price)
        if compute_edges:
            self.compute_node_edges(ticker)

    def add_ticker(
        self,
        ticker: str,
        prices: Dict[str, Dict[str, str | float]] | None = None,
        compute_edges: bool = True,
    ) -> None:
        if self.G.has_node(ticker):
            return

        if prices is None:
            data = []
        else:
            data = [
                {"0. time": ts, **{k: float(v) for k, v in p.items()}}
                for ts, p in reversed(prices.items())
            ]
        self.G.add_node(
            ticker,
            node_type=Node.Price,
            prices=deque(data, maxlen=self._buffer_size),
        )
        if compute_edges:
            self.compute_node_edges(ticker)

    def add_news(
        self,
        content: Dict[str, str | List[str | Dict[str, str]]],
        compute_edges: bool = True,
    ) -> None:
        url = content["url"]
        idx = hashlib.sha256(url.encode()).hexdigest()
        if self.G.has_node(idx):
            return

        self.G.add_node(idx, node_type=Node.News, **content)
        if compute_edges:
            self.compute_node_edges(idx)

    def del_ticker(self, ticker: str) -> None:
        if not self.G.has_node(ticker):
            return
        self.G.remove_node(ticker)

    def del_news(self, url: str) -> None:
        idx = hashlib.sha256(url.encode()).hexdigest()
        if not self.G.has_node(idx):
            return
        self.G.remove_node(idx)

    def compute_edges(self) -> None:
        for node_id in self.G.nodes:
            self.compute_node_edges(node_id)

    def compute_node_edges(self, node_id: str) -> None:
        curr = self.G.nodes.get(node_id)

        if curr["node_type"] == Node.News:
            curr_ticker = None
            curr_source = curr["source"]
            curr_authors = set(curr["authors"])
            curr_topics = {t["topic"] for t in curr["topics"]}
            curr_tickers = {t["ticker"] for t in curr["ticker_sentiment"]}
        elif curr["node_type"] == Node.Price:
            if len(curr["prices"]) == 0:
                return
            curr_ticker = node_id
            curr_source = None
            curr_authors = set()
            curr_topics = set()
            curr_tickers = set()

        for name, node in self.G.nodes.items():
            if name != node_id:
                if node["node_type"] == Node.News:
                    node_source = node["source"]
                    node_authors = set(node["authors"])
                    node_topics = {t["topic"] for t in node["topics"]}
                    node_tickers = {t["ticker"] for t in node["ticker_sentiment"]}

                    # Checks if the two nodes have the same publisher
                    if curr_source == node_source:
                        self.G.add_edge(node_id, name, edge_type=Edge.Publisher)

                    # Checks if the two nodes have common topics
                    if curr_topics:
                        common_topics = node_topics & curr_topics
                        unique_topics = node_topics.union(curr_topics)
                        common_ratio = len(common_topics) / len(unique_topics)
                        self.G.add_edge(
                            node_id,
                            name,
                            edge_type=Edge.Topics,
                            common_ratio=common_ratio,
                        )

                    # Checks if the two nodes have common tickers
                    if curr_tickers:
                        common_tickers = node_tickers & curr_tickers
                        unique_tickers = node_tickers.union(curr_tickers)
                        common_ratio = len(common_tickers) / len(unique_tickers)
                        self.G.add_edge(
                            node_id,
                            name,
                            edge_type=Edge.Tickers,
                            common_ratio=common_ratio,
                        )

                    # Checks if the two nodes have common authors
                    if curr_authors:
                        common_authors = node_authors & curr_authors
                        unique_authors = node_authors.union(curr_authors)
                        common_ratio = len(common_authors) / len(unique_authors)
                        self.G.add_edge(
                            node_id,
                            name,
                            edge_type=Edge.Authors,
                            common_ratio=common_ratio,
                        )

                    # Checks if the current ticker is in the node tickers
                    if curr_ticker in node_tickers:
                        relevance = [
                            t["relevance_score"]
                            for t in node["ticker_sentiment"]
                            if t["ticker"] == curr_ticker
                        ][0]
                        self.G.add_edge(
                            name,
                            node_id,
                            edge_type=Edge.Reference,
                            relevance=float(relevance),
                        )

                elif node["node_type"] == Node.Price:
                    if len(node["prices"]) == 0:
                        continue
                    node_ticker = name

                    # Checks if the node ticker is in current tickers
                    if node_ticker in curr_tickers:
                        relevance = [
                            t["relevance_score"]
                            for t in curr["ticker_sentiment"]
                            if t["ticker"] == node_ticker
                        ][0]
                        self.G.add_edge(
                            node_id,
                            name,
                            edge_type=Edge.Reference,
                            relevance=float(relevance),
                        )

                    if curr["node_type"] == Node.Price:
                        curr_df = (
                            pd.DataFrame.from_dict(curr["prices"])
                            .set_index("0. time")
                            .astype(float)
                        )
                        node_df = (
                            pd.DataFrame.from_dict(node["prices"])
                            .set_index("0. time")
                            .astype(float)
                        )
                        curr_df.columns = pd.MultiIndex.from_product(
                            [[curr_ticker], curr_df.columns]
                        )
                        node_df.columns = pd.MultiIndex.from_product(
                            [[node_ticker], node_df.columns]
                        )
                        corr = (
                            pd.concat((curr_df, node_df), axis=1)
                            .dropna()
                            .corr(method="spearman")
                            .loc[node_ticker]  # row
                            .loc[:, curr_ticker]  # col
                        )
                        # TODO Need to add a way to filter edges between tickers
                        self.G.add_edge(
                            node_id,
                            name,
                            edge_type=Edge.Correlation,
                            corr=corr.values,
                        )

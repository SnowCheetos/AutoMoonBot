import hashlib
import torch
import numpy as np
import pandas as pd
import networkx as nx
from torch import Tensor
from collections import deque
from typing import Dict, List, Type
from torch_geometric.data import HeteroData

from backend.data import EdgeType, NodeType, Edge


class Graph:
    def __init__(
        self,
        prices: Dict[str, Dict[str, str | float]],
        news: List[Dict[str, str | List[str | Dict[str, str]]]],
        buffer_size: int,
    ) -> None:
        self.G = nx.DiGraph()
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

    def node_index(self, node_type: Type[NodeType]) -> Dict[str, int]:
        nodes = [
            node
            for node, attr in self.G.nodes(data=True)
            if attr["node_type"] == node_type
        ]
        return {node: index for index, node in enumerate(nodes)}

    def node_features(self, node_type: Type[NodeType]) -> Tensor:
        nodes = self.node_index
        return torch.zeros((len(nodes), 100))

    def edge_index(self, edge_type: Type[EdgeType]) -> Tensor:
        edges = self.get_edges(edge_type)
        indices = [[edge.src_index, edge.tgt_index] for edge in edges]
        return torch.tensor(indices).long().contiguous().t()

    def edge_attr(self, edge_type: Type[EdgeType]) -> Tensor:
        edges = self.get_edges(edge_type)
        attrs = [
            [edge.edge_attr[k] for k in edge.edge_attr if k not in {"edge_type", ""}]
            for edge in edges
        ]
        return torch.tensor(attrs).float()

    def get_edges(self, edge_type: Type[EdgeType]) -> List[Edge]:
        src_index = self.node_index(edge_type.src_type)
        tgt_index = self.node_index(edge_type.tgt_type)

        edges = [
            Edge(
                src_id=src,
                tgt_id=tgt,
                src_index=src_index[src],
                tgt_index=tgt_index[tgt],
                src_type=edge_type.src_type,
                tgt_type=edge_type.tgt_type,
                edge_type=edge_type,
                edge_attr=attr,
            )
            for src, tgt, attr in self.G.edges(data=True)
            if attr.get("edge_type") == edge_type
        ]
        return edges

    def to_pyg(self) -> HeteroData:
        data = HeteroData()

        data["ticker"].x = ...
        data["news"].x = ...

        data["news", "references_to", "ticker"].edge_index = self.edge_index(
            EdgeType.Reference
        )
        data["news", "common_authors", "news"].edge_index = self.edge_index(
            EdgeType.Authors
        )
        data["news", "common_tickers", "news"].edge_index = self.edge_index(
            EdgeType.Tickers
        )
        data["news", "common_topics", "news"].edge_index = self.edge_index(
            EdgeType.Topics
        )
        data["ticker", "correlates_with", "ticker"].edge_index = self.edge_index(
            EdgeType.Correlation
        )

        data["news", "references_to", "ticker"].edge_attr = self.edge_attr(
            EdgeType.Reference
        )
        data["news", "common_authors", "news"].edge_attr = self.edge_attr(
            EdgeType.Authors
        )
        data["news", "common_tickers", "news"].edge_attr = self.edge_attr(
            EdgeType.Tickers
        )
        data["news", "common_topics", "news"].edge_attr = self.edge_attr(
            EdgeType.Topics
        )
        data["ticker", "correlates_with", "ticker"].edge_attr = self.edge_attr(
            EdgeType.Correlation
        )

        return data

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
            node_type=NodeType.Ticker,
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

        self.G.add_node(idx, node_type=NodeType.News, **content)
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

    def compute_node_edges(self, node_id: str, min_corr_norm: float = 2.5) -> None:
        curr = self.G.nodes.get(node_id)

        if curr["node_type"] == NodeType.News:
            curr_ticker = None
            curr_source = curr["source"]
            curr_authors = set(curr["authors"])
            curr_topics = {
                t["topic"]: float(t["relevance_score"]) for t in curr["topics"]
            }
            curr_tickers = {
                t["ticker"]: float(t["relevance_score"])
                for t in curr["ticker_sentiment"]
            }
        elif curr["node_type"] == NodeType.Ticker:
            if len(curr["prices"]) == 0:
                return
            curr_ticker = node_id
            curr_source = None
            curr_authors = {}
            curr_topics = {}
            curr_tickers = {}

        for name, node in self.G.nodes.items():
            if name != node_id:
                if node["node_type"] == NodeType.News:
                    node_source = node["source"]
                    node_authors = set(node["authors"])
                    node_topics = {
                        t["topic"]: float(t["relevance_score"]) for t in node["topics"]
                    }
                    node_tickers = {t["ticker"] for t in node["ticker_sentiment"]}

                    # Checks if the two nodes have the same publisher
                    if curr_source == node_source:
                        self.G.add_edge(node_id, name, edge_type=EdgeType.Publisher)

                    # Checks if the two nodes have common topics
                    if curr_topics:
                        common_topics = set(node_topics) & set(curr_topics)
                        unique_topics = set(node_topics).union(set(curr_topics))
                        common_ratio = len(common_topics) / len(unique_topics)
                        if common_ratio > 0:
                            curr_w = len(common_topics) / len(curr_topics)
                            node_w = len(common_topics) / len(node_topics)

                            curr_relevance = [curr_topics[k] for k in common_topics]
                            node_relevance = [node_topics[k] for k in common_topics]

                            cross_relevance = curr_w * np.mean(
                                curr_relevance
                            ) + node_w * np.mean(node_relevance)

                            self.G.add_edge(
                                node_id,
                                name,
                                edge_type=EdgeType.Topics,
                                common_ratio=common_ratio,
                                cross_relevance=cross_relevance,
                            )

                    # Checks if the two nodes have common tickers
                    if curr_tickers:
                        common_tickers = set(node_tickers) & set(curr_tickers)
                        unique_tickers = set(node_tickers).union(set(curr_tickers))
                        common_ratio = len(common_tickers) / len(unique_tickers)
                        if common_ratio > 0:
                            curr_w = len(common_tickers) / len(curr_tickers)
                            node_w = len(common_tickers) / len(node_tickers)

                            curr_relevance = [curr_tickers[k] for k in common_tickers]
                            node_relevance = [curr_tickers[k] for k in common_tickers]

                            cross_relevance = curr_w * np.mean(
                                curr_relevance
                            ) + node_w * np.mean(node_relevance)

                            self.G.add_edge(
                                node_id,
                                name,
                                edge_type=EdgeType.Tickers,
                                common_ratio=common_ratio,
                                cross_relevance=cross_relevance,
                            )

                    # Checks if the two nodes have common authors
                    if curr_authors:
                        common_authors = node_authors & curr_authors
                        unique_authors = node_authors.union(curr_authors)
                        common_ratio = len(common_authors) / len(unique_authors)
                        self.G.add_edge(
                            node_id,
                            name,
                            edge_type=EdgeType.Authors,
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
                            edge_type=EdgeType.Reference,
                            relevance=float(relevance),
                        )

                elif node["node_type"] == NodeType.Ticker:
                    if len(node["prices"]) > 0:
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
                                edge_type=EdgeType.Reference,
                                relevance=float(relevance),
                            )

                        if curr["node_type"] == NodeType.Ticker:
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
                            ).values

                            if np.linalg.norm(corr, ord=np.inf) > min_corr_norm:
                                self.G.add_edge(
                                    node_id,
                                    name,
                                    edge_type=EdgeType.Correlation,
                                    corr=corr.flatten(),
                                )

import json
import pytest
import networkx as nx
from typing import Tuple, Dict, List
from backend.data import Graph, N, E

BUFFER_SIZE = 10


@pytest.fixture
def data() -> Tuple[
    List[Dict[str, str | List[str | Dict[str, str]]]],
    Dict[str, Dict[str, str | float]],
]:
    with open("prices.json", "r") as f:
        _prices = json.load(f)

    with open("news.json", "r") as f:
        _news = json.load(f)

    prices = {
        item["Meta Data"]["2. Symbol"]: item["Time Series (1min)"] for item in _prices
    }
    news = _news["feed"]
    return prices, news


def test_graph_basics(
    data: Tuple[
        List[Dict[str, str | List[str | Dict[str, str]]]],
        Dict[str, Dict[str, str | float]],
    ]
) -> None:
    prices, news = data
    num_init_nodes = len(prices) + len(news)
    graph = Graph(BUFFER_SIZE, tickers=prices, news=news)

    assert graph.num_nodes == len(prices) + len(
        news
    ), "graph did not initialize with the right number of nodes"
    assert (
        graph.num_edges > 0
    ), "graph did not initialize with the right number of edges"

    nvda = graph.G.nodes.get("NVDA")
    assert (
        len(nvda["prices"]) == BUFFER_SIZE
    ), "graph did not properly clip node buffer size"

    graph.del_ticker("GLD")
    assert graph.num_nodes == num_init_nodes - 1, "graph did not properly delete ticker"

    graph.del_news(news[0]["url"])
    assert graph.num_nodes == num_init_nodes - 2, "graph did not properly delete news"

    num_edges = graph.num_edges
    graph.add_ticker("TEST")
    assert (
        graph.num_nodes == num_init_nodes - 1
    ), "graph did not properly add new ticker"
    assert graph.num_edges == num_edges, "graph did not properly skip edge computation"

    graph.update_ticker("TEST", prices["SPY"])
    test_prices = graph.G.nodes.get("TEST")["prices"]
    assert len(test_prices) == BUFFER_SIZE, "graph did not properly add data to ticker"
    assert graph.num_edges > num_edges, "graph did not properly update edges"

    news_index = graph.node_index(N.News)
    assert len(news_index) == len(news) - 1, "graph did not return proper news nodes"

    ticker_index = graph.node_index(N.Ticker)
    assert len(ticker_index) == len(prices), "graph did not return proper ticker nodes"

    edges = graph.get_edges(E.Authors)
    assert 0 < len(edges) < num_edges, "graph did not return proper edges"
    assert all(
        edge.edge_type == E.Authors for edge in edges
    ), "graph returned wrong edge type"
    assert all(
        edge.src_type == N.News for edge in edges
    ), "graph returned wrong src type for edge"
    assert all(
        edge.tgt_type == N.News for edge in edges
    ), "graph returned wrong tgt type for edge"

    edges = graph.get_edges(E.Tickers)
    assert 0 < len(edges) < num_edges, "graph did not return proper edges"
    assert all(
        edge.edge_type == E.Tickers for edge in edges
    ), "graph returned wrong edge type"
    assert all(
        edge.src_type == N.News for edge in edges
    ), "graph returned wrong src type for edge"
    assert all(
        edge.tgt_type == N.News for edge in edges
    ), "graph returned wrong tgt type for edge"

    edges = graph.get_edges(E.Correlation)
    assert 0 < len(edges) < num_edges, "graph did not return proper edges"
    assert all(
        edge.edge_type == E.Correlation for edge in edges
    ), "graph returned wrong edge type"
    assert all(
        edge.src_type == N.Ticker for edge in edges
    ), "graph returned wrong src type for edge"
    assert all(
        edge.tgt_type == N.Ticker for edge in edges
    ), "graph returned wrong tgt type for edge"

    edges = graph.get_edges(E.Reference)
    assert 0 < len(edges) < num_edges, "graph did not return proper edges"
    assert all(
        edge.edge_type == E.Reference for edge in edges
    ), "graph returned wrong edge type"
    assert all(
        edge.src_type == N.News for edge in edges
    ), "graph returned wrong src type for edge"
    assert all(
        edge.tgt_type == N.Ticker for edge in edges
    ), "graph returned wrong tgt type for edge"

    edges = graph.get_edges(E.Influence)
    assert 0 < len(edges) < num_edges, "graph did not return proper edges"
    assert all(
        edge.edge_type == E.Influence for edge in edges
    ), "graph returned wrong edge type"
    assert all(
        edge.src_type == N.Ticker for edge in edges
    ), "graph returned wrong src type for edge"
    assert all(
        edge.tgt_type == N.News for edge in edges
    ), "graph returned wrong tgt type for edge"


def test_graph_arithmetics(
    data: Tuple[
        List[Dict[str, str | List[str | Dict[str, str]]]],
        Dict[str, Dict[str, str | float]],
    ]
) -> None:
    prices, news = data
    graph = Graph(BUFFER_SIZE, tickers=prices, news=news)

    node_index = graph.node_index(N.Ticker)
    node_features = graph.node_features(N.Ticker)
    assert node_features.shape[0] == len(
        node_index
    ), "Node features have wrong row dimensions"
    assert node_features.shape[1] == graph.node_feature_size(
        N.Ticker
    ), "Node features have wrong col dimensions"

    node_index = graph.node_index(N.News)
    node_features = graph.node_features(N.News)
    assert node_features.shape[0] == len(
        node_index
    ), "Node features have wrong row dimensions"
    assert node_features.shape[1] == graph.node_feature_size(
        N.News
    ), "Node features have wrong col dimensions"

    # TODO actually implement the features


def test_graph_exports(
    data: Tuple[
        List[Dict[str, str | List[str | Dict[str, str]]]],
        Dict[str, Dict[str, str | float]],
    ]
) -> None:
    prices, news = data
    graph = Graph(BUFFER_SIZE, tickers=prices, news=news)

    data_pyg = graph.to_pyg()

    assert data_pyg[N.Ticker.S].x.size(0) == len(
        prices
    ), "graph exported the wrong number of ticker nodes to pyg"
    assert data_pyg[N.News.S].x.size(0) == len(
        news
    ), "graph exported the wrong number of news nodes to pyg"

    for edge_type in E.names():
        assert (
            data_pyg[edge_type].edge_index.size(0) == 2
        ), f"graph exported the wrong dimension of {edge_type} edges indices to pyg"
        assert data_pyg[
            edge_type
        ].edge_index.is_contiguous(), (
            f"graph exported non-contiguous {edge_type} edges indices to pyg"
        )
        assert data_pyg[
            edge_type
        ].edge_attr.is_contiguous(), (
            f"graph exported non-contiguous {edge_type} edges indices to pyg"
        )
        attrs_rows = data_pyg[edge_type].edge_attr.size(0)
        index_cols = data_pyg[edge_type].edge_index.size(1)
        assert (
            attrs_rows == index_cols
        ), f"graph exported inconsistent dimensions for {edge_type} index and attributes to pyg"

import json
import pytest
from typing import Tuple, Dict, List
from backend.data import Graph, NodeType, EdgeType

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
    graph = Graph(prices, news, BUFFER_SIZE)

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

    news_index = graph.node_index(NodeType.News)
    assert len(news_index) == len(news) - 1, "graph did not return proper news nodes"

    ticker_index = graph.node_index(NodeType.Ticker)
    assert len(ticker_index) == len(prices), "graph did not return proper ticker nodes"

    edges = graph.get_edges(EdgeType.Authors)
    assert 0 < len(edges) < num_edges, "graph did not return proper edges"
    assert all(
        edge.edge_type == EdgeType.Authors for edge in edges
    ), "graph returned wrong edge type"
    assert all(
        edge.src_type == NodeType.News for edge in edges
    ), "graph returned wrong src type for edge"
    assert all(
        edge.tgt_type == NodeType.News for edge in edges
    ), "graph returned wrong tgt type for edge"

    edges = graph.get_edges(EdgeType.Correlation)
    assert 0 < len(edges) < num_edges, "graph did not return proper edges"
    assert all(
        edge.edge_type == EdgeType.Correlation for edge in edges
    ), "graph returned wrong edge type"
    assert all(
        edge.src_type == NodeType.Ticker for edge in edges
    ), "graph returned wrong src type for edge"
    assert all(
        edge.tgt_type == NodeType.Ticker for edge in edges
    ), "graph returned wrong tgt type for edge"

    edges = graph.get_edges(EdgeType.Reference)
    assert 0 < len(edges) < num_edges, "graph did not return proper edges"
    assert all(
        edge.edge_type == EdgeType.Reference for edge in edges
    ), "graph returned wrong edge type"
    assert all(
        edge.src_type == NodeType.News for edge in edges
    ), "graph returned wrong src type for edge"
    assert all(
        edge.tgt_type == NodeType.Ticker for edge in edges
    ), "graph returned wrong tgt type for edge"

    edges = graph.get_edges(EdgeType.Influence)
    assert 0 < len(edges) < num_edges, "graph did not return proper edges"
    assert all(
        edge.edge_type == EdgeType.Influence for edge in edges
    ), "graph returned wrong edge type"
    assert all(
        edge.src_type == NodeType.Ticker for edge in edges
    ), "graph returned wrong src type for edge"
    assert all(
        edge.tgt_type == NodeType.News for edge in edges
    ), "graph returned wrong tgt type for edge"


def test_graph_arithmetics(
    data: Tuple[
        List[Dict[str, str | List[str | Dict[str, str]]]],
        Dict[str, Dict[str, str | float]],
    ]
) -> None:
    prices, news = data
    num_init_nodes = len(prices) + len(news)
    graph = Graph(prices, news, BUFFER_SIZE)

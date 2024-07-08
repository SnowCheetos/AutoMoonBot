import json
import pytest
import networkx as nx
from typing import Tuple, Dict, List
from backend.data import Graph

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
):
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

    graph.update_ticker(
        "TEST",
        {
            "2024-07-05 18:29:00": {
                "1. open": "553.9800",
                "2. high": "554.0800",
                "3. low": "553.9800",
                "4. close": "553.9800",
                "5. volume": "15",
            }
        },
    )
    test_prices = graph.G.nodes.get("TEST")["prices"]
    assert len(test_prices) == 1, "graph did not properly add data to ticker"

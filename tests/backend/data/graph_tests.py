import json
import pytest
import networkx as nx
from typing import Tuple, Dict, List
from backend.data import graph as g, nodes as n, edges as e


@pytest.fixture
def data() -> Tuple[
    List[Dict[str, str | List[str | Dict[str, str]]]],
    Dict[str, Dict[str, str | float]],
]:
    return {
        n.Company: [
            {"x": 1, "y": True, "symbol": "foo"},
            {"x": 2, "y": False, "symbol": "bar"},
        ],
        n.Equity: [
            {"a": True, "symbol": "bar"},
            {"a": False, "symbol": "foo"},
        ],
    }


def test_basics(data):
    graph = g.HeteroGraph()

    graph.add_node(n.Company, "c1", **data[n.Company][0])
    assert graph.number_of_nodes() == 1, "graph did not properly add node"

    element = graph.nodes.get("c1")["element"]
    assert element == n.Company, "graph did not properly add node element"
    assert element.x == 1, "graph did not properly add element attrs"
    assert element.y, "graph did not properly add element attrs"
    assert element.symbol == "foo", "graph did not properly add element attrs"

    graph.add_node(n.Company, "c2", **data[n.Company][1])
    assert graph.number_of_nodes() == 2, "graph did not properly add node"

    element = graph.nodes.get("c2")["element"]
    assert element == n.Company, "graph did not properly add node element"
    assert element.x == 2, "graph did not properly add element attrs"
    assert not element.y, "graph did not properly add element attrs"
    assert element.symbol == "bar", "graph did not properly add element attrs"
    assert graph.number_of_edges() == 0, "graph has wrong number of edges"

    graph.compute_edges()
    assert graph.number_of_edges() == 0, "graph has wrong number of edges"

    graph.add_node(n.Equity, "e1", **data[n.Equity][0])
    assert graph.number_of_nodes() == 3, "graph did not properly add node"

    element = dict(graph.nodes(data="element")).get("e1")
    assert element == n.Equity, "graph did not properly add node element"
    assert element.a, "graph did not properly add element attrs"
    assert element.symbol == "bar", "graph did not properly add element attrs"
    assert graph.number_of_edges() == 0, "graph has wrong number of edges"

    graph.compute_edges()
    assert graph.number_of_edges() == 1, "graph has wrong number of edges"

    edge = graph.get_edge_data("c1", "e1", e.Issues.name)
    assert not edge, "graph added wrong edge"

    edge = graph.get_edge_data("c2", "e1", e.Issues.name)
    assert edge, "graph did not properly add edge"

    element = edge.get("element")
    assert element == e.Issues, "graph added wrong edge type"

    graph.clear()
    assert graph.number_of_nodes() == 0, "graph did not proper clear nodes"
    assert graph.number_of_edges() == 0, "graph did not proper clear edges"

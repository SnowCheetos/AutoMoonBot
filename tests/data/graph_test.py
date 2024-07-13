import pytest
from typing import Tuple, Dict, List
from automoonbot.data import graph as g, nodes as n, edges as e


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

    companies = graph.get_nodes(n.Company)
    assert len(companies) == 2, "graph memo stored the wrong number of nodes"

    graph.compute_edges()
    assert graph.number_of_edges() == 1, "graph has wrong number of edges"

    edges = graph.get_edge_elements(e.Issues)
    assert len(edges) == 1, "graph edge memo stored wrong number of edges"

    edge = graph.get_edge_data("c1", "e1", e.Issues.name)
    assert not edge, "graph added wrong edge"

    edge = graph.get_edge_data("c2", "e1", e.Issues.name)
    assert edge, "graph did not properly add edge"

    element = edge.get("element")
    assert element == e.Issues, "graph added wrong edge type"
    assert element.source == "c2", "edge src element wrong value"
    assert element.target == "e1", "edge tgt element wrong value"

    pyg = graph.pyg
    assert pyg[n.Company.name].x.size(0) == 2, "wrong rows of company nodes"
    assert (
        pyg[n.Company.name].x.size(1) == n.Company.tensor_dim
    ), "wrong cols of company nodes"
    assert pyg[n.Equity.name].x.size(0) == 1, "wrong rows of equity nodes"
    assert (
        pyg[n.Equity.name].x.size(1) == n.Equity.tensor_dim
    ), "wrong cols of equity nodes"
    assert pyg[e.Issues.name].edge_index.size(0) == 2, "wrong rows of edge index"
    assert pyg[e.Issues.name].edge_index.size(1) == 1, "wrong cols of edge index"
    assert pyg[e.Issues.name].edge_index.is_contiguous(), "edge index not contiguous"
    assert pyg[e.Issues.name].edge_attr.size(0) == 1, "wrong rows of edge attr"
    assert (
        pyg[e.Issues.name].edge_attr.size(1) == e.Issues.tensor_dim
    ), "wrong cols of edge attr"
    assert pyg[e.Issues.name].edge_attr.is_contiguous(), "edge attr not contiguous"
    assert pyg.is_directed(), "pyg outputted undirected graph"
    assert pyg.has_isolated_nodes(), "pyg did not pick up isolated node"

    graph.clear()
    assert graph.number_of_nodes() == 0, "graph did not proper clear nodes"
    assert graph.number_of_edges() == 0, "graph did not proper clear edges"

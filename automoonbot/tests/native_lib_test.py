import pytest
import time


def test_import():
    try:
        import moonrs
    except ImportError:
        pytest.fail("Failed to import the moonrs module")

    assert hasattr(moonrs, "__all__"), "moonrs module has no __all__ attribute"
    assert len(moonrs.__all__) > 0, "moonrs module has empty content"
    assert moonrs.hello_python() == "Hello From Rust", "incorrect hello message"


def test_basics():
    try:
        from moonrs import HeteroGraph
    except ImportError:
        pytest.fail("Failed to import the moonrs module")
    assert (
        HeteroGraph.hello_python() == "Hello From HeteroGraph"
    ), "incorrect hello message"

    graph = HeteroGraph()
    assert graph.node_count() == 0, "incorrect node count"
    assert graph.edge_count() == 0, "incorrect edge count"

    graph.add_test_node(name="test", value=1.0, capacity=1)
    assert graph.node_count() == 1, "incorrect node count"
    assert graph.edge_count() == 0, "incorrect edge count"

    graph.remove_node(name="test")
    assert graph.node_count() == 0, "incorrect node count"

    graph.add_test_node(name="node_1", value=1.0, capacity=1)
    assert graph.node_count() == 1, "incorrect node count"
    assert graph.edge_count() == 0, "incorrect edge count"

    graph.add_test_node(name="node_2", value=2.0, capacity=1)
    assert graph.node_count() == 2, "incorrect node count"
    assert graph.edge_count() == 2, "incorrect edge count"

    graph.add_test_node(name="node_3", value=2.0, capacity=1)
    assert graph.node_count() == 3, "incorrect node count"
    assert graph.edge_count() == 4, "incorrect edge count"

    graph.remove_node(name="node_1")
    assert graph.node_count() == 2, "incorrect node count"
    assert graph.edge_count() == 0, "incorrect edge count"


def test_wrapping():
    try:
        from moonrs import HeteroGraph
    except ImportError:
        pytest.fail("Failed to import the moonrs module")

    class Wrapper(HeteroGraph):
        def __init__(self):
            super().__init__()

    wrapper = Wrapper()
    assert wrapper.node_count() == 0, "incorrect node count"
    assert wrapper.edge_count() == 0, "incorrect edge count"


def test_currency():
    try:
        from moonrs import HeteroGraph
    except ImportError:
        pytest.fail("Failed to import the moonrs module")

    graph = HeteroGraph()
    graph.add_currency(symbol="USD", capacity=10)
    graph.add_currency(symbol="EUR", capacity=10)

    assert graph.node_count() == 2, "incorrect node count"

    graph.update_currency(
        symbol="USD",
        timestamp=time.time(),
        duration=60,
        adjusted=True,
        open=1.0,
        high=1.1,
        low=0.9,
        close=1.0,
        volume=100,
    )

    graph.update_currency(
        symbol="EUR",
        timestamp=time.time(),
        duration=60,
        adjusted=True,
        open=1.0,
        high=1.1,
        low=0.9,
        close=1.0,
        volume=100,
    )

    x, edge_index, edge_attr = graph.to_pyg()
    assert len(x) == 1, "incorrect node class count"
    assert len(edge_index) == 0, "incorrect edge count"
    assert len(edge_attr) == 0, "incorrect edge count"
    assert len(x.get("Currency")) == 2, "incorrect currency node count"


def test_equity():
    try:
        from moonrs import HeteroGraph
    except ImportError:
        pytest.fail("Failed to import the moonrs module")

    graph = HeteroGraph()
    graph.add_equity(symbol="foo", company="", capacity=10)
    graph.add_equity(symbol="bar", company="", capacity=10)

    assert graph.node_count() == 2, "incorrect node count"
    assert graph.edge_count() == 0, "incorrect edge count"

    graph.update_equity(
        symbol="foo",
        timestamp=time.time(),
        duration=60,
        adjusted=True,
        open=1.0,
        high=1.1,
        low=0.9,
        close=1.0,
        volume=100,
    )

    graph.update_equity(
        symbol="bar",
        timestamp=time.time(),
        duration=60,
        adjusted=True,
        open=1.0,
        high=1.1,
        low=0.9,
        close=1.0,
        volume=100,
    )

    x, edge_index, edge_attr = graph.to_pyg()
    print(x, edge_attr, edge_index)
    assert len(x.get("Equity")) == 2, "incorrect currency node count"
    assert len(x) == 1, "incorrect node class count"
    assert len(edge_index) == 2, "incorrect edge count"
    assert len(edge_attr) == 2, "incorrect edge count"

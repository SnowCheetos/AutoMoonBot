import pytest


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

    graph.add_test_node(name="test", value=1.0)
    assert graph.node_count() == 1, "incorrect node count"
    assert graph.edge_count() == 0, "incorrect edge count"

    graph.remove_node(name="test")
    assert graph.node_count() == 0, "incorrect node count"

    graph.add_test_node(name="node_1", value=1.0)
    assert graph.node_count() == 1, "incorrect node count"
    assert graph.edge_count() == 0, "incorrect edge count"

    graph.add_test_node(name="node_2", value=2.0)
    assert graph.node_count() == 2, "incorrect node count"
    assert graph.edge_count() == 2, "incorrect edge count"

    graph.add_test_node(name="node_3", value=2.0)
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
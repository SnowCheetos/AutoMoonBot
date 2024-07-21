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
    assert graph.edge_count() == 0, "incorrect node count"
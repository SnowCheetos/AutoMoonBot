"""End-to-end tests of the Python API over the Rust store."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from automoonbot import OPEN_END, Channel, TemporalGraph, to_ns

DAY = 86_400 * 10**9


def t(day: float) -> int:
    return int(day * DAY)


@pytest.fixture
def g() -> TemporalGraph:
    g = TemporalGraph()
    g.define_entity("company", [Channel("fundamentals", 3, nullable=True)])
    g.define_entity("equity", [Channel("bar", 5)])
    g.define_event("article", window=dt.timedelta(days=3), channels=[Channel("embedding", 4)])
    g.define_relation("issues", "company", "equity")
    g.define_relation("supplies", "company", "company", 1, bounds=(0.0, 1.0))
    g.define_relation("mentions", "article", "company", 2, bounds=(-1.0, 1.0))
    g.upsert_nodes("company", ["tsmc", "apple", "nvidia"], known_time=0)
    g.upsert_nodes("equity", ["TSM", "AAPL", "NVDA"], known_time=0)
    g.assert_edges("issues", ["tsmc", "apple", "nvidia"], ["TSM", "AAPL", "NVDA"], known_time=0)
    return g


def test_time_conversion() -> None:
    assert to_ns("1970-01-01T00:00:01Z") == 10**9
    assert to_ns("1970-01-01T01:00:00+01:00") == 0
    assert to_ns(np.datetime64("1970-01-02", "D")) == DAY
    assert (
        to_ns(dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(microseconds=5))
        == 5000
    )
    with pytest.raises(ValueError, match="naive"):
        to_ns(dt.datetime(2024, 1, 1))
    with pytest.raises(ValueError, match="naive"):
        to_ns("2024-01-01T00:00:00")
    with pytest.raises(TypeError):
        to_ns(True)
    with pytest.raises(TypeError):
        to_ns(np.timedelta64(1, "D"))


def test_event_window_units() -> None:
    a, b = TemporalGraph(), TemporalGraph()
    a.define_event("article", window=np.timedelta64(3, "D"))
    b.define_event("article", window=dt.timedelta(days=3))
    assert a.schema == b.schema
    assert a.schema["kinds"][0]["window_ns"] == 3 * DAY


def test_bars_are_invisible_until_known(g: TemporalGraph) -> None:
    # A daily bar valid at the close (day 1), published 15 minutes later.
    close, published = t(1), t(1) + 15 * 60 * 10**9
    g.observe("equity", "bar", "AAPL", close, published, [100, 101, 99, 100.5, 1e6])
    before = g.snapshot(published - 1, close)
    after = g.snapshot(published, close)
    row = before["nodes"]["equity"]["keys"].index("AAPL")
    assert before["nodes"]["equity"]["channels"]["bar"]["present"][row, 0] == 0
    assert after["nodes"]["equity"]["channels"]["bar"]["present"][row, 0] == 1
    np.testing.assert_allclose(
        after["nodes"]["equity"]["channels"]["bar"]["values"][row, 0], [100, 101, 99, 100.5, 1e6]
    )


def test_history_layout(g: TemporalGraph) -> None:
    days = np.arange(1, 6)
    g.observe(
        "equity", "bar", ["NVDA"] * 5, days * DAY, days * DAY + 1, np.repeat(days, 5).reshape(5, 5)
    )
    s = g.snapshot(t(10), history={"equity.bar": 7})
    bar = s["nodes"]["equity"]["channels"]["bar"]
    row = s["nodes"]["equity"]["keys"].index("NVDA")
    assert bar["values"].shape == (3, 7, 5)
    np.testing.assert_array_equal(bar["present"][row], [0, 0, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(bar["values"][row, :, 0], [0, 0, 1, 2, 3, 4, 5])
    assert bar["values"].dtype == np.float32 and bar["values"].flags.c_contiguous


def test_restatement_and_nullable(g: TemporalGraph) -> None:
    q = t(90)
    g.observe(
        "company",
        "fundamentals",
        ["apple", "apple"],
        [q, q],
        [t(100), t(200)],
        [[10.0, np.nan, 1.0], [8.0, 2.0, 1.0]],
    )
    f = lambda k: g.snapshot(t(k))["nodes"]["company"]["channels"]["fundamentals"]  # noqa: E731
    row = g.snapshot(t(150))["nodes"]["company"]["keys"].index("apple")
    np.testing.assert_array_equal(f(150)["values"][row, 0], [10.0, 0.0, 1.0])
    np.testing.assert_array_equal(f(150)["element_present"][row, 0], [1, 0, 1])
    np.testing.assert_array_equal(f(250)["values"][row, 0], [8.0, 2.0, 1.0])


def test_atomic_and_idempotent_batches(g: TemporalGraph) -> None:
    rows = dict(
        keys=["AAPL", "TSM"],
        valid_time=[t(1), t(1)],
        known_time=[t(1), t(1)],
        values=np.ones((2, 5)),
    )
    assert g.observe("equity", "bar", **rows) == {"new": 2, "duplicate": 0}
    assert g.observe("equity", "bar", **rows) == {"new": 0, "duplicate": 2}
    with pytest.raises(ValueError, match="conflicting"):
        g.observe("equity", "bar", ["NVDA", "AAPL"], [t(1), t(1)], [t(1), t(1)], np.zeros((2, 5)))
    assert g.observation_count("equity", "bar") == 2  # NVDA row rolled back
    with pytest.raises(ValueError, match="bounds"):
        g.assert_edges("supplies", "tsmc", "apple", known_time=0, attrs=[[1.5]])
    with pytest.raises(KeyError):
        g.observe("equity", "bar", "GOOG", t(1), t(1), np.ones(5))
    with pytest.raises(ValueError, match="NaN"):
        g.observe("equity", "bar", "AAPL", t(2), t(2), [1, 1, 1, np.nan, 1])


def test_relationships_end_when_known(g: TemporalGraph) -> None:
    g.assert_edges("supplies", "tsmc", "apple", known_time=t(10), valid_from=t(5), attrs=[[0.25]])
    g.retract_edges("supplies", "tsmc", "apple", known_time=t(40), valid_from=t(30))

    def n(known: int, valid: int) -> int:
        return int(g.snapshot(t(known), t(valid))["edges"]["supplies"]["edge_index"].shape[1])

    assert n(9, 20) == 0
    assert n(10, 20) == 1
    assert n(35, 35) == 1, "termination not yet known"
    assert n(40, 35) == 0
    assert n(40, 25) == 1


def test_articles_window_and_mentions(g: TemporalGraph) -> None:
    g.upsert_nodes("article", ["a1", "a2"], known_time=[t(1), t(5)], valid_from=[t(1), t(5)])
    g.observe(
        "article", "embedding", ["a1", "a2"], [t(1), t(5)], [t(1), t(5)], np.full((2, 4), 0.1)
    )
    g.assert_edges(
        "mentions",
        ["a1", "a2"],
        ["nvidia", "apple"],
        known_time=[t(1), t(5)],
        valid_from=[t(1), t(5)],
        attrs=[[0.8, 0.9], [-0.5, 0.7]],
    )
    s = g.snapshot(t(6))
    assert s["nodes"]["article"]["keys"] == ["a2"]
    ei = s["edges"]["mentions"]["edge_index"]
    assert ei.shape == (2, 1)
    assert s["nodes"]["company"]["keys"][ei[1, 0]] == "apple"
    wide = g.snapshot(t(6), windows={"article": dt.timedelta(days=30)})
    assert wide["nodes"]["article"]["keys"] == ["a1", "a2"]


def test_homogeneous_view(g: TemporalGraph) -> None:
    s = g.snapshot(t(1), homogeneous="reverse")
    h = s["homogeneous"]
    n = sum(len(b["keys"]) for b in s["nodes"].values())
    assert h["kind_offsets"][-1] == n
    assert h["edge_index"].shape == (2, 6)  # 3 issues edges, both directions
    assert np.all(np.diff(h["indptr"]) >= 0) and h["indptr"][-1] == 6
    assert set(h["edge_type"].tolist()) == {0, 3}  # issues=0, rev_issues=0+3


def test_persistence(g: TemporalGraph, tmp_path) -> None:  # type: ignore[no-untyped-def]
    g.observe("equity", "bar", "AAPL", t(1), t(1), np.arange(5))
    p = tmp_path / "store.moon"
    g.save(p)
    h = TemporalGraph.load(p)
    a, b = g.snapshot(t(2), history={"equity.bar": 2}), h.snapshot(t(2), history={"equity.bar": 2})
    np.testing.assert_array_equal(
        a["nodes"]["equity"]["channels"]["bar"]["values"],
        b["nodes"]["equity"]["channels"]["bar"]["values"],
    )
    data = bytearray(p.read_bytes())
    data[-1] ^= 1
    p.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="checksum"):
        TemporalGraph.load(p)
    assert h.max_known_time() == t(1)


def test_open_end_constant() -> None:
    assert OPEN_END == 2**63 - 1


def test_pyg_conversion(g: TemporalGraph) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from automoonbot.pyg import to_hetero_data, to_token_graph

    g.observe("equity", "bar", "AAPL", t(1), t(1), np.arange(5))
    s = g.snapshot(t(2), homogeneous="reverse")
    data = to_hetero_data(s)
    assert data["equity"].x.shape == (3, 5 + 1 + 1)
    assert data["company", "issues", "equity"].edge_index.shape == (2, 3)
    tok = to_token_graph(s)
    assert tok["edge_index"].shape == (2, 6)
    assert tok["x::equity"].shape[0] == 3

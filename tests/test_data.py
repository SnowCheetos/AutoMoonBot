"""Tests for the data layer: store semantics, EDGAR normalization, and
bitemporal fundamentals.

EDGAR payloads below are synthetic (a fictional issuer, CIK 1234567) but
follow the documented shapes of the submissions and companyfacts APIs.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pytest
from automoonbot.data import ConflictError, ParquetStore
from automoonbot.data.fundamentals import FEATURE_NAMES, quarterly_observations
from automoonbot.data.http import Response
from automoonbot.data.ingest import (
    define_core_schema,
    load_entities,
    load_filings,
    load_fundamentals,
)
from automoonbot.data.providers.edgar import EdgarProvider, entity_id
from automoonbot.data.schemas import BARS, TS

from automoonbot import TemporalGraph, to_ns

CIK = 1234567
EID = "cik:0001234567"

# Fiscal year = calendar year. Q1..Q3 10-Qs, then a 10-K, then a 10-K/A that
# restates FY revenue.
FILINGS = [
    ("0001234567-24-000010", "10-Q", "2024-05-01", "2024-05-01T16:05:00.000Z", "2024-03-31"),
    ("0001234567-24-000020", "10-Q", "2024-08-01", "2024-08-01T16:05:00.000Z", "2024-06-30"),
    ("0001234567-24-000030", "10-Q", "2024-10-31", "2024-10-31T16:05:00.000Z", "2024-09-30"),
    ("0001234567-25-000005", "10-K", "2025-02-14", "2025-02-14T07:30:00.000Z", "2024-12-31"),
    ("0001234567-25-000009", "10-K/A", "2025-03-20", "2025-03-20T09:00:00.000Z", "2024-12-31"),
]


def submissions() -> dict[str, Any]:
    cols = list(zip(*FILINGS, strict=True))
    return {
        "cik": str(CIK),
        "name": "Fictional Widgets Inc.",
        "tickers": ["FWID"],
        "filings": {
            "recent": {
                "accessionNumber": list(cols[0]),
                "form": list(cols[1]),
                "filingDate": list(cols[2]),
                "acceptanceDateTime": list(cols[3]),
                "reportDate": list(cols[4]),
                "primaryDocument": ["doc.htm"] * len(FILINGS),
            },
            "files": [],
        },
    }


def fact(start: str | None, end: str, val: float, accn: str, filed: str) -> dict[str, Any]:
    out: dict[str, Any] = {"end": end, "val": val, "accn": accn, "filed": filed, "form": "10-Q"}
    if start:
        out["start"] = start
    return out


def companyfacts() -> dict[str, Any]:
    q1, q2, q3, k, ka = (f[0] for f in FILINGS)
    revenue = [
        fact("2024-01-01", "2024-03-31", 100.0, q1, "2024-05-01"),
        fact("2024-04-01", "2024-06-30", 110.0, q2, "2024-08-01"),
        fact("2024-07-01", "2024-09-30", 120.0, q3, "2024-10-31"),
        fact("2024-01-01", "2024-12-31", 460.0, k, "2025-02-14"),  # Q4 = 130
        fact("2024-01-01", "2024-12-31", 450.0, ka, "2025-03-20"),  # restated: Q4 = 120
    ]
    # Cash flow is reported year-to-date.
    ocf = [
        fact("2024-01-01", "2024-03-31", 30.0, q1, "2024-05-01"),
        fact("2024-01-01", "2024-06-30", 70.0, q2, "2024-08-01"),
    ]
    assets = [fact(None, "2024-06-30", 1000.0, q2, "2024-08-01")]
    # A fact whose filing is missing from submissions -> filed-date fallback.
    orphan = [fact(None, "2023-12-31", 900.0, "0001234567-24-000001", "2024-02-01")]
    return {
        "cik": CIK,
        "facts": {
            "us-gaap": {
                "Revenues": {"units": {"USD": revenue}},
                "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": ocf}},
                "Assets": {"units": {"USD": assets + orphan}},
            },
            "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": []}}},
        },
    }


class FakeClient:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def get(self, url: str, *, use_cache: bool = True) -> Response:
        self.urls.append(url)
        if url.endswith(f"submissions/CIK{CIK:010d}.json"):
            body = submissions()
        elif url.endswith(f"companyfacts/CIK{CIK:010d}.json"):
            body = companyfacts()
        else:
            raise AssertionError(url)
        return Response(url, 200, json.dumps(body).encode(), to_ns("2025-06-01T00:00:00Z"), False)


@pytest.fixture
def edgar() -> EdgarProvider:
    return EdgarProvider(client=FakeClient())  # type: ignore[arg-type]


def test_entity_ids() -> None:
    assert entity_id(320193) == "cik:0000320193"
    assert entity_id("cik:320193") == "cik:0000320193"
    for bad in ["", "abc", "0", "12345678901", "-1"]:
        with pytest.raises(ValueError):
            entity_id(bad)


def test_requires_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(ValueError, match="User-Agent"):
        EdgarProvider()


def test_acceptance_time_is_conservative(edgar: EdgarProvider) -> None:
    # "...T16:05:00.000Z" is read as 16:05 New York (EDT, UTC-4) = 20:05 UTC.
    assert edgar.acceptance_ns("2024-05-01T16:05:00.000Z") == to_ns("2024-05-01T20:05:00Z")
    # Winter: EST, UTC-5.
    assert edgar.acceptance_ns("2025-02-14T07:30:00.000Z") == to_ns("2025-02-14T12:30:00Z")
    t = edgar.filings_table(submissions())
    assert t.column("known_time").cast(pa.int64())[0].as_py() == to_ns("2024-05-01T20:06:00Z")


def test_store_semantics(tmp_path: Path, edgar: EdgarProvider) -> None:
    store = ParquetStore(tmp_path)
    r1 = edgar.ingest(store, "filings", [str(CIK)])
    assert (r1.new, r1.duplicate) == (5, 0)
    r2 = edgar.ingest(store, "filings", [str(CIK)])
    assert (r2.new, r2.duplicate) == (0, 5)
    assert len(list(tmp_path.rglob("part-*.parquet"))) == 2  # 2024 and 2025 partitions

    # A conflicting version (same accession and known_time, different form).
    bad = store.read("filings").slice(0, 1)
    bad = bad.set_column(bad.schema.get_field_index("form"), "form", pa.array(["8-K"]))
    with pytest.raises(ConflictError):
        store.write("filings", bad)

    # Point-in-time reads.
    as_of = to_ns("2024-09-01T00:00:00Z")
    assert store.read("filings", as_of).num_rows == 2
    assert store.read("filings").num_rows == 5
    assert store.compact("filings") == 0


def test_store_rejects_bad_tables(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    naive = pa.table(
        {
            name: pa.array([], type=f.type)
            for name, f in zip(BARS.schema.names, BARS.schema, strict=True)
        }
    )
    naive = naive.set_column(
        naive.schema.get_field_index("known_time"), "known_time", pa.array([], pa.timestamp("ns"))
    )
    with pytest.raises(ValueError, match="timezone"):
        store.write("bars", naive)
    with pytest.raises(ValueError, match="columns"):
        store.write("bars", pa.table({"security_id": ["x"]}))


def test_batch_internal_conflicts_and_latest(tmp_path: Path) -> None:
    store = ParquetStore(tmp_path)
    t0, t1 = to_ns("2024-01-02T21:00:00Z"), to_ns("2024-01-03T21:00:00Z")

    def bars(close: list[float], known: list[int]) -> pa.Table:
        n = len(close)
        return pa.table(
            {
                "security_id": ["s1"] * n,
                "start": pa.array([to_ns("2024-01-02T14:30:00Z")] * n).cast(TS),
                "end": pa.array([to_ns("2024-01-02T21:00:00Z")] * n).cast(TS),
                "open": [1.0] * n,
                "high": [1.0] * n,
                "low": [1.0] * n,
                "close": close,
                "volume": [1.0] * n,
                "vwap": pa.array([None] * n, pa.float64()),
                "known_time": pa.array(known).cast(TS),
                "source": ["test"] * n,
            }
        )

    with pytest.raises(ConflictError):
        store.write("bars", bars([1.0, 2.0], [t0, t0]))
    assert store.write("bars", bars([1.0, 1.0], [t0, t0])).new == 1
    store.write("bars", bars([1.5], [t1]))  # a vendor correction a day later
    assert store.read("bars", latest=True).column("close").to_pylist() == [1.5]
    assert store.read("bars", t0, latest=True).column("close").to_pylist() == [1.0]
    assert store.compact("bars") == 2  # two parts merged into one new file
    assert len(list(tmp_path.rglob("part-*.parquet"))) == 1
    assert store.read("bars", latest=True).column("close").to_pylist() == [1.5]


def test_quarterly_fundamentals_are_bitemporal(tmp_path: Path, edgar: EdgarProvider) -> None:
    store = ParquetStore(tmp_path)
    edgar.ingest(store, "facts", [str(CIK)])
    obs = quarterly_observations(store.read("facts"))
    rev, ocf, assets = (
        FEATURE_NAMES.index(n) for n in ("revenue", "operating_cash_flow", "total_assets")
    )
    by_q: dict[dt.date, list[Any]] = {}
    for o in obs:
        by_q.setdefault(o.period_end, []).append(o)
    q4 = by_q[dt.date(2024, 12, 31)]
    # Q4 revenue derived as FY - 9M once the 10-K is known, then restated.
    assert [o.values[rev] for o in q4] == [130.0, 120.0]
    assert q4[0].known_time == to_ns("2025-02-14T12:31:00Z")
    assert q4[1].known_time == to_ns("2025-03-20T13:01:00Z")
    # Q2 cash flow derived from year-to-date figures (70 - 30).
    q2 = by_q[dt.date(2024, 6, 30)]
    assert q2[-1].values[ocf] == 40.0
    assert q2[-1].values[assets] == 1000.0
    assert np.isnan(by_q[dt.date(2024, 3, 31)][0].values[assets])


def test_end_to_end_graph(tmp_path: Path, edgar: EdgarProvider) -> None:
    store = ParquetStore(tmp_path)
    for ds in ("entities", "filings", "facts"):
        edgar.ingest(store, ds, [str(CIK)])
    g = TemporalGraph()
    define_core_schema(g)
    load_entities(store, g)
    load_filings(store, g)
    load_fundamentals(store, g)
    again = load_fundamentals(store, g)
    assert again["new"] == 0  # idempotent

    def q4_revenue(at: str) -> float | None:
        s = g.snapshot(at, history={"company.fundamentals": 1})
        ch = s["nodes"]["company"]["channels"]["fundamentals"]
        if not ch["present"][0, 0]:
            return None
        return float(ch["values"][0, 0, FEATURE_NAMES.index("revenue")])

    assert q4_revenue("2025-02-14T12:30:59Z") == 120.0  # still Q3 (latest known quarter)
    assert q4_revenue("2025-02-14T12:31:00Z") == 130.0
    assert q4_revenue("2025-03-20T13:00:59Z") == 130.0
    assert q4_revenue("2025-03-20T13:01:00Z") == 120.0

    s = g.snapshot("2025-03-01T00:00:00Z")
    assert s["nodes"]["filing"]["keys"] == ["0001234567-25-000005"]  # 30-day window
    assert s["edges"]["filed"]["edge_index"].shape == (2, 1)


def test_http_client_guards(tmp_path: Path) -> None:
    import time

    from automoonbot.data.http import HttpClient, RateLimiter

    with pytest.raises(ValueError, match="User-Agent"):
        HttpClient("x", allowed_hosts=frozenset({"data.sec.gov"}), rate_per_host=1)
    with pytest.raises(ValueError, match="User-Agent"):
        HttpClient("ok agent\r\nX-Injected: 1", allowed_hosts=frozenset({"a"}), rate_per_host=1)
    c = HttpClient(
        "research client a@b.c", allowed_hosts=frozenset({"data.sec.gov"}), rate_per_host=5
    )
    for url in ["http://data.sec.gov/x", "https://evil.example/x", "file:///etc/passwd"]:
        with pytest.raises(ValueError, match="refusing"):
            c.get(url)
    rl = RateLimiter(rate=50.0, burst=1)
    start = time.monotonic()
    for _ in range(6):
        rl.acquire()
    assert time.monotonic() - start >= 5 / 50 * 0.9

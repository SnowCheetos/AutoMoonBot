"""Load canonical datasets from a :class:`ParquetStore` into a
:class:`~automoonbot.TemporalGraph`.

Loaders are idempotent (the graph ignores exact replays), so a pipeline can
simply re-run them after every data update.
"""

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from automoonbot.data.fundamentals import FEATURE_NAMES, quarterly_observations, to_graph_arrays
from automoonbot.data.store import ParquetStore
from automoonbot.graph import OPEN_END, OPEN_START, Channel, TemporalGraph

DAY_NS = 86_400 * 10**9

#: Filing form families encoded one-hot on filing nodes.
FORM_FAMILIES: tuple[tuple[str, ...], ...] = (
    ("10-K", "10-K/A", "10-KT", "20-F", "40-F"),
    ("10-Q", "10-Q/A"),
    ("8-K", "8-K/A", "6-K"),
    ("S-1", "S-1/A", "424B4", "S-3", "S-4"),
    ("4", "3", "5", "SC 13D", "SC 13G", "SC 13D/A", "SC 13G/A"),
)


def define_core_schema(g: TemporalGraph, filing_window_days: int = 30) -> None:
    """Declare the node kinds and relations the data loaders populate.
    Idempotent."""
    g.define_entity("company", [Channel("fundamentals", len(FEATURE_NAMES), nullable=True)])
    g.define_event(
        "filing",
        window=filing_window_days * DAY_NS,
        channels=[Channel("form", len(FORM_FAMILIES) + 1, bounds=(0.0, 1.0))],
    )
    g.define_relation("filed", "company", "filing")


def _ns(col: pa.ChunkedArray, fill: int) -> np.ndarray:
    return np.asarray(pc.fill_null(col.cast(pa.int64()), fill).to_numpy(), dtype=np.int64)


def load_entities(
    store: ParquetStore, g: TemporalGraph, as_of: int | None = None
) -> dict[str, int]:
    t = store.read("entities", as_of, filter=pc.field("kind") == "company")
    if t.num_rows == 0:
        return {"new": 0, "duplicate": 0}
    return g.upsert_nodes(
        "company",
        t.column("entity_id").to_pylist(),
        known_time=_ns(t.column("known_time"), 0),
        valid_from=_ns(t.column("valid_from"), OPEN_START),
        valid_to=_ns(t.column("valid_to"), OPEN_END),
    )


def form_one_hot(forms: list[str]) -> np.ndarray:
    out = np.zeros((len(forms), len(FORM_FAMILIES) + 1), dtype=np.float32)
    index = {f: i for i, fam in enumerate(FORM_FAMILIES) for f in fam}
    for row, f in enumerate(forms):
        out[row, index.get(f, len(FORM_FAMILIES))] = 1.0
    return out


def load_filings(store: ParquetStore, g: TemporalGraph, as_of: int | None = None) -> dict[str, int]:
    """Filings become event nodes starting at acceptance and are linked to
    their issuer. Issuers must already be loaded."""
    t = store.read("filings", as_of, latest=True)
    if t.num_rows == 0:
        return {"new": 0, "duplicate": 0}
    acc = t.column("accession").to_pylist()
    accepted = _ns(t.column("accepted_at"), 0)
    known = _ns(t.column("known_time"), 0)
    s1 = g.upsert_nodes("filing", acc, known_time=known, valid_from=accepted)
    s2 = g.observe(
        "filing", "form", acc, accepted, known, form_one_hot(t.column("form").to_pylist())
    )
    s3 = g.assert_edges(
        "filed", t.column("entity_id").to_pylist(), acc, known_time=known, valid_from=accepted
    )
    return {k: s1[k] + s2[k] + s3[k] for k in ("new", "duplicate")}


def load_fundamentals(
    store: ParquetStore, g: TemporalGraph, as_of: int | None = None
) -> dict[str, int]:
    """Quarterly fundamentals with full restatement history."""
    facts = store.read("facts", as_of, filter=pc.field("taxonomy") == "us-gaap")
    keys, valid, known, values = to_graph_arrays(quarterly_observations(facts))
    if not keys:
        return {"new": 0, "duplicate": 0}
    return g.observe("company", "fundamentals", keys, valid, known, values)

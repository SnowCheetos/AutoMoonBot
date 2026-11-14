"""Canonical, point-in-time record schemas.

Every provider normalizes into these Arrow schemas. Two columns are
mandatory everywhere:

* ``known_time`` - the earliest UTC instant at which the record could have
  been known by a market participant using this source (e.g. an SEC filing's
  acceptance time, a vendor's publication time plus latency). Providers must
  never fill it with the ingestion wall-clock unless that *is* the first time
  the information existed, and must document their policy in
  :attr:`~automoonbot.data.providers.base.Provider.knowledge_policy`.
* ``source`` - the provider that produced the row, so mixed-quality sources
  can be audited and filtered.

Prices are stored **unadjusted**. Split/dividend-adjusted series embed future
corporate actions into past prices (look-ahead), so adjustment is applied
point-in-time from ``corporate_actions`` instead.
"""

from dataclasses import dataclass

import pyarrow as pa

TS = pa.timestamp("ns", tz="UTC")


@dataclass(frozen=True)
class Dataset:
    name: str
    schema: pa.Schema
    #: Columns identifying one fact; together with ``known_time`` they
    #: identify one *version* of it.
    key: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        names = set(self.schema.names)
        missing = {"known_time", "source", *self.key} - names
        if missing:
            raise ValueError(f"{self.name}: schema lacks {sorted(missing)}")


def _f(name: str, typ: pa.DataType, nullable: bool = False) -> pa.Field:
    return pa.field(name, typ, nullable=nullable)


ENTITIES = Dataset(
    "entities",
    pa.schema(
        [
            _f("entity_id", pa.string()),
            _f("kind", pa.string()),
            _f("name", pa.string(), True),
            _f("valid_from", TS, True),
            _f("valid_to", TS, True),
            _f("known_time", TS),
            _f("source", pa.string()),
        ]
    ),
    ("entity_id",),
    "Issuers and other long-lived entities, keyed by a stable id (e.g. 'cik:0000320193').",
)

IDENTIFIERS = Dataset(
    "identifiers",
    pa.schema(
        [
            _f("entity_id", pa.string()),
            _f("scheme", pa.string()),
            _f("value", pa.string()),
            _f("valid_from", TS, True),
            _f("valid_to", TS, True),
            _f("known_time", TS),
            _f("source", pa.string()),
        ]
    ),
    ("entity_id", "scheme", "value"),
    "Time-varying identifiers (ticker, CUSIP, FIGI). Tickers are reused, so never key on them.",
)

FILINGS = Dataset(
    "filings",
    pa.schema(
        [
            _f("accession", pa.string()),
            _f("entity_id", pa.string()),
            _f("form", pa.string()),
            _f("filing_date", pa.date32()),
            _f("report_period", pa.date32(), True),
            _f("accepted_at", TS),
            _f("primary_document", pa.string(), True),
            _f("known_time", TS),
            _f("source", pa.string()),
        ]
    ),
    ("entity_id", "accession"),
    "Regulatory filings, one row per filer: a co-registrant filing (e.g. a "
    "parent and its subsidiary) shares one accession across entities. "
    "known_time derives from the acceptance timestamp.",
)

FACTS = Dataset(
    "facts",
    pa.schema(
        [
            _f("entity_id", pa.string()),
            _f("accession", pa.string()),
            _f("taxonomy", pa.string()),
            _f("concept", pa.string()),
            _f("unit", pa.string()),
            _f("period_start", pa.date32(), True),
            _f("period_end", pa.date32()),
            _f("value", pa.float64()),
            _f("fiscal_year", pa.int32(), True),
            _f("fiscal_period", pa.string(), True),
            _f("form", pa.string(), True),
            _f("known_time", TS),
            _f("source", pa.string()),
        ]
    ),
    ("entity_id", "accession", "taxonomy", "concept", "unit", "period_start", "period_end"),
    "XBRL facts as reported in each filing (re-reported comparatives are separate versions).",
)

BARS = Dataset(
    "bars",
    pa.schema(
        [
            _f("security_id", pa.string()),
            _f("start", TS),
            _f("end", TS),
            _f("open", pa.float64()),
            _f("high", pa.float64()),
            _f("low", pa.float64()),
            _f("close", pa.float64()),
            _f("volume", pa.float64()),
            _f("vwap", pa.float64(), True),
            _f("known_time", TS),
            _f("source", pa.string()),
        ]
    ),
    ("security_id", "start", "end"),
    "Unadjusted OHLCV bars over [start, end).",
)

CORPORATE_ACTIONS = Dataset(
    "corporate_actions",
    pa.schema(
        [
            _f("security_id", pa.string()),
            _f("action", pa.string()),  # 'split' | 'cash_dividend'
            _f("ex_date", pa.date32()),
            _f("value", pa.float64()),  # split ratio (new/old) or cash amount per share
            _f("known_time", TS),
            _f("source", pa.string()),
        ]
    ),
    ("security_id", "action", "ex_date"),
    "Splits and dividends, known from their announcement, effective at ex_date.",
)

ARTICLES = Dataset(
    "articles",
    pa.schema(
        [
            _f("article_id", pa.string()),
            _f("publisher", pa.string()),
            _f("published_at", TS),
            _f("title", pa.string()),
            _f("url", pa.string(), True),
            _f("body_sha256", pa.string(), True),
            _f("entities", pa.list_(pa.string())),
            _f("known_time", TS),
            _f("source", pa.string()),
        ]
    ),
    ("article_id",),
    "News items; bodies are stored out of band and referenced by hash.",
)

RELATIONS = Dataset(
    "relations",
    pa.schema(
        [
            _f("relation", pa.string()),
            _f("src_id", pa.string()),
            _f("dst_id", pa.string()),
            _f("valid_from", TS, True),
            _f("valid_to", TS, True),
            _f("attrs", pa.list_(pa.float32())),
            _f("asserted", pa.bool_()),  # False = retraction
            _f("provenance", pa.string()),
            _f("known_time", TS),
            _f("source", pa.string()),
        ]
    ),
    ("relation", "src_id", "dst_id", "valid_from", "valid_to"),
    "Typed relationships between entities with their evidence (accession, URL, model run).",
)

DATASETS: dict[str, Dataset] = {
    d.name: d
    for d in (ENTITIES, IDENTIFIERS, FILINGS, FACTS, BARS, CORPORATE_ACTIONS, ARTICLES, RELATIONS)
}


def conform(dataset: Dataset, table: pa.Table) -> pa.Table:
    """Validate ``table`` against ``dataset`` and return it in canonical
    column order and types.

    Columns must all be present; extra columns are rejected (they usually
    signal a provider bug). Safe casts are applied; unsafe ones raise. Nulls
    in non-nullable columns raise. ``known_time`` must be timezone-aware UTC.
    """
    extra = set(table.column_names) - set(dataset.schema.names)
    missing = set(dataset.schema.names) - set(table.column_names)
    if extra or missing:
        raise ValueError(
            f"{dataset.name}: columns do not match schema (missing={sorted(missing)}, "
            f"extra={sorted(extra)})"
        )
    for f in dataset.schema:
        typ = table.schema.field(f.name).type
        if pa.types.is_timestamp(f.type) and pa.types.is_timestamp(typ) and typ.tz is None:
            raise ValueError(f"{dataset.name}.{f.name}: timestamps must be timezone-aware")
    cols = []
    for f in dataset.schema:
        col = table.column(f.name)
        if col.type != f.type:
            col = col.cast(f.type, safe=True)
        if not f.nullable and col.null_count:
            raise ValueError(
                f"{dataset.name}.{f.name}: {col.null_count} nulls in a required column"
            )
        cols.append(col)
    return pa.Table.from_arrays(cols, schema=dataset.schema)

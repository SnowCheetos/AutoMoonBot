"""Point-in-time quarterly fundamentals from as-reported XBRL facts.

Filers report flows inconsistently: income statements per quarter, cash-flow
statements year-to-date, and annual reports only the full year. This module
derives *discrete fiscal-quarter* values for additive flows (directly, or as
the difference of two cumulative facts sharing a start date, e.g.
``FY - 9M`` for Q4 or ``6M - 3M`` for a Q2 cash flow) and quarter-end values
for stocks (balance-sheet instants).

Crucially it is bitemporal: every fact version has the ``known_time`` of the
filing that reported it, a later filing may restate it, and a derived value is
only known once *all* its components are. :func:`quarterly_observations`
therefore emits, per company and quarter, one observation each time the
known vector changes, with that change's ``known_time``. Feeding these into
the graph reproduces exactly what was knowable at any instant, including
restatements.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

QUARTER_DAYS = (80, 100)


@dataclass(frozen=True)
class Feature:
    name: str
    #: us-gaap concepts in priority order; the first one reported wins.
    concepts: tuple[str, ...]
    unit: str
    #: "flow" (duration) or "stock" (instant).
    kind: str
    #: Whether quarterly values may be derived by subtraction.
    additive: bool = True


FEATURES: tuple[Feature, ...] = (
    Feature(
        "revenue",
        (
            "Revenues",
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "SalesRevenueNet",
        ),
        "USD",
        "flow",
    ),
    Feature(
        "cost_of_revenue",
        ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"),
        "USD",
        "flow",
    ),
    Feature("gross_profit", ("GrossProfit",), "USD", "flow"),
    Feature("research_and_development", ("ResearchAndDevelopmentExpense",), "USD", "flow"),
    Feature("sga", ("SellingGeneralAndAdministrativeExpense",), "USD", "flow"),
    Feature("operating_income", ("OperatingIncomeLoss",), "USD", "flow"),
    Feature("net_income", ("NetIncomeLoss", "ProfitLoss"), "USD", "flow"),
    Feature("operating_cash_flow", ("NetCashProvidedByUsedInOperatingActivities",), "USD", "flow"),
    Feature("capex", ("PaymentsToAcquirePropertyPlantAndEquipment",), "USD", "flow"),
    Feature("eps_diluted", ("EarningsPerShareDiluted",), "USD/shares", "flow", additive=False),
    Feature(
        "diluted_shares",
        ("WeightedAverageNumberOfDilutedSharesOutstanding",),
        "shares",
        "flow",
        additive=False,
    ),
    Feature("total_assets", ("Assets",), "USD", "stock"),
    Feature("total_liabilities", ("Liabilities",), "USD", "stock"),
    Feature(
        "stockholders_equity",
        (
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ),
        "USD",
        "stock",
    ),
    Feature(
        "cash",
        (
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ),
        "USD",
        "stock",
    ),
    Feature("long_term_debt", ("LongTermDebtNoncurrent", "LongTermDebt"), "USD", "stock"),
    Feature(
        "shares_outstanding", ("CommonStockSharesOutstanding",), "shares", "stock", additive=False
    ),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURES)

# (concept, unit, start, end) -> list of (known_time, value), one per version
_FactKey = tuple[str, str, dt.date | None, dt.date]


@dataclass(frozen=True)
class QuarterlyObservation:
    entity_id: str
    period_start: dt.date
    period_end: dt.date
    known_time: int
    #: ``len(FEATURES)`` values, NaN where unknown.
    values: np.ndarray


def _is_quarter(start: dt.date, end: dt.date) -> bool:
    return QUARTER_DAYS[0] <= (end - start).days <= QUARTER_DAYS[1]


class _Versions:
    """All versions of one fact; ``at(t)`` is the latest known by ``t``."""

    __slots__ = ("items",)

    def __init__(self) -> None:
        self.items: list[tuple[int, float]] = []

    def add(self, known: int, value: float) -> None:
        self.items.append((known, value))

    def finish(self) -> None:
        # Same filing reporting the same fact twice with different values is a
        # filer error; take the deterministic max to stay order-independent.
        best: dict[int, float] = {}
        for k, v in self.items:
            best[k] = max(v, best.get(k, -np.inf))
        self.items = sorted(best.items())

    def at(self, t: int) -> float | None:
        out = None
        for k, v in self.items:
            if k > t:
                break
            out = v
        return out

    def first_known(self) -> int:
        return self.items[0][0]


def _tile(
    spans: dict[tuple[dt.date, dt.date], _Versions], start: dt.date, end: dt.date
) -> list[_Versions] | None:
    """Facts exactly covering ``[start, end]``: one direct span, or a chain of
    consecutive quarterly spans."""
    if (v := spans.get((start, end))) is not None:
        return [v]
    out: list[_Versions] = []
    cursor = start
    while cursor <= end:
        nxt = [
            (e, v) for (s, e), v in spans.items() if s == cursor and e <= end and _is_quarter(s, e)
        ]
        if not nxt:
            return None
        e, v = min(nxt, key=lambda x: x[0])
        out.append(v)
        if e == end:
            return out
        cursor = e + dt.timedelta(days=1)
    return None


def quarterly_observations(
    facts: pa.Table, features: Sequence[Feature] = FEATURES
) -> list[QuarterlyObservation]:
    """Derive bitemporal quarterly feature vectors from a ``facts`` table
    (any number of entities). Output is sorted by entity, period end, and
    known time."""
    wanted = {(c, f.unit) for f in features for c in f.concepts}
    by_entity: dict[str, dict[_FactKey, _Versions]] = defaultdict(lambda: defaultdict(_Versions))
    cols = facts.select(
        [
            "entity_id",
            "taxonomy",
            "concept",
            "unit",
            "period_start",
            "period_end",
            "value",
            "known_time",
        ]
    ).to_pydict()
    known = facts.column("known_time").cast(pa.int64()).to_pylist()
    for i in range(facts.num_rows):
        if cols["taxonomy"][i] != "us-gaap":
            continue
        key = (cols["concept"][i], cols["unit"][i])
        if key not in wanted:
            continue
        fact_key: _FactKey = (key[0], key[1], cols["period_start"][i], cols["period_end"][i])
        by_entity[cols["entity_id"][i]][fact_key].add(known[i], cols["value"][i])
    out: list[QuarterlyObservation] = []
    for eid in sorted(by_entity):
        out.extend(_entity_observations(eid, by_entity[eid], features))
    return out


def _entity_observations(
    eid: str, store: dict[_FactKey, _Versions], features: Sequence[Feature]
) -> Iterable[QuarterlyObservation]:
    for v in store.values():
        v.finish()
    # Index durations by (concept, unit) for subtraction lookups.
    durations: dict[tuple[str, str], dict[tuple[dt.date, dt.date], _Versions]] = defaultdict(dict)
    instants: dict[tuple[str, str], dict[dt.date, _Versions]] = defaultdict(dict)
    for (concept, unit, start, end), v in store.items():
        if start is None:
            instants[(concept, unit)][end] = v
        else:
            durations[(concept, unit)][(start, end)] = v

    # Candidate quarters: direct quarterly durations, and the last quarter of
    # any longer cumulative span (e.g. Q4 inside a fiscal year), delimited by
    # the end of some other reported span of the same concept.
    quarters: set[tuple[dt.date, dt.date]] = set()
    for spans in durations.values():
        ends = {e for _, e in spans}
        for s, e in spans:
            if _is_quarter(s, e):
                quarters.add((s, e))
            elif (e - s).days > QUARTER_DAYS[1]:
                for prev in ends:
                    if s < prev < e and _is_quarter(prev + dt.timedelta(days=1), e):
                        quarters.add((prev + dt.timedelta(days=1), e))

    for qs, qe in sorted(quarters, key=lambda q: (q[1], q[0])):
        # For each feature, a list of "recipes": each recipe is a list of
        # (versions, sign); its value is known once all components are.
        recipes: list[list[list[tuple[_Versions, float]]]] = []
        for f in features:
            options: list[list[tuple[_Versions, float]]] = []
            for c in f.concepts:
                key = (c, f.unit)
                if f.kind == "stock":
                    if (inst := instants.get(key, {}).get(qe)) is not None:
                        options.append([(inst, 1.0)])
                    continue
                spans = durations.get(key, {})
                if (direct := spans.get((qs, qe))) is not None:
                    options.append([(direct, 1.0)])
                if f.additive:
                    # cumulative(s0, qe) - cumulative(s0, qs - 1), where the
                    # subtrahend is reported directly or tiled by quarters.
                    prev_end = qs - dt.timedelta(days=1)
                    for (s0, e0), total in sorted(spans.items(), key=lambda kv: kv[0]):
                        if e0 != qe or s0 >= qs:
                            continue
                        tile = _tile(spans, s0, prev_end)
                        if tile is not None:
                            options.append([(total, 1.0), *((p, -1.0) for p in tile)])
            recipes.append(options)
        times = sorted(
            {k for opts in recipes for recipe in opts for v, _ in recipe for k, _ in v.items}
        )
        last: np.ndarray | None = None
        for t in times:
            vec = np.full(len(features), np.nan, dtype=np.float64)
            for j, opts in enumerate(recipes):
                for recipe in opts:  # priority order: direct before derived
                    parts = [v.at(t) for v, _ in recipe]
                    known_parts = [p for p in parts if p is not None]
                    if len(known_parts) == len(parts):
                        vec[j] = sum(
                            sign * p for (_, sign), p in zip(recipe, known_parts, strict=True)
                        )
                        break
            if np.isnan(vec).all():
                continue
            if last is not None and np.array_equal(vec, last, equal_nan=True):
                continue
            last = vec
            yield QuarterlyObservation(eid, qs, qe, t, vec)


def to_graph_arrays(
    observations: Sequence[QuarterlyObservation],
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """``(keys, valid_time_ns, known_time_ns, values float32[n, F])`` for
    :meth:`automoonbot.TemporalGraph.observe`. Valid time is the period end
    at 00:00 UTC. Values beyond float32 range raise rather than overflow."""
    n = len(observations)
    keys = [o.entity_id for o in observations]
    epoch = dt.date(1970, 1, 1)
    valid = np.fromiter(
        ((o.period_end - epoch).days * 86_400 * 10**9 for o in observations),
        dtype=np.int64,
        count=n,
    )
    known = np.fromiter((o.known_time for o in observations), dtype=np.int64, count=n)
    values = np.stack([o.values for o in observations]) if n else np.zeros((0, len(FEATURES)))
    finite = values[np.isfinite(values)]
    if finite.size and np.abs(finite).max() > np.finfo(np.float32).max:
        raise OverflowError("fundamental value exceeds float32 range")
    return keys, valid, known, values.astype(np.float32)

"""Universe graph: companies linked to their industry and sector.

The classification is stored in the point-in-time graph store like any other
relationship, and the model reads the structure through
:meth:`TemporalGraph.snapshot` (homogeneous view) at each decision time, so
membership changes (an IPO, a reclassification, a delisting) reach the model
only once known.

Caveat recorded with the data: a classification source that only publishes
the *current* mapping (e.g. today's GICS codes) is asserted as known from
each company's first bar, which back-dates reclassifications.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from automoonbot.graph import TemporalGraph

KINDS = ("company", "industry", "sector")
RELATIONS = ("in_industry", "in_sector")


@dataclass(frozen=True)
class Membership:
    security_id: str
    industry: str
    sector: str


def _key(prefix: str, name: str) -> str:
    """Store keys for classification nodes (store keys are opaque bytes)."""
    return f"{prefix}:{name.strip()}"


def build_universe_graph(
    members: Sequence[Membership], first_known: npt.ArrayLike
) -> TemporalGraph:
    """A graph with ``company``/``industry``/``sector`` nodes and
    ``in_industry`` (company -> industry) and ``in_sector`` (industry ->
    sector) edges. ``first_known[i]`` (ns UTC) is when member ``i`` and its
    classification become known."""
    known = np.asarray(first_known, dtype=np.int64)
    if known.shape != (len(members),):
        raise ValueError("first_known must have one time per member")
    g = TemporalGraph()
    for k in KINDS:
        g.define_entity(k)
    g.define_relation("in_industry", "company", "industry")
    g.define_relation("in_sector", "industry", "sector")
    companies = [m.security_id for m in members]
    if len(set(companies)) != len(companies):
        raise ValueError("duplicate security ids")
    inds = [_key("industry", m.industry) for m in members]
    secs = [_key("sector", m.sector) for m in members]
    g.upsert_nodes("company", companies, known, valid_from=known)
    # A class node is known from its first member's time.
    first: dict[tuple[str, str], int] = {}
    pairs: dict[str, str] = {}
    for ind, sec, t in zip(inds, secs, known.tolist(), strict=True):
        if pairs.setdefault(ind, sec) != sec:
            raise ValueError(f"{ind} is mapped to two sectors")
        first[("industry", ind)] = min(first.get(("industry", ind), t), t)
        first[("sector", sec)] = min(first.get(("sector", sec), t), t)
    for kind in ("industry", "sector"):
        keys = sorted(k for kk, k in first if kk == kind)
        g.upsert_nodes(kind, keys, [first[(kind, k)] for k in keys])
    g.assert_edges("in_industry", companies, inds, known, valid_from=known)
    ind_keys = sorted(pairs)
    ind_known = [first[("industry", k)] for k in ind_keys]
    g.assert_edges(
        "in_sector", ind_keys, [pairs[k] for k in ind_keys], ind_known, valid_from=ind_known
    )
    return g


@dataclass(frozen=True)
class Structure:
    """Token layout of one snapshot: companies first (in snapshot key
    order), then industries, then sectors."""

    company_keys: list[str]
    #: Token kind per token: 0 company, 1 industry, 2 sector.
    token_kind: npt.NDArray[np.int64]
    #: Index of each token within its kind (class-node identity).
    token_index: npt.NDArray[np.int64]
    class_keys: dict[str, list[str]]
    #: ``[E]`` directed edges between tokens and their relation id
    #: (reverse edges have ids ``r + len(RELATIONS)``).
    src: npt.NDArray[np.int64]
    dst: npt.NDArray[np.int64]
    edge_type: npt.NDArray[np.int64]

    @property
    def n_tokens(self) -> int:
        return int(self.token_kind.size)

    @property
    def n_edge_types(self) -> int:
        return 2 * len(RELATIONS)


def structure_at(graph: TemporalGraph, known_at: int) -> Structure:
    """The universe's token graph as known at ``known_at``."""
    snap = graph.snapshot(known_at, homogeneous="reverse")
    h = snap["homogeneous"]
    offsets = np.asarray(h["kind_offsets"], dtype=np.int64)
    node_type = np.asarray(h["node_type"], dtype=np.int64)
    kinds = list(snap["nodes"])
    if kinds != list(KINDS):
        raise ValueError(f"unexpected node kinds {kinds}")
    idx = np.arange(node_type.size) - offsets[node_type]
    rels = list(snap["edges"])
    if rels != list(RELATIONS):
        raise ValueError(f"unexpected relations {rels}")
    ei = np.asarray(h["edge_index"], dtype=np.int64)
    return Structure(
        company_keys=list(snap["nodes"]["company"]["keys"]),
        token_kind=node_type,
        token_index=idx.astype(np.int64),
        class_keys={k: list(snap["nodes"][k]["keys"]) for k in ("industry", "sector")},
        src=ei[0],
        dst=ei[1],
        edge_type=np.asarray(h["edge_type"], dtype=np.int64),
    )

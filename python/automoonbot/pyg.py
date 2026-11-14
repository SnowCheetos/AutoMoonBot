"""Conversion of snapshots to PyTorch / PyG inputs.

Two layouts are provided:

* :func:`to_hetero_data` - a ``torch_geometric.data.HeteroData`` with one node
  store per kind and one edge store per relation (for heterogeneous GNNs).
* :func:`to_token_graph` - a flat token graph (one token per node, type ids,
  CSR adjacency, temporal fields) for graph transformers.

Temporal information is exposed explicitly instead of being imputed: every
feature slot comes with its ``present`` mask and its age relative to the
snapshot's valid time, so a model can learn how much to trust stale data.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from automoonbot._moonrs import MISSING_TIME

if TYPE_CHECKING:
    import torch
    from torch_geometric.data import HeteroData

NS_PER_DAY = 86_400 * 1_000_000_000


def age_days(times: npt.NDArray[np.int64], valid_at: int) -> npt.NDArray[np.float32]:
    """Age of each timestamp relative to ``valid_at`` in days; ``NaN`` for
    missing slots or unbounded (``OPEN_START``) times."""
    times = np.asarray(times, dtype=np.int64)
    out = np.full(times.shape, np.nan, dtype=np.float32)
    ok = times != MISSING_TIME
    out[ok] = ((valid_at - times[ok]) / NS_PER_DAY).astype(np.float32)
    return out


def log_age(times: npt.NDArray[np.int64], valid_at: int) -> npt.NDArray[np.float32]:
    """``log1p(age in days)``, 0 for missing slots (pair with the mask)."""
    a = age_days(times, valid_at)
    return np.where(np.isnan(a), 0.0, np.log1p(np.maximum(a, 0.0))).astype(np.float32)


def node_features(snapshot: dict[str, Any], kind: str) -> npt.NDArray[np.float32]:
    """Concatenate every channel of ``kind`` into one ``[N, F]`` matrix.

    For each channel the latest history slot contributes its values, its
    presence bit, element presence (nullable channels) and ``log_age``. Older
    history slots are left to sequence-aware models via
    :func:`channel_history`.
    """
    block = snapshot["nodes"][kind]
    n = len(block["keys"])
    parts: list[npt.NDArray[np.float32]] = []
    for ch in block["channels"].values():
        parts.append(ch["values"][:, -1, :])
        parts.append(ch["present"][:, -1:].astype(np.float32))
        if ch["element_present"] is not None:
            parts.append(ch["element_present"][:, -1, :].astype(np.float32))
        parts.append(log_age(ch["valid_time"][:, -1:], snapshot["valid_at"]))
    if not parts:
        return np.zeros((n, 0), dtype=np.float32)
    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)


def edge_features(snapshot: dict[str, Any], relation: str) -> npt.NDArray[np.float32]:
    """``[E, A']``: attributes, attribute presence (if nullable) and the
    edge's ``log_age`` (zero for edges with an unbounded start)."""
    e = snapshot["edges"][relation]
    since = e["since"]
    age = log_age(
        np.where(since == np.iinfo(np.int64).min, MISSING_TIME, since)[:, None],
        snapshot["valid_at"],
    )
    parts = [e["attrs"]]
    if e["attr_present"] is not None:
        parts.append(e["attr_present"].astype(np.float32))
    parts.append(age)
    return np.ascontiguousarray(np.concatenate(parts, axis=1), dtype=np.float32)


def to_hetero_data(snapshot: dict[str, Any]) -> HeteroData:
    """Build a ``HeteroData`` (requires ``torch`` and ``torch_geometric``).

    Tensors share memory with the snapshot's NumPy arrays where dtypes allow.
    """
    import torch
    from torch_geometric.data import HeteroData

    data = HeteroData()
    for kind, block in snapshot["nodes"].items():
        store = data[kind]
        store.x = torch.from_numpy(node_features(snapshot, kind))
        store.num_nodes = len(block["keys"])
        store.keys = block["keys"]
        for name, ch in block["channels"].items():
            store[f"{name}_history"] = torch.from_numpy(ch["values"])
            store[f"{name}_mask"] = torch.from_numpy(ch["present"].astype(bool))
    for rel, e in snapshot["edges"].items():
        store = data[e["src_kind"], rel, e["dst_kind"]]
        store.edge_index = torch.from_numpy(e["edge_index"])
        store.edge_attr = torch.from_numpy(edge_features(snapshot, rel))
    data.known_at = snapshot["known_at"]
    data.valid_at = snapshot["valid_at"]
    return data


def to_token_graph(snapshot: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Flat token graph for graph transformers.

    Requires a snapshot taken with ``homogeneous="forward"`` or
    ``"reverse"``. Node features of different kinds have different widths, so
    they are returned per kind (``x::<kind>``) together with ``kind_offsets``;
    a model projects each kind to the shared width and concatenates in offset
    order. Edge attributes are likewise per relation, gathered by
    ``edge_type``/``edge_row``.
    """
    import torch

    h = snapshot.get("homogeneous")
    if h is None:
        raise ValueError("snapshot was taken without homogeneous=...; pass 'forward' or 'reverse'")
    out: dict[str, torch.Tensor] = {
        "kind_offsets": torch.from_numpy(h["kind_offsets"]),
        "node_type": torch.from_numpy(h["node_type"]),
        "edge_index": torch.from_numpy(h["edge_index"]),
        "edge_type": torch.from_numpy(h["edge_type"]),
        "edge_row": torch.from_numpy(h["edge_row"]),
        "indptr": torch.from_numpy(h["indptr"]),
    }
    for kind in snapshot["nodes"]:
        out[f"x::{kind}"] = torch.from_numpy(node_features(snapshot, kind))
    for rel in snapshot["edges"]:
        out[f"edge_attr::{rel}"] = torch.from_numpy(edge_features(snapshot, rel))
    return out

"""Tensors and per-day token graphs for the graph transformer.

:class:`PanelData` holds everything the model reads from a
:class:`MarketPanel` as dense tensors, plus the training labels. Inputs at
row ``t`` depend on bars ``<= t`` only; labels are forward-looking and only
ever used as training targets.

:class:`TokenLayout` fixes one token layout for a universe: the panel's
securities first (in panel order), then every class node (industry,
sector) the graph ever holds. Each day's *structure*, i.e. which class
nodes exist and which edges connect tokens, comes from the point-in-time
graph snapshot at that day's decision time, so a class that is not yet
known is masked and an edge not yet known carries no bias. A fixed layout
keeps every day's computation independent of the others (and of how far
the panel extends), which is what makes inference bit-for-bit causal.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from automoonbot.backtest.labels import forward_returns
from automoonbot.backtest.panel import MarketPanel
from automoonbot.graph import TemporalGraph
from automoonbot.models.features import price_features
from automoonbot.models.universe import KINDS, RELATIONS, structure_at

I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class PanelData:
    """Dense tensors for one panel (``T`` bars, ``S`` securities)."""

    #: ``[S, T, C]`` features (security-major for window gathers), 0 where
    #: undefined.
    feats: Tensor
    #: ``[T, S]`` the security has a full feature vector at ``t``.
    ok: Tensor
    #: ``[T, S]`` open-to-close log return of bar ``t`` (0 where no bar):
    #: drifts the book from the fill at the open to the decision.
    intraday: Tensor
    #: ``[T, S]`` close-to-close total log return (0 where undefined).
    close_ret: Tensor
    #: ``[T, S]`` total log return from open ``t + 1`` to open ``t + 2``,
    #: i.e. what a decision at ``t`` earns (0 where undefined). Label.
    r1: Tensor
    #: ``[T, S, H]`` multi-horizon labels and their mask.
    labels: Tensor
    label_ok: Tensor

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.ok.shape[0]), int(self.ok.shape[1])

    @staticmethod
    def build(panel: MarketPanel, horizons: Sequence[int]) -> PanelData:
        fp = price_features(panel)
        lab = forward_returns(panel, horizons)
        one = forward_returns(panel, [1]).values[..., 0]
        with np.errstate(invalid="ignore", divide="ignore"):
            intraday = np.where(panel.tradable, np.log(panel.close / panel.open), 0.0)
        close_ret = panel.log_returns()

        def f32(a: npt.NDArray[np.floating]) -> Tensor:
            return torch.from_numpy(np.ascontiguousarray(np.nan_to_num(a, nan=0.0), np.float32))

        return PanelData(
            feats=f32(np.transpose(fp.values, (1, 0, 2))),
            ok=torch.from_numpy(fp.ok.copy()),
            intraday=f32(np.where(np.isfinite(intraday), intraday, 0.0)),
            close_ret=f32(close_ret),
            r1=f32(one),
            labels=f32(lab.values),
            label_ok=torch.from_numpy(lab.mask.copy()),
        )

    def windows(
        self, days: Tensor, window: int, cols: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Feature windows ending at each of ``days[B]``, for the companies
        (``cols``, default all) valid that day: ``(x[M, K, C], present[M, K],
        pair[M], valid[B, S'])`` with ``pair`` the flat index into ``valid``."""
        valid = self.ok[days] if cols is None else self.ok[days][:, cols]
        b_idx, s_idx = valid.nonzero(as_tuple=True)
        sec = s_idx if cols is None else cols[s_idx]
        rows = days[b_idx, None] - window + 1 + torch.arange(window)
        inside = rows >= 0
        rows = rows.clamp(min=0)
        x = self.feats[sec[:, None], rows]
        present = self.ok[rows, sec[:, None]] & inside
        x = x * present.unsqueeze(-1)
        pair = b_idx * valid.shape[1] + s_idx
        return x, present, pair, valid


@dataclass(frozen=True)
class DayGraph:
    #: ``[N]`` token is present as a key (classes; companies use ``ok``).
    class_known: npt.NDArray[np.bool_]
    #: ``[3, E]`` (src, dst, type) in layout token indices.
    edges: I64


class TokenLayout:
    """Token layout of a universe graph over a panel's securities."""

    def __init__(self, graph: TemporalGraph, securities: Sequence[str]) -> None:
        self.graph = graph
        self.securities = list(securities)
        self.n_companies = len(self.securities)
        horizon = graph.max_known_time()
        full = structure_at(graph, horizon) if horizon is not None else None
        keys: list[tuple[int, str]] = []
        if full is not None:
            for kind_id, kind in enumerate(KINDS[1:], start=1):
                keys.extend((kind_id, k) for k in full.class_keys[kind])
        self.class_keys = keys
        self.n_tokens = self.n_companies + len(keys)
        self._company = {k: i for i, k in enumerate(self.securities)}
        self._class = {k: self.n_companies + i for i, k in enumerate(keys)}
        kinds = np.zeros(self.n_tokens, np.int64)
        kinds[self.n_companies :] = [k for k, _ in keys]
        self.token_kind = torch.from_numpy(kinds)
        self._days: dict[int, DayGraph] = {}

    @property
    def n_edge_types(self) -> int:
        return 2 * len(RELATIONS)

    def vocabulary(self, known_at: int) -> dict[tuple[int, str], int]:
        """Class-embedding ids (from 1) for the classes known at
        ``known_at``; anything else maps to 0 (unseen)."""
        day = self.day(known_at)
        known = [k for k, ok in zip(self.class_keys, day.class_known, strict=True) if ok]
        return {k: i + 1 for i, k in enumerate(known)}

    def class_ids(self, vocab: dict[tuple[int, str], int]) -> Tensor:
        ids = np.zeros(self.n_tokens, np.int64)
        ids[self.n_companies :] = [vocab.get(k, 0) for k in self.class_keys]
        return torch.from_numpy(ids)

    def day(self, known_at: int) -> DayGraph:
        """The structure known at ``known_at`` (memoized)."""
        hit = self._days.get(known_at)
        if hit is not None:
            return hit
        known = np.zeros(len(self.class_keys), np.bool_)
        edges = np.zeros((3, 0), np.int64)
        try:
            st = structure_at(self.graph, known_at)
        except ValueError:
            st = None  # nothing known yet (no kinds or relations in the snapshot)
        if st is not None:
            n_comp = len(st.company_keys)
            # Snapshot token -> layout token; -1 for companies outside the
            # panel and for classes retracted before the graph's latest
            # knowledge time (not in the layout).
            snap_keys = (
                [self._company.get(k, -1) for k in st.company_keys]
                + [self._class.get((1, k), -1) for k in st.class_keys["industry"]]
                + [self._class.get((2, k), -1) for k in st.class_keys["sector"]]
            )
            m = np.asarray(snap_keys, np.int64)
            cls = m[n_comp:]
            known[cls[cls >= 0] - self.n_companies] = True
            src, dst = m[st.src], m[st.dst]
            keep = (src >= 0) & (dst >= 0)
            edges = np.stack((src[keep], dst[keep], st.edge_type[keep]))
        out = DayGraph(known, edges)
        self._days[known_at] = out
        return out

    def batch(
        self, known_at: Sequence[int], ok: Tensor, cols: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """``(edges[4, E], key_ok[B, N'], token_kind[N'])`` for days with
        decision times ``known_at[B]`` and availability ``ok[B, S']`` of the
        companies ``cols`` (default all, else a sorted subset; class tokens
        are always kept and follow the companies)."""
        n_cls = self.n_tokens - self.n_companies
        if cols is None:
            remap = None
            kind = self.token_kind
        else:
            # Old token -> new token; -1 for dropped companies.
            remap = np.full(self.n_tokens, -1, np.int64)
            keep = cols.numpy()
            remap[keep] = np.arange(keep.size)
            remap[self.n_companies :] = keep.size + np.arange(n_cls)
            kind = torch.cat((self.token_kind[cols], self.token_kind[self.n_companies :]))
        parts, known = [], []
        for b, t in enumerate(known_at):
            g = self.day(int(t))
            e = g.edges
            if remap is not None and e.shape[1]:
                src, dst = remap[e[0]], remap[e[1]]
                m = (src >= 0) & (dst >= 0)
                e = np.stack((src[m], dst[m], e[2][m]))
            if e.shape[1]:
                parts.append(np.vstack((np.full((1, e.shape[1]), b), e)))
            known.append(g.class_known)
        edges = (
            torch.from_numpy(np.concatenate(parts, axis=1))
            if parts
            else torch.zeros(4, 0, dtype=torch.int64)
        )
        cls = (
            torch.from_numpy(np.stack(known)) if known else torch.zeros(0, n_cls, dtype=torch.bool)
        )
        return edges, torch.cat((ok, cls), dim=1), kind

    def class_ids_for(self, ids: Tensor, cols: Tensor | None) -> Tensor:
        """Class ids (from :meth:`class_ids`) for the token subset ``cols``."""
        if cols is None:
            return ids
        return torch.cat((ids[cols], ids[self.n_companies :]))

"""Training and walk-forward deployment of the graph transformer policy.

Objective
---------
The policy is trained directly on what it is evaluated on: the realized,
cost-adjusted log-growth of the portfolio it chooses. For a window of ``L``
consecutive decision days the policy is rolled forward from cash; on each
day it sees its current (drifted) weights, picks new ones, pays a linear
cost on the turnover against the holdings at the fill, and earns the
next open-to-open total return. With daily growth ``g`` the loss is::

    -(mean(g) - lambda * CVaR_alpha(-g)) * periods_per_year + beta * NLL

where the CVaR is taken over every day of the batch and ``NLL`` is the
Student-t likelihood of the multi-horizon return heads (an auxiliary task
that shapes the representation). Prices do not react to the portfolio, so
the whole path is differentiable and no critic or policy-gradient
estimator is needed; the first ``burn_in`` days of each window only warm
the book up and are not scored.

Selection and causality
-----------------------
:meth:`GraphTransformerStrategy.fit` receives a panel truncated before the
test window, builds labels from that panel only (so no label realized after
the cut exists), holds out the last ``val_days`` training steps (purged by
the longest horizon) for early stopping, and warm-starts from the previous
fold's model. :meth:`~GraphTransformerStrategy.targets` encodes each day on
its own with a fixed token layout, so the weights for a day are a
deterministic function of bars up to that day: the causality audit checks
this bit for bit.

At inference a no-trade band leaves small adjustments undone (``NaN``,
hold), and an optional drawdown overlay scales exposure down as the
strategy's own tracked equity falls from its peak.
"""

import copy
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from automoonbot.backtest.panel import MarketPanel
from automoonbot.graph import TemporalGraph
from automoonbot.models.data import PanelData, TokenLayout
from automoonbot.models.features import CHANNELS
from automoonbot.models.transformer import GraphTransformer, ModelConfig, student_t_nll

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class TrainConfig:
    window_days: int = 24
    burn_in: int = 4
    batch_windows: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-2
    #: Epochs for the first fold and for later (warm-started) folds.
    epochs: int = 30
    warm_epochs: int = 10
    steps_per_epoch: int = 100
    patience: int = 5
    #: Training steps held out (most recent) for early stopping.
    val_days: int = 252
    #: Linear cost per unit of turnover assumed in training (bps).
    cost_bps: float = 5.0
    cvar_alpha: float = 0.95
    cvar_weight: float = 0.01
    aux_weight: float = 0.1
    grad_clip: float = 1.0
    periods_per_year: float = 252.0
    #: Skip training windows with fewer valid companies on their first day.
    min_companies: int = 10
    #: Companies per training batch, drawn at random from those with data
    #: (``None``: all). Cuts the cost per step and regularizes; the policy
    #: is invariant to the number of companies, so it transfers.
    universe: int | None = 192
    #: Days encoded per chunk when scoring the validation path.
    eval_chunk: int = 32

    def __post_init__(self) -> None:
        if not 0 <= self.burn_in < self.window_days:
            raise ValueError("burn_in must be in [0, window_days)")
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ValueError("cvar_alpha must be in (0, 1)")
        if min(self.epochs, self.steps_per_epoch, self.batch_windows) < 1:
            raise ValueError("epochs, steps_per_epoch and batch_windows must be >= 1")


@dataclass(frozen=True)
class Overlay:
    """Exposure multiplier from the strategy's own drawdown: 1 below
    ``start``, ``floor`` beyond ``full``, linear in between."""

    start: float = 0.10
    full: float = 0.25
    floor: float = 0.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.start < self.full < 1.0 or not 0.0 <= self.floor <= 1.0:
            raise ValueError("need 0 <= start < full < 1 and floor in [0, 1]")

    def exposure(self, drawdown: float) -> float:
        x = (drawdown - self.start) / (self.full - self.start)
        return 1.0 - (1.0 - self.floor) * min(max(x, 0.0), 1.0)


def drift(w: Tensor, r: Tensor) -> Tensor:
    """Company weights ``w[..., S]`` (rest in cash) after log returns ``r``."""
    v = w * torch.exp(r)
    total = 1.0 - w.sum(-1, keepdim=True) + v.sum(-1, keepdim=True)
    return v / total


def cvar(losses: Tensor, alpha: float) -> Tensor:
    """Mean of the worst ``1 - alpha`` fraction of ``losses`` (at least one)."""
    k = max(1, math.ceil((1.0 - alpha) * losses.numel() - 1e-9))
    return torch.topk(losses.flatten(), k).values.mean()


@dataclass
class _Book:
    weights: F64
    equity: float
    peak: float


@dataclass
class GraphTransformerStrategy:
    """Walk-forward :class:`~automoonbot.backtest.evaluate.Strategy` around
    :class:`~automoonbot.models.transformer.GraphTransformer`."""

    graph: TemporalGraph
    model: dict[str, Any] = field(default_factory=dict)
    train: TrainConfig = field(default_factory=TrainConfig)
    overlay: Overlay | None = None
    #: No-trade band: weight changes smaller than this are not traded.
    band: float = 0.002
    seed: int = 0
    #: Run the encoder under bfloat16 autocast (fp32 weights, policy and
    #: objective). Several times faster on CPUs with AMX/AVX512-BF16.
    bf16: bool = False
    name: str = "graph_transformer"

    def __post_init__(self) -> None:
        self.cfg = ModelConfig(channels=len(CHANNELS), **self.model)
        self.net: GraphTransformer | None = None
        self.vocab: dict[tuple[int, str], int] = {}
        self.history: list[dict[str, float]] = []
        self._layout: TokenLayout | None = None
        self._class_ids: Tensor | None = None
        self._carry: dict[int, _Book] = {}
        self._cache: tuple[npt.NDArray[np.int64], F64, PanelData] | None = None
        self._folds = 0

    @property
    def params(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "band": self.band,
            "bf16": self.bf16,
            "overlay": None if self.overlay is None else asdict(self.overlay),
            **{f"model.{k}": v for k, v in self.model.items()},
            **{f"train.{k}": v for k, v in asdict(self.train).items()},
        }

    # ------------------------------------------------------------- plumbing
    def layout(self, panel: MarketPanel) -> TokenLayout:
        lay = self._layout
        if lay is None or lay.securities != list(panel.securities):
            lay = TokenLayout(self.graph, panel.securities)
            self._layout = lay
        return lay

    def _inputs(self, panel: MarketPanel) -> PanelData:
        """Model inputs (causal rows only), reused across prefixes of one
        panel. Labels from this cache must never be used: they may extend
        past the panel being decided on."""
        t = panel.shape[0]
        if self._cache is not None:
            times, close, data = self._cache
            if (
                t <= times.size
                and np.array_equal(panel.times, times[:t])
                and np.array_equal(panel.close, close[:t], equal_nan=True)
            ):
                return data
        data = PanelData.build(panel, self.cfg.horizons)
        if self._cache is None or t > self._cache[0].size:
            self._cache = (panel.times.copy(), panel.close.copy(), data)
        return data

    def _update_vocab(self, layout: TokenLayout, known_at: int) -> None:
        """Add classes known at ``known_at`` to the embedding vocabulary;
        new rows start from the 'unseen' embedding."""
        assert self.net is not None
        new = [k for k in layout.vocabulary(known_at) if k not in self.vocab]
        if new:
            for k in new:
                self.vocab[k] = len(self.vocab) + 1
            old = self.net.classes
            emb = torch.nn.Embedding(len(self.vocab) + 1, old.embedding_dim)
            with torch.no_grad():
                emb.weight[: old.num_embeddings] = old.weight
                emb.weight[old.num_embeddings :] = old.weight[0]
            self.net.classes = emb
        self._class_ids = layout.class_ids(self.vocab)

    def _encode(
        self,
        data: PanelData,
        layout: TokenLayout,
        times: I64,
        days: Tensor,
        cols: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode ``days`` for the companies ``cols`` (default all)."""
        assert self.net is not None and self._class_ids is not None
        x, present, pair, valid = data.windows(days, self.cfg.window, cols)
        edges, key_ok, kind = layout.batch(times[days.numpy()].tolist(), valid, cols)
        e = self.net.edges(kind.numel(), edges, key_ok)
        class_ids = layout.class_ids_for(self._class_ids, cols)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=self.bf16):
            z, preds = self.net.encode(x, present, pair, valid, kind, class_ids, e)
        return z, preds, valid

    def _rollout(
        self, z: Tensor, preds: Tensor, valid: Tensor, intraday: Tensor, r1: Tensor
    ) -> Tensor:
        """Daily log-growth ``[B, L]`` of the policy rolled forward from
        cash over ``L`` days (inputs ``[B, L, ...]``)."""
        assert self.net is not None
        cost = self.train.cost_bps * 1e-4
        b, n_days, s = valid.shape
        book = z.new_zeros(b, s)
        out = []
        for i in range(n_days):
            w = self.net.policy(z[:, i], preds[:, i], valid[:, i], drift(book, intraday[:, i]))
            wc = w[:, :-1]
            held = drift(book, r1[:, i - 1]) if i else book
            growth = 1.0 + (wc * torch.expm1(r1[:, i])).sum(-1) - cost * (wc - held).abs().sum(-1)
            out.append(torch.log(growth.clamp(min=1e-6)))
            book = wc
        return torch.stack(out, dim=1)

    def _objective(self, g: Tensor) -> Tensor:
        t = self.train
        return (g.mean() - t.cvar_weight * cvar(-g, t.cvar_alpha)) * t.periods_per_year

    # ------------------------------------------------------------- training
    def fit(self, panel: MarketPanel, train: I64) -> None:
        tc = self.train
        train = np.asarray(train, dtype=np.int64)
        if train.size == 0:
            raise ValueError("no training steps")
        torch.manual_seed(self.seed + self._folds)
        rng = np.random.default_rng(self.seed + self._folds)
        self._folds += 1
        # Labels must come from this (truncated) panel: never the cache.
        data = PanelData.build(panel, self.cfg.horizons)
        layout = self.layout(panel)
        times = panel.decision_time
        warm = self.net is not None
        if self.net is None:
            self.net = GraphTransformer(replace(self.cfg, n_edge_types=layout.n_edge_types))
        self._update_vocab(layout, int(times[train[-1]]))
        net = self.net

        max_h = max(self.cfg.horizons)
        val = train[-tc.val_days :] if train.size >= 3 * tc.val_days else train[:0]
        fit_steps = train[train + 1 + max_h < val[0]] if val.size else train
        # Window starts whose whole window lies in the fitting steps.
        inside = np.zeros(panel.shape[0] + 1, np.int64)
        inside[fit_steps + 1] = 1
        inside = np.cumsum(inside)
        cand = fit_steps[fit_steps + tc.window_days <= panel.shape[0]]
        full = inside[cand + tc.window_days] - inside[cand] == tc.window_days
        n_ok = data.ok.sum(dim=1).numpy()
        starts = cand[full]
        starts = starts[n_ok[starts] >= tc.min_companies]
        if starts.size == 0:
            raise ValueError("not enough training data for one window")

        opt = torch.optim.AdamW(net.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        epochs = tc.warm_epochs if warm else tc.epochs
        best: tuple[float, dict[str, Tensor]] | None = None
        stale = 0
        offs = torch.arange(tc.window_days)
        for epoch in range(epochs):
            net.train()
            tot = {"objective": 0.0, "nll": 0.0}
            for _ in range(tc.steps_per_epoch):
                pick = torch.from_numpy(rng.choice(starts, size=tc.batch_windows))
                days = (pick[:, None] + offs).flatten()
                cols = self._sub_universe(data, pick, rng)
                sel = slice(None) if cols is None else cols
                z, preds, valid = self._encode(data, layout, times, days, cols)
                shape = (tc.batch_windows, tc.window_days)
                g = self._rollout(
                    z.view(*shape, *z.shape[1:]),
                    preds.view(*shape, *preds.shape[1:]),
                    valid.view(*shape, -1),
                    data.intraday[days][:, sel].view(*shape, -1),
                    data.r1[days][:, sel].view(*shape, -1),
                )[:, tc.burn_in :]
                obj = self._objective(g)
                lab = data.labels[days][:, sel]
                lok = data.label_ok[days][:, sel] & valid.unsqueeze(-1)
                nll = student_t_nll(lab, preds[..., 0], preds[..., 1], self.cfg.t_dof)
                nll = (nll * lok).sum() / lok.sum().clamp(min=1)
                loss = -obj + tc.aux_weight * nll
                opt.zero_grad(set_to_none=True)
                loss.backward()  # type: ignore[no-untyped-call]
                torch.nn.utils.clip_grad_norm_(net.parameters(), tc.grad_clip)
                opt.step()
                tot["objective"] += obj.item() / tc.steps_per_epoch
                tot["nll"] += nll.item() / tc.steps_per_epoch
            rec = {"fold": float(self._folds), "epoch": float(epoch), **tot}
            if val.size:
                score = self._validate(data, layout, times, val)
                rec["val_objective"] = score
                if best is None or score > best[0]:
                    best = (score, copy.deepcopy(net.state_dict()))
                    stale = 0
                else:
                    stale += 1
            self.history.append(rec)
            if val.size and stale >= tc.patience:
                break
        if best is not None:
            net.load_state_dict(best[1])
        net.eval()

    def _sub_universe(
        self, data: PanelData, starts: Tensor, rng: np.random.Generator
    ) -> Tensor | None:
        """A random sorted subset of ``universe`` companies among those with
        data on the first day of any window (``None``: all companies)."""
        k = self.train.universe
        if k is None:
            return None
        live = np.flatnonzero(data.ok[starts].any(dim=0).numpy())
        if live.size <= k:
            return torch.from_numpy(live)
        return torch.from_numpy(np.sort(rng.choice(live, size=k, replace=False)))

    @torch.no_grad()
    def _validate(self, data: PanelData, layout: TokenLayout, times: I64, steps: I64) -> float:
        assert self.net is not None
        self.net.eval()
        zs, ps, vs = [], [], []
        days = torch.from_numpy(steps)
        for chunk in torch.split(days, self.train.eval_chunk):
            z, p, v = self._encode(data, layout, times, chunk)
            zs.append(z)
            ps.append(p)
            vs.append(v)
        g = self._rollout(
            torch.cat(zs)[None],
            torch.cat(ps)[None],
            torch.cat(vs)[None],
            data.intraday[days][None],
            data.r1[days][None],
        )[:, self.train.burn_in :]
        return float(self._objective(g))

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def targets(self, panel: MarketPanel, steps: I64) -> F64:
        steps = np.asarray(steps, dtype=np.int64)
        s = panel.shape[1]
        out = np.full((steps.size, s), np.nan)
        if steps.size == 0:
            return out
        if self.net is None:
            raise RuntimeError("fit must be called before targets")
        net = self.net
        net.eval()
        data = self._inputs(panel)
        layout = self.layout(panel)
        times = panel.decision_time
        carried = self._carry.get(int(steps[0]))
        book = copy.deepcopy(carried) if carried is not None else _Book(np.zeros(s), 1.0, 1.0)
        for k, t in enumerate(steps.tolist()):
            # Track the strategy's own equity for the overlay (the book
            # filled at this bar's open, marked close to close), then drift
            # the book to this close.
            ret = data.close_ret[t].double().numpy()
            book.equity *= 1.0 + float(book.weights @ np.expm1(ret))
            book.peak = max(book.peak, book.equity)
            cur = drift(torch.from_numpy(book.weights), data.intraday[t].double()).numpy()
            z, preds, valid = self._encode(data, layout, times, torch.tensor([t]))
            w = net.policy(z, preds, valid, torch.from_numpy(cur).float()[None])
            wc = w[0, :-1].double().numpy()
            if self.overlay is not None:
                wc = wc * self.overlay.exposure(1.0 - book.equity / book.peak)
            hold = np.abs(wc - cur) < self.band
            out[k] = np.where(hold, np.nan, wc)
            book.weights = np.where(hold, cur, wc)
        self._carry[int(steps[-1]) + 1] = copy.deepcopy(book)
        return out

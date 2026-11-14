"""PGPerceptron as a walk-forward :class:`~automoonbot.backtest.Strategy`.

Reproduces the original system end to end: the training environment
(``reinforce/environment.py``: reward shaping, episode termination), the
REINFORCE update (``compute_loss``), the training loop with its model
selection (``train``), the server's deployment rule (retrain every
``retrain_freq`` bars on the last ``max_training_data`` bars, deploy only if
the new model's in-sample "beat" ratio exceeds the best so far) and its live
inference (``inference`` with ``method="prob"``, buy size = action
probability, and the server's own :class:`TradeManager`).

The original only ever reported in-sample numbers. Here the same procedure is
scored out of sample through the execution simulator with realistic costs.

``variant="legacy"`` keeps every behavior of the original code, including
these defects found while porting it:

* The position fed to the network was never updated (always ``Cash``), in
  training and live.
* Rewards are paired with the log-probability of the *next* action: the
  reward of action ``a_k`` trains ``log pi(a_{k+1})``.
* ``best_model["weights"]`` held references to the live parameters, so the
  "revert to the best episode" was a no-op and the final weights are always
  those after the last episode (the reported score is still the best's).
* The live trade manager was built with the default ``leverage=0``, so its
  unrealized gain (a network input) was always 0 in a trade and -1 out.

``variant="repaired"`` fixes all four. Independently, ``selection`` chooses
how a retrained model is picked: ``"in_sample"`` (the original: best training
episode, deployed only on a new best beat ratio) or ``"validation"``: train on
the first part of the window, pick the episode that does best on the held-out
tail, and deploy it if it beats the deployed model on that same tail.
"""

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from automoonbot.backtest.panel import MarketPanel
from automoonbot.baselines.pgperceptron.descriptors import (
    FEATURE_PARAMS,
    QUEUE_SIZE,
    descriptors,
    feature_dim,
)
from automoonbot.baselines.pgperceptron.model import PolicyNet
from automoonbot.baselines.pgperceptron.trading import Action, Position, TradeManager

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass(frozen=True)
class Config:
    """Defaults are the original ``config.json``."""

    variant: Literal["legacy", "repaired"] = "legacy"
    selection: Literal["in_sample", "validation"] = "in_sample"
    validation_fraction: float = 0.25
    feature_params: Mapping[str, Any] = field(default_factory=lambda: FEATURE_PARAMS)
    queue_size: int = QUEUE_SIZE
    embedding_dim: int = 8
    num_mem: int = 1024
    mem_dim: int = 256
    inaction_cost: float = 10.0
    action_cost: float = 0.0
    return_thresh: float = 0.8
    sharpe_cutoff: int = 100
    train_leverage: float = 1.0
    max_training_data: int = 1000
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.1
    zeta: float = 0.5
    #: ``compute_loss`` was called without ``gamma``, so its default applied.
    discount: float = 0.99
    episodes: int = 100
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.9
    max_grad_norm: float = 1.0
    min_episodes: int = 10
    inference_method: Literal["prob", "argmax"] = "prob"
    min_prob: float = 0.31
    tz: str = "America/New_York"

    @property
    def legacy(self) -> bool:
        return self.variant == "legacy"


# ----------------------------------------------------------------- policy
class _NumpyPolicy:
    """Single-sample forward pass of a frozen :class:`PolicyNet` in numpy,
    for rollouts (several times faster than per-step torch calls)."""

    def __init__(self, net: PolicyNet) -> None:
        sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
        n_in = sd["f1.weight"].shape[1] - net.embedding.embedding_dim - 1
        self.w1x = np.ascontiguousarray(sd["f1.weight"][:, :n_in].T)
        self.emb = sd["embedding.weight"] @ sd["f1.weight"][:, n_in:-1].T + sd["f1.bias"]
        self.w1g = sd["f1.weight"][:, -1].copy()
        self.n1 = (sd["norm1.weight"], sd["norm1.bias"])
        self.n2 = (sd["norm2.weight"], sd["norm2.bias"])
        self.w2, self.b2 = np.ascontiguousarray(sd["f2.weight"].T), sd["f2.bias"]
        # fk and fv fused into one matmul; f4 split over its (v, q) input.
        self.wkv = np.ascontiguousarray(np.concatenate((sd["fk.weight"], sd["fv.weight"])).T)
        self.bkv = np.concatenate((sd["fk.bias"], sd["fv.bias"]))
        self.nk = sd["fk.weight"].shape[0]
        w4 = sd["f4.weight"]
        self.w4v = np.ascontiguousarray(w4[:, : self.nk].T)
        self.w4q = np.ascontiguousarray(w4[:, self.nk :].T)
        self.b4 = sd["f4.bias"]
        mem = sd["mem.mem"]
        self.mkey = np.ascontiguousarray(
            _softmax(mem @ sd["mem.fk.weight"].T + sd["mem.fk.bias"]).T
        )
        self.mval = np.maximum(mem @ sd["mem.fv.weight"].T + sd["mem.fv.bias"], 0.0)

    def project(self, x: F32) -> F32:
        """The feature part of the first layer for many rows at once."""
        out: F32 = np.asarray(x.astype(np.float32) @ self.w1x, dtype=np.float32)
        return out

    def probs(self, xw: F32, p: int, g: float, masks: tuple[F32, F32] | None = None) -> F32:
        h = np.maximum(xw + self.emb[p] + self.w1g * np.float32(g), 0.0)
        h = _layernorm(h, *self.n1)
        if masks is not None:
            h = h * masks[0]
        h = _layernorm(np.maximum(h @ self.w2 + self.b2, 0.0), *self.n2)
        if masks is not None:
            h = h * masks[1]
        kv = h @ self.wkv + self.bkv
        k = _softmax(kv[: self.nk])
        v = np.maximum(kv[self.nk :], 0.0)
        q = _softmax(k @ self.mkey) @ self.mval
        return _softmax(v @ self.w4v + q @ self.w4q + self.b4)


def _softmax(x: F32) -> F32:
    z = np.exp(x - x.max(axis=-1, keepdims=True))
    out: F32 = np.asarray(z / z.sum(axis=-1, keepdims=True), dtype=np.float32)
    return out


def _layernorm(x: F32, w: F32, b: F32, eps: float = 1e-5) -> F32:
    """Single-vector LayerNorm (biased variance, as torch)."""
    d = x - x.sum() / x.size
    out: F32 = d * (w / np.sqrt(d.dot(d) / x.size + eps)) + b
    return out


def _sample(probs: F32, rng: np.random.Generator) -> int:
    """Inverse-CDF draw from a probability vector."""
    c = np.cumsum(probs, dtype=np.float64)
    return min(int(np.searchsorted(c, rng.random() * c[-1], side="right")), probs.size - 1)


def _infer(
    pol: _NumpyPolicy, xw: F32, p: int, g: float, cfg: Config, rng: np.random.Generator
) -> tuple[int, float]:
    """The original ``inference``: sample, falling back to argmax when the
    sampled action's probability is below ``min(0.34, min_prob)``."""
    probs = pol.probs(xw, p, g)
    if cfg.inference_method == "argmax":
        a = int(np.argmax(probs))
    else:
        a = _sample(probs, rng)
        if probs[a] < min(0.34, cfg.min_prob):
            a = int(np.argmax(probs))
    return a, float(probs[a])


def _sharpe(returns: list[float], rf: float) -> float:
    """The original ``compute_sharpe_ratio`` (per-trade gains vs a
    buy-and-hold gross return)."""
    if not returns:
        return 0.0
    ex = np.asarray(returns) - rf
    sd = ex.std()
    return 0.0 if sd == 0 else float(ex.mean() / sd)


# ------------------------------------------------------------ environment
@dataclass
class _Episode:
    rows: list[int]
    positions: list[int]
    potentials: list[float]
    #: Dropout masks of the recorded steps (training rollouts only).
    masks1: F32
    masks2: F32
    actions: list[int]
    rewards: list[float]
    portfolio: float
    log_return: float
    final_close: float


def _rollout(
    pol: _NumpyPolicy,
    xw: F32,
    close: F64,
    cov: F64,
    cfg: Config,
    rng: np.random.Generator,
    *,
    train: bool,
) -> _Episode:
    """One episode of the original ``TradeEnv`` over bars ``0..n-1`` of the
    given window. With ``train`` the policy samples with dropout (masks
    recorded); otherwise it acts like live inference (no dropout)."""
    n = close.size
    mgr = TradeManager(cov[0], cfg.alpha, cfg.gamma, cfg.action_cost, cfg.train_leverage)
    init = close[0]
    empty = np.zeros((0, 0), dtype=np.float32)
    ep = _Episode([], [], [], empty, empty, [], [], 1.0, 0.0, init)
    action = int(Action.Hold)  # train() always starts with env.step(1)
    if train:
        width1, width2 = pol.emb.shape[1], pol.n2[0].size
        masks1 = (rng.random((n, width1)) < 0.5).astype(np.float32) * np.float32(2.0)
        masks2 = (rng.random((n, width2)) < 0.5).astype(np.float32) * np.float32(2.0)
    for k in range(1, n):
        price, cv = close[k], cov[k]
        done = k == n - 1
        reward = 0.0
        rf = price / init
        if action == Action.Buy:
            a = mgr.try_buy(price, cv)
            if a == Action.Double:
                reward -= cfg.action_cost * (mgr.potential_gain(price) - 1)
            else:
                reward -= cfg.action_cost
        elif action == Action.Sell:
            gain = mgr.try_sell(price, cv)
            if gain > 0:
                if mgr.portfolio < cfg.return_thresh:
                    done = True
                ep.log_return += math.log(gain)
                sr = _sharpe(mgr.returns[-cfg.sharpe_cutoff :], rf * (1 - cfg.action_cost))
                double = mgr.last_trade.amount > 0.5
                reward += (1 - cfg.zeta) * sr + cfg.zeta * (gain - 1) * (2 if double else 1)
        else:  # Hold
            if mgr.position in (Position.Asset, Position.Partial):
                reward -= cfg.inaction_cost * mgr.potential_gain(price)
            else:
                prev = mgr.prev_exit if mgr.prev_exit > 0 else price
                reward -= cfg.inaction_cost * (price / prev)
        pos = 0 if cfg.legacy else int(mgr.position)
        g = mgr.potential_gain(price) - 1
        if train:
            action = _sample(pol.probs(xw[k], pos, g, (masks1[k], masks2[k])), rng)
        else:
            action, _ = _infer(pol, xw[k], pos, g, cfg, rng)
        ep.rows.append(k)
        ep.positions.append(pos)
        ep.potentials.append(g)
        ep.actions.append(action)
        ep.rewards.append(reward)
        ep.final_close = price
        if done:
            break
    ep.portfolio = mgr.portfolio
    if train:
        used = np.asarray(ep.rows)
        ep.masks1, ep.masks2 = masks1[used], masks2[used]
    return ep


def _loss(net: PolicyNet, x: torch.Tensor, ep: _Episode, cfg: Config) -> torch.Tensor:
    """The original ``compute_loss`` on one episode, with log-probabilities
    recomputed in a single batched pass using the rollout's dropout masks."""
    rows = torch.as_tensor(ep.rows)
    probs = net(
        x[rows],
        torch.as_tensor(ep.positions, dtype=torch.long),
        torch.as_tensor(ep.potentials, dtype=torch.float32)[:, None],
        masks=(torch.from_numpy(ep.masks1), torch.from_numpy(ep.masks2)),
    )
    logp = torch.log(probs[torch.arange(len(ep.actions)), torch.as_tensor(ep.actions)])
    rewards = np.asarray(ep.rewards, dtype=np.float64)
    if not cfg.legacy:
        # Credit each action with the reward it earned (one step later).
        logp, rewards = logp[:-1], rewards[1:]
    disc = np.empty_like(rewards)
    acc = 0.0
    for i in range(rewards.size - 1, -1, -1):
        acc = rewards[i] + cfg.discount * acc
        disc[i] = acc
    d = torch.as_tensor(disc, dtype=torch.float32)
    sd = d.std() if d.numel() > 1 else torch.tensor(0.0)
    norm = (d - d.mean()) / (torch.nan_to_num(sd) + 1e-9)
    adv = cfg.beta * norm + (1 - cfg.beta) * ep.log_return
    return -(logp * adv).sum()


# --------------------------------------------------------------- strategy
@dataclass
class _Book:
    mgr: TradeManager


class PGPerceptron:
    """See the module docstring. Trades ``security`` of the panel; every
    other security gets weight 0."""

    def __init__(self, config: Config | None = None, *, security: int = 0, seed: int = 0) -> None:
        self.cfg = config or Config()
        if not 0 < self.cfg.validation_fraction < 1:
            raise ValueError("validation_fraction must be in (0, 1)")
        self.security = security
        self.seed = seed
        self.name = f"pgperceptron[{self.cfg.variant},{self.cfg.selection}]"
        self._net: PolicyNet | None = None
        self._opt: torch.optim.Optimizer | None = None
        self._deployed: _NumpyPolicy | None = None
        self._beat = 0.0
        self._fits = 0
        self._carry: dict[int, _Book] = {}
        self._cache: tuple[I64, F64, F64, F64] | None = None
        self.history: list[dict[str, float]] = []

    @property
    def params(self) -> dict[str, object]:
        return {"variant": self.cfg.variant, "selection": self.cfg.selection, "seed": self.seed}

    # -------------------------------------------------------------- data
    def _features(self, panel: MarketPanel) -> tuple[F64, F64]:
        """Features and coefficient of variation for the traded security,
        cached per panel length (rows of a longer panel equal those of its
        prefixes, by causality)."""
        t = panel.shape[0]
        col = panel.close[:, self.security]
        if self._cache is not None:
            times, closes, feats, cv = self._cache
            if (
                t <= times.size
                and np.array_equal(panel.times, times[:t])
                and np.array_equal(col, closes[:t], equal_nan=True)
            ):
                return feats[:t], cv[:t]
        cols: dict[str, Any] = {
            f: (
                v[:, self.security : self.security + 1]
                if isinstance(v, np.ndarray) and v.ndim == 2
                else v
            )
            for f, v in vars(panel).items()
        }
        single = MarketPanel(**cols)
        single.securities = [panel.securities[self.security]]
        f, cov = descriptors(
            single, self.cfg.feature_params, queue=self.cfg.queue_size, tz=self.cfg.tz
        )
        feats, cv = f[:, 0, :], cov[:, 0]
        if self._cache is None or t > self._cache[0].size:
            self._cache = (panel.times.copy(), col.copy(), feats, cv)
        return feats, cv

    def _prices(self, panel: MarketPanel) -> F64:
        p: F64 = panel.total_return_index()[:, self.security]
        return p

    # ---------------------------------------------------------- training
    def _ensure_net(self, dim: int) -> PolicyNet:
        if self._net is None:
            torch.manual_seed(self.seed)
            self._net = PolicyNet(
                dim, 3, len(Position), self.cfg.embedding_dim, self.cfg.num_mem, self.cfg.mem_dim
            )
        return self._net

    def fit(self, panel: MarketPanel, train: I64) -> None:
        cfg = self.cfg
        feats, cov = self._features(panel)
        close = self._prices(panel)
        rows = np.asarray(train, dtype=np.int64)
        ok = (
            np.isfinite(feats[rows]).all(axis=1) & np.isfinite(close[rows]) & np.isfinite(cov[rows])
        )
        rows = rows[ok][-cfg.max_training_data :]
        net = self._ensure_net(feature_dim(cfg.feature_params))
        self._fits += 1
        if rows.size < 3:
            return
        rng = np.random.default_rng([self.seed, self._fits])
        x = torch.as_tensor(feats[rows], dtype=torch.float32)
        if cfg.selection == "in_sample":
            beat = self._train(net, x, close[rows], cov[rows], rng)
            if beat > self._beat or self._deployed is None:
                self._beat = max(beat, self._beat)
                self._deployed = _NumpyPolicy(net)
            self.history.append({"fit": self._fits, "beat": beat, "deployed_beat": self._beat})
            return
        split = int(rows.size * (1 - cfg.validation_fraction))
        if split < 3 or rows.size - split < 3:
            return
        vx = x[split:]
        best_state, best_score = self._train(
            net,
            x[:split],
            close[rows[:split]],
            cov[rows[:split]],
            rng,
            val=(vx, close[rows[split:]], cov[rows[split:]]),
        )
        net.load_state_dict(best_state)
        cand = _NumpyPolicy(net)
        incumbent = (
            self._score(self._deployed, vx, close[rows[split:]], cov[rows[split:]])
            if self._deployed is not None
            else -math.inf
        )
        if best_score >= incumbent:
            self._deployed = cand
        self.history.append({"fit": self._fits, "validation": best_score, "incumbent": incumbent})

    def _score(self, pol: _NumpyPolicy, x: torch.Tensor, close: F64, cov: F64) -> float:
        """Validation score: portfolio / buy-and-hold over the window, acting
        as live inference would (no dropout, fixed seed)."""
        ep = _rollout(
            pol,
            pol.project(x.numpy()),
            close,
            cov,
            self.cfg,
            np.random.default_rng([self.seed, 0x5EED]),
            train=False,
        )
        return float(ep.portfolio / (close[-1] / close[0]))

    def _train(
        self,
        net: PolicyNet,
        x: torch.Tensor,
        close: F64,
        cov: F64,
        rng: np.random.Generator,
        val: tuple[torch.Tensor, F64, F64] | None = None,
    ) -> Any:
        """The original ``train`` loop. Returns the beat ratio (in-sample
        selection) or ``(best_state, best_validation_score)``."""
        cfg = self.cfg
        # A fresh optimizer per call, as in the original.
        opt = torch.optim.SGD(
            net.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
        xn = x.numpy()
        bh: float | None = None
        best_result, best_state = -1.0, copy.deepcopy(net.state_dict())
        best_val = -math.inf
        for e in range(cfg.episodes):
            net.train()
            pol = _NumpyPolicy(net)
            ep = _rollout(pol, pol.project(xn), close, cov, cfg, rng, train=True)
            opt.zero_grad()
            loss = _loss(net, x, ep, cfg)
            loss.backward()  # type: ignore[no-untyped-call]
            nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
            opt.step()
            net.eval()
            if bh is None:
                bh = ep.final_close / close[0]
            if val is not None:
                score = self._score(_NumpyPolicy(net), *val)
                if score > best_val:
                    best_val, best_state = score, copy.deepcopy(net.state_dict())
                continue
            if ep.portfolio > best_result:
                best_result = ep.portfolio
                if not cfg.legacy:
                    best_state = copy.deepcopy(net.state_dict())
            if ep.portfolio > max(1.0, bh) and e + 1 >= cfg.min_episodes:
                break
            if not cfg.legacy:
                net.load_state_dict(best_state)
        if val is not None:
            return best_state, best_val
        if not cfg.legacy:
            net.load_state_dict(best_state)
        return best_result / (bh or 1.0)

    # --------------------------------------------------------- inference
    def targets(self, panel: MarketPanel, steps: I64) -> F64:
        cfg = self.cfg
        steps = np.asarray(steps, dtype=np.int64)
        out = np.full((steps.size, panel.shape[1]), np.nan)
        others = np.arange(panel.shape[1]) != self.security
        out[:, others] = 0.0
        if steps.size == 0:
            return out
        feats, cov = self._features(panel)
        close = self._prices(panel)
        live_leverage = 0.0 if cfg.legacy else cfg.train_leverage
        start = int(steps[0])
        book = self._carry.get(start)
        mgr = (
            copy.deepcopy(book.mgr)
            if book is not None
            else TradeManager(
                float(np.nan_to_num(cov[start], nan=0.0)),
                cfg.alpha,
                cfg.gamma,
                cfg.action_cost,
                live_leverage,
            )
        )
        rng = np.random.default_rng([self.seed, start])
        pol = self._deployed
        if pol is None:
            out[:, self.security] = 0.0
            return out
        xw = pol.project(np.nan_to_num(feats[steps]))
        valid = (
            np.isfinite(feats[steps]).all(axis=1)
            & np.isfinite(close[steps])
            & np.isfinite(cov[steps])
        )
        for k, t in enumerate(steps):
            if not valid[k] or not panel.tradable[t, self.security]:
                continue
            price, cv = float(close[t]), float(cov[t])
            pos = 0 if cfg.legacy else int(mgr.position)
            a, prob = _infer(pol, xw[k], pos, mgr.potential_gain(price) - 1, cfg, rng)
            if a == Action.Buy:
                done = mgr.try_buy(price, cv, prob)
                if done in (Action.Buy, Action.Double):
                    out[k, self.security] = mgr.trade.amount
            elif a == Action.Sell:
                was_open = mgr.trade.opened
                mgr.try_sell(price, cv)
                if was_open and not mgr.trade.opened:
                    out[k, self.security] = 0.0
        self._carry[int(steps[-1]) + 1] = _Book(copy.deepcopy(mgr))
        return out


def variants(seed: int = 0, **overrides: Any) -> list[PGPerceptron]:
    """The three configurations reported against the baselines."""
    base = replace(Config(), **overrides)
    return [
        PGPerceptron(replace(base, variant="legacy", selection="in_sample"), seed=seed),
        PGPerceptron(replace(base, variant="repaired", selection="in_sample"), seed=seed),
        PGPerceptron(replace(base, variant="repaired", selection="validation"), seed=seed),
    ]

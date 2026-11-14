"""Graph transformer with multi-horizon (MTP-style) heads and a portfolio
policy head.

Architecture, per decision day:

1. **Temporal encoder.** Each company's last ``K`` days of features
   (``[K, C]`` plus a presence channel) are cut into patches, embedded and
   mixed by a small transformer over the patch sequence (PatchTST-style),
   then pooled into one token. Only companies with data that day are
   encoded.
2. **Graph transformer.** Company, industry and sector tokens go through
   GPS-style layers (Rampasek et al., 2022): typed message passing along
   the graph's known edges (one linear map per relation and direction,
   mean-aggregated), then full multi-head attention over all tokens, then
   a feed-forward block. Tokens that are absent that day (no data, or a
   class not yet known) send no messages and are masked as keys. Global
   attention over ~650 tokens is cheap at universe scale and lets the model
   find relationships the graph does not state; message passing injects the
   ones it does. (A learned per-edge attention bias, Graphormer-style, was
   measured ~5x slower on CPU: its gradient needs the full attention
   matrix, which fused attention kernels never materialize.)
3. **Multi-horizon heads.** Chained residual modules, one per horizon
   (e.g. 1, 5, 21 days), each predicting a Student-t (fixed degrees of
   freedom) over the forward total log return; module ``k`` refines module
   ``k - 1``'s representation, as in multi-token prediction, so longer
   horizons build on shorter ones. Heavy tails keep buyouts and crashes
   from dominating the fit.
4. **Policy head.** A market-level head on the pooled company
   representations sets the invested fraction; a softmax over company
   scores (from each company's representation, its predicted return
   distributions and its current weight) allocates it. Weights are long
   only and each is capped (the excess goes to cash), in training and
   inference alike. Neither part depends on the number of companies, so
   training on random sub-universes transfers to the full one.

The policy is trained by gradient ascent on the realized, cost-adjusted
portfolio log-growth with a CVaR penalty (see
:mod:`automoonbot.models.strategy`). Prices are exogenous to the policy, so
this objective is differentiable end to end and needs no critic.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelConfig:
    channels: int
    window: int = 64
    patch: int = 8
    d_model: int = 64
    heads: int = 4
    temporal_layers: int = 1
    graph_layers: int = 3
    ff_mult: int = 4
    dropout: float = 0.1
    horizons: tuple[int, ...] = (1, 5, 21)
    #: Class-node vocabulary size (industries and sectors); id 0 is
    #: reserved for classes unseen when the model was fitted.
    n_classes: int = 1
    n_edge_types: int = 4
    #: Student-t degrees of freedom of the return heads.
    t_dof: float = 5.0
    #: Maximum weight per company (the excess goes to cash).
    max_weight: float = 0.05

    def __post_init__(self) -> None:
        if self.window % self.patch:
            raise ValueError("window must be a multiple of patch")
        if self.d_model % self.heads:
            raise ValueError("d_model must be divisible by heads")
        if not self.horizons or min(self.horizons) < 1:
            raise ValueError("horizons must be positive")
        if not 0.0 < self.max_weight <= 1.0:
            raise ValueError("max_weight must be in (0, 1]")
        if self.t_dof <= 2.0:
            raise ValueError("t_dof must exceed 2 (finite variance)")


class Block(nn.Module):
    """Pre-norm transformer block with an optional key mask."""

    def __init__(self, d: int, heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, ff_mult * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_mult * d, d)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, key_ok: Tensor | None = None) -> Tensor:
        """``x[B, N, d]``; ``key_ok[B, N]`` masks absent tokens as keys.
        (No dropout on attention weights: it forces the unfused kernel.)"""
        b, n, d = x.shape
        q, k, v = (
            self.qkv(self.norm1(x))
            .view(b, n, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        mask = None if key_ok is None else key_ok[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(b, n, d)))
        out: Tensor = x + self.drop(self.ff(self.norm2(x)))
        return out


@dataclass(frozen=True)
class Edges:
    """Typed edges of a batch of days, flattened over ``B * N`` tokens."""

    src: Tensor
    dst: Tensor
    etype: Tensor
    #: ``1 / in-degree`` of the destination for its relation, counting only
    #: present sources; 0 for edges from absent sources.
    weight: Tensor
    key_ok: Tensor

    @staticmethod
    def build(n_tokens: int, edges: Tensor, key_ok: Tensor, n_types: int) -> Edges:
        """From ``edges[4, E] = (day, src, dst, type)`` and ``key_ok[B, N]``."""
        day, src, dst, et = edges
        fs, fd = day * n_tokens + src, day * n_tokens + dst
        live = key_ok.flatten()[fs].to(torch.float32)
        slot = fd * n_types + et
        deg = torch.zeros(key_ok.numel() * n_types).index_add_(0, slot, live)
        weight = live / deg[slot].clamp(min=1.0)
        return Edges(fs, fd, et, weight, key_ok)


class MessagePassing(nn.Module):
    """Relational mean aggregation: ``h_i += W_o sum_r mean_{j ->_r i} W_r h_j``."""

    def __init__(self, d: int, n_types: int, dropout: float) -> None:
        super().__init__()
        self.n_types = n_types
        self.norm = nn.LayerNorm(d)
        self.msg = nn.Linear(d, n_types * d, bias=False)
        self.out = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, e: Edges) -> Tensor:
        b, n, d = x.shape
        if e.src.numel() == 0:
            return x
        m = self.msg(self.norm(x)).view(b * n, self.n_types, d)
        sent = m[e.src, e.etype] * e.weight.to(m.dtype).unsqueeze(-1)
        agg = torch.zeros(b * n, d, dtype=sent.dtype, device=x.device).index_add_(0, e.dst, sent)
        out: Tensor = x + self.drop(self.out(F.gelu(agg)).view(b, n, d))
        return out


class GraphBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.mp = MessagePassing(cfg.d_model, cfg.n_edge_types, cfg.dropout)
        self.attn = Block(cfg.d_model, cfg.heads, cfg.ff_mult, cfg.dropout)

    def forward(self, x: Tensor, e: Edges) -> Tensor:
        out: Tensor = self.attn(self.mp(x, e), e.key_ok)
        return out


class TemporalEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.patch = cfg.patch
        n_patches = cfg.window // cfg.patch
        self.embed = nn.Linear(cfg.patch * (cfg.channels + 1), cfg.d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_patches, cfg.d_model))
        nn.init.normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList(
            Block(cfg.d_model, cfg.heads, cfg.ff_mult, cfg.dropout)
            for _ in range(cfg.temporal_layers)
        )
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, x: Tensor, present: Tensor) -> Tensor:
        """``x[M, K, C]`` (zeros where absent), ``present[M, K]`` -> ``[M, d]``."""
        m, k, c = x.shape
        z = torch.cat((x, present.unsqueeze(-1).to(x.dtype)), dim=-1)
        z = z.reshape(m, k // self.patch, self.patch * (c + 1))
        h = self.embed(z) + self.pos
        for blk in self.blocks:
            h = blk(h)
        out: Tensor = self.norm(h).mean(dim=1)
        return out


class HorizonModule(nn.Module):
    """One MTP step: refine the representation, predict (mu, log_scale)."""

    def __init__(self, d: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d)
        )
        self.head = nn.Linear(d, 2)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        z = z + self.mlp(self.norm(z))
        return z, self.head(z)


class GraphTransformer(nn.Module):
    #: Typical scale of each horizon's log return (buffer).
    unit: Tensor

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.temporal = TemporalEncoder(cfg)
        self.kind = nn.Embedding(3, d)
        self.classes = nn.Embedding(cfg.n_classes + 1, d)
        self.missing = nn.Parameter(torch.zeros(d))
        self.blocks = nn.ModuleList(GraphBlock(cfg) for _ in range(cfg.graph_layers))
        self.norm = nn.LayerNorm(d)
        self.horizons = nn.ModuleList(HorizonModule(d, cfg.dropout) for _ in cfg.horizons)
        # Each head predicts in units of its horizon's typical scale, so the
        # zero-initialized heads start at a sensible distribution.
        self.register_buffer(
            "unit", torch.tensor([0.02 * math.sqrt(h) for h in cfg.horizons]), persistent=False
        )
        n_h = len(cfg.horizons)
        self.score = nn.Sequential(nn.Linear(d + 2 * n_h + 1, d), nn.GELU(), nn.Linear(d, 1))
        self.cash = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def edges(self, n_tokens: int, edges: Tensor, key_ok: Tensor) -> Edges:
        """Message-passing structure for ``edges[4, E] = (day, src, dst,
        type)`` and token presence ``key_ok[B, N]``."""
        # A day with no token present at all (no data, nothing known) would
        # give NaN attention; its outputs are unused, so unmask it.
        key_ok = key_ok | ~key_ok.any(dim=1, keepdim=True)
        return Edges.build(n_tokens, edges, key_ok, self.cfg.n_edge_types)

    def encode(
        self,
        x: Tensor,
        present: Tensor,
        pair: Tensor,
        valid: Tensor,
        token_kind: Tensor,
        class_id: Tensor,
        edges: Edges,
    ) -> tuple[Tensor, Tensor]:
        """Company representations and per-horizon predictions.

        ``x[M, K, C]`` and ``present[M, K]`` are the feature windows of the
        ``M`` valid (day, company) pairs, at flat positions ``pair[M]`` of
        ``valid[B, S]``. ``token_kind``/``class_id`` describe all ``N >= S``
        tokens (companies first); ``edges`` is from :meth:`edges`.
        Returns ``z[B, S, d]`` and ``preds[B, S, H, 2]`` (mu, log_scale) in
        log-return units.
        """
        b, s = valid.shape
        d = self.cfg.d_model
        n = token_kind.numel()
        comp = self.missing.expand(b * s, d)
        if pair.numel():
            comp = comp.index_copy(0, pair, self.temporal(x, present))
        comp = comp.view(b, s, d)
        cls = self.classes(class_id[s:]) + self.kind(token_kind[s:])
        h = torch.cat((comp + self.kind(token_kind[:s]), cls.unsqueeze(0).expand(b, n - s, d)), 1)
        for blk in self.blocks:
            h = blk(h, edges)
        z = self.norm(h)[:, :s].float()
        preds = []
        zk = z
        for mod in self.horizons:
            zk, p = mod(zk)
            preds.append(p)
        raw = torch.stack(preds, dim=2).float()
        mu = raw[..., 0] * self.unit
        log_scale = (raw[..., 1] + self.unit.log()).clamp(-12.0, 2.0)
        return z, torch.stack((mu, log_scale), dim=-1)

    def policy(self, z: Tensor, preds: Tensor, valid: Tensor, prev: Tensor) -> Tensor:
        """Weights ``[B, S + 1]`` (last column is cash) for representations
        ``z[B, S, d]``, predictions ``preds[B, S, H, 2]`` and current
        (drifted) weights ``prev[B, S]``. Invalid companies get weight 0."""
        cnt = valid.sum(dim=1, keepdim=True).clamp(min=1)
        # Predictions in units of each horizon's scale; current weight
        # relative to equal weight, so inputs are O(1).
        p = torch.stack((preds[..., 0] / self.unit, preds[..., 1] - self.unit.log()), dim=-1)
        rel = (prev * cnt).unsqueeze(-1)
        feats = torch.cat((z, p.flatten(2), rel), dim=-1)
        scores = self.score(feats).squeeze(-1).masked_fill(~valid, float("-inf"))
        pooled = (z * valid.unsqueeze(-1)).sum(dim=1) / cnt
        # Exposure from the market representation, allocation from the
        # cross-section: neither depends on how many companies there are.
        invested = torch.sigmoid(-self.cash(pooled))
        alloc = torch.softmax(scores, dim=1).nan_to_num(0.0)  # no valid company: all NaN
        w = (invested * alloc).clamp(max=self.cfg.max_weight)
        return torch.cat((w, 1.0 - w.sum(dim=1, keepdim=True)), dim=1)


def student_t_nll(y: Tensor, mu: Tensor, log_scale: Tensor, dof: float) -> Tensor:
    """Elementwise negative log-likelihood of a location-scale Student-t."""
    z = (y - mu) * torch.exp(-log_scale)
    const = math.lgamma((dof + 1) / 2) - math.lgamma(dof / 2) - 0.5 * math.log(dof * math.pi)
    return log_scale + 0.5 * (dof + 1) * torch.log1p(z * z / dof) - const

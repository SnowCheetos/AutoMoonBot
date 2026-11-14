"""PGPerceptron's policy network (``reinforce/model.py``), same architecture
and initialization, with dropout masks that can be supplied explicitly.

The original sampled every action through a separate autograd graph. Here a
rollout runs without gradients, recording the dropout masks it drew, and the
episode's log-probabilities are then recomputed in one batched forward pass
with exactly those masks. The gradient is identical (dropout is the only
stochastic layer) and training is an order of magnitude faster.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

DROPOUT = 0.5


class MemoryNet(nn.Module):
    def __init__(self, num_mem: int, mem_dim: int, inp_dim: int) -> None:
        super().__init__()
        self.mem = nn.Parameter(torch.randn((num_mem, mem_dim), dtype=torch.float32))
        self.fk = nn.Linear(mem_dim, inp_dim)
        self.fv = nn.Linear(mem_dim, inp_dim)
        nn.init.kaiming_uniform_(self.fk.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.fv.weight, a=math.sqrt(5))

    def keys_values(self) -> tuple[Tensor, Tensor]:
        return torch.softmax(self.fk(self.mem), dim=-1), torch.relu(self.fv(self.mem))

    def forward(self, k: Tensor, kv: tuple[Tensor, Tensor] | None = None) -> Tensor:
        key, val = kv if kv is not None else self.keys_values()
        return torch.softmax(k @ key.t(), dim=-1) @ val


class PolicyNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 3,
        position_dim: int = 3,
        embedding_dim: int = 8,
        num_mem: int = 1024,
        mem_dim: int = 256,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(position_dim, embedding_dim)
        self.f1 = nn.Linear(input_dim + embedding_dim + 1, 512)
        self.f2 = nn.Linear(512, 256)
        self.fk = nn.Linear(256, 128)
        self.fv = nn.Linear(256, 128)
        self.f4 = nn.Linear(256, output_dim)
        self.mem = MemoryNet(num_mem, mem_dim, 128)
        self.norm1 = nn.LayerNorm(512)
        self.norm2 = nn.LayerNorm(256)

    def forward(
        self,
        x: Tensor,
        p: Tensor,
        g: Tensor,
        masks: tuple[Tensor, Tensor] | None = None,
        kv: tuple[Tensor, Tensor] | None = None,
    ) -> Tensor:
        """Action probabilities ``[n, output_dim]`` for features ``x[n, F]``,
        positions ``p[n]`` (long) and portfolio inputs ``g[n, 1]``.

        In training mode, dropout uses ``masks`` (already scaled by
        ``1 / (1 - p)``) if given, else draws new ones. ``kv`` optionally
        passes precomputed memory keys/values.
        """
        h = torch.cat((x, self.embedding(p), g), dim=-1)
        h = self.norm1(torch.relu(self.f1(h)))
        h = self._drop(h, None if masks is None else masks[0])
        h = self.norm2(torch.relu(self.f2(h)))
        h = self._drop(h, None if masks is None else masks[1])
        k = torch.softmax(self.fk(h), dim=-1)
        v = torch.relu(self.fv(h))
        return torch.softmax(self.f4(torch.cat((v, self.mem(k, kv)), dim=-1)), dim=-1)

    def _drop(self, h: Tensor, mask: Tensor | None) -> Tensor:
        if not self.training:
            return h
        if mask is None:
            return nn.functional.dropout(h, DROPOUT, training=True)
        return h * mask


def draw_masks(n: int, generator: torch.Generator, device: torch.device) -> tuple[Tensor, Tensor]:
    """Scaled dropout masks for ``n`` forward passes."""
    keep = 1.0 - DROPOUT

    def one(width: int) -> Tensor:
        m = torch.rand((n, width), generator=generator, device=device) < keep
        return m.to(torch.float32) / keep

    return one(512), one(256)

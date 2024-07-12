import torch
import torch.nn as nn

from torch import Tensor


class MultiHeadMemory(nn.Module):
    def __init__(
        self,
        heads: int,
        mem_size: int,
        mem_dim: int,
        key_dim: int,
        val_dim: int,
    ) -> None:
        super().__init__()

        self._mem = nn.Parameter(torch.randn((heads, mem_size, mem_dim)).float())
        self._fk = nn.Linear(mem_dim, key_dim)
        self._fv = nn.Linear(mem_dim, val_dim)
        self._fx = nn.Linear(val_dim * heads, val_dim)

    def forward(self, q: Tensor) -> Tensor:
        """
        q: [b x k] Query tensor
        """
        k = self._fk(self._mem)  # [h x n x k]
        k = torch.softmax(k, dim=-1)  # Memory keys

        v = self._fv(self._mem)  # [h x n x v]
        v = torch.relu(v)

        q = q.unsqueeze(1)  # [b x 1 x k]
        a = torch.einsum("bqk,hnk->bhn", q, k)  # [b x h x n]
        w = torch.softmax(a, dim=-1)  # Attention weights
        v = torch.einsum("bhn,hnv->bhv", w, v)  # [b x h x v]

        x = v.reshape(v.size(0), -1)  # [b x (h * v)]
        x = self._fx(x)  # Queried memory
        x = torch.relu(x)
        return x

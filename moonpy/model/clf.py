import torch
import torch.nn as nn
from torch import Tensor

from moonpy.model import MultiHeadMemory


class MemoryClassifier(nn.Module):
    """
    position classifier, evaluates actions for current positions
    """
    def __init__(
        self,
        inp_dim: int,
        hdn_dim: int,
        out_dim: int,
        mem_heads: int,
        mem_size: int,
        mem_dim: int,
        key_dim: int,
        val_dim: int,
    ) -> None:
        super().__init__()

        self.mem = MultiHeadMemory(mem_heads, mem_size, mem_dim, key_dim, val_dim)

        self.query = nn.Linear(inp_dim, mem_dim)
        self.value = nn.Linear(inp_dim, val_dim)

        self.lin1 = nn.Linear(2*val_dim, hdn_dim)
        self.lin2 = nn.Linear(hdn_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: sagenet output nodes that require classification
        """
        q = self.query(x)
        v = self.value(x)
        m = self.mem(q) # queries memory and see if it has seen similar stuff before

        x = torch.cat((v, m), dim=-1)
        x = self.lin1(x)
        x = x.relu()

        x = self.lin2(x)
        return x.softmax(-1)

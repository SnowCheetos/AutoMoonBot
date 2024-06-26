import logging
import numpy as np

import torch
import torch.nn as nn

from typing import Tuple, List
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, LayerNorm, SAGPooling, global_mean_pool


class MarketGraphNet(nn.Module):
    def __init__(
            self, 
            inp_dim: int, 
            out_dim: int) -> None:
        
        super().__init__()

        self._c1 = GCNConv(inp_dim, 512, improved=True)
        self._c2 = GCNConv(512, 256, improved=True)
        self._n1 = LayerNorm(512)
        self._n2 = LayerNorm(256)
        self._p1 = SAGPooling(512)
        self._p2 = SAGPooling(256)
        self._f  = nn.Linear(256, out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        
        x = self._c1(x, edge_index)
        x = torch.relu(self._n1(x))
        x, edge_index, _, _, _, _ = self._p1(x, edge_index)

        x = self._c2(x, edge_index)
        x = torch.relu(self._n2(x))
        x, edge_index, _, _, _, _ = self._p2(x, edge_index)
        
        x = global_mean_pool(x, None)
        return self._f(x)


class MemoryNet(nn.Module):
    def __init__(
            self, 
            num_mem: int, 
            mem_dim: int, 
            key_dim: int, 
            val_dim: int) -> None:
        
        super().__init__()

        self._mem = nn.Parameter(torch.randn((num_mem, mem_dim), dtype=torch.float32))
        self._fk  = nn.Linear(mem_dim, key_dim)
        self._fv  = nn.Linear(mem_dim, val_dim)

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """
        k: [n x d]
        """
        key = torch.softmax(self._fk(self._mem), dim=-1) # [h x k]
        val = self._fv(self._mem)                        # [h x v]

        # Compute attention scores
        att = torch.matmul(k, key.t()) # [n x h]

        # Apply softmax to get attention weights
        att_weights = torch.softmax(att, dim=-1) # [n x h]

        # Compute the weighted sum of the values
        output = torch.matmul(att_weights, val) # [n x v]

        return output


class MultiHeadMemory(nn.Module):
    def __init__(
            self, 
            heads:    int, 
            mem_size: int, 
            mem_dim:  int, 
            key_dim:  int, 
            val_dim:  int) -> None:
        
        super().__init__()

        self._mems = [MemoryNet(mem_size, mem_dim, key_dim, val_dim) for _ in range(heads)]
        self._f    = nn.Linear(heads * val_dim, val_dim)

    def forward(self, k):
        outputs = []
        for m in self._mems:
            outputs += [m(k)]
        
        x = torch.cat(outputs, dim=-1)

        return self._f(x)


class PolicyNet(nn.Module):
    def __init__(
            self, 
            inp_dim:   int, 
            out_dim:   int, 
            inp_types: int,
            emb_dim:   int, 
            mem_heads: int,
            mem_size:  int, 
            mem_dim:   int,
            key_dim:   int,
            val_dim:   int) -> None:
        
        super().__init__()

        self._embedding = nn.Embedding(inp_types, emb_dim)

        self._gnn       = MarketGraphNet(inp_dim, 256)
        self._f1        = nn.Linear(inp_dim + emb_dim + val_dim * 2 + 1, 512)
        self._f2        = nn.Linear(512, 256)
        self._f3        = nn.Linear(256, out_dim)

        self._fk        = nn.Linear(256, key_dim)
        self._fv        = nn.Linear(256, val_dim)

        self._mem       = MultiHeadMemory(mem_heads, mem_size, mem_dim, key_dim, val_dim)
        
        self._d  = nn.Dropout(p=0.5)
        self._n1 = nn.LayerNorm(512)
        self._n2 = nn.LayerNorm(256)

    def forward(
            self,
            data:       Data,       # Market graph data
            index:      int,        # Node index of the position of the ticker
            inp_types:  List[int],  # Type of input, trade or market
            log_return: List[float] # Log return of the input
    ) -> torch.Tensor:
        """
        Forward pass for the PolicyNet. Data contains the features from market data in graph format.

        Parameters:
        data       (torch_geometric.data.Data): Input features tensor.
        index      (int):                       Node feature index.
        inp_type   (List[int]):                 Type of input, trade or market.
        log_return (List[float]):               Log return of the input.
        
        Returns:
        torch.Tensor: Output tensor after passing through the network.
        """
        assert len(inp_types) == len(log_return), "input types and log returns must have the same length"

        z = self._gnn(data)                    # Computes the market representation tensor, [1 x v]
        k = torch.softmax(self._fk(z), dim=-1) # Computes the market key tensor, [1 x k]
        v = self._fv(z)                        # Computes the market value tensor, [1 x v]
        q = self._mem(k)                       # Queries the memory, [1 x v]
        z = torch.cat((v, q), dim=-1)          # Concats the market value and memory, [1 x 2v]
        x = data.x[index][None, :]             # Index the feature, [1 x inp]
        
        p = torch.tensor(inp_types, device=x.device).long()
        l = torch.tensor(log_return, device=x.device).float().view(-1, 1)

        x = x.expand(p.size(0), x.size(1))
        z = z.expand(p.size(0), z.size(1))

        p = self._embedding(p)
        x = torch.cat((x, p, l, z), dim=-1)
        
        x = self._f1(x)
        x = torch.relu(self._n1(x))
        x = self._d(x)
        
        x = self._f2(x)
        x = torch.relu(self._n2(x))
        x = self._d(x)

        x = torch.softmax(self._f3(x), dim=-1)
        return x
import torch
import torch.nn as nn

from typing import List
from torch_geometric.data import Data
from torch_geometric.nn import (
    SAGEConv, 
    GATConv,
    LayerNorm, 
    SoftmaxAggregation, 
    PowerMeanAggregation,
    MemPooling,
    global_mean_pool
)


class MarketGraphNet(nn.Module):
    def __init__(
            self, 
            inp_dim: int, 
            out_dim: int,
            dropout: float=0.3) -> None:
        super().__init__()

        self._c1 = SAGEConv(
            in_channels  = inp_dim, 
            out_channels = 256, 
            bias         = True,
            normalize    = True,
            project      = True,
            aggr         = SoftmaxAggregation(
                channels = inp_dim, 
                learn    = True))
        
        self._c2 = SAGEConv(
            in_channels  = 256, 
            out_channels = 128, 
            bias         = True,
            normalize    = True,
            project      = True,
            aggr         = SoftmaxAggregation(
                channels = 256, 
                learn    = True))

        self._dp = nn.Dropout(p=dropout)
        self._px = MemPooling(128, 128, 4, 4)
        self._fx = nn.Linear(128, out_dim, bias=True)
        self._nx = nn.LayerNorm(out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        x = self._c1(x, edge_index)
        x = torch.relu(x)
        x = self._dp(x)

        x = self._c2(x, edge_index)
        x = torch.relu(x)
        x = self._dp(x)
        
        x, _ = self._px(x)
        x = global_mean_pool(x.squeeze(0), None)
        x = self._dp(x)

        x = self._nx(self._fx(x))
        x = torch.relu(x)
        x = self._dp(x)
        return x


class MultiHeadMemory(nn.Module):
    def __init__(
            self,
            heads:    int,
            mem_size: int,
            mem_dim:  int,
            key_dim:  int,
            val_dim:  int,
            dropout:  float=0.3) -> None:
        super().__init__()

        self._mem = nn.Parameter(torch.randn((heads, mem_size, mem_dim)).float())
        self._fk  = nn.Linear(mem_dim, key_dim, bias=True)
        self._fv  = nn.Linear(mem_dim, val_dim, bias=True)
        self._fx  = nn.Linear(val_dim * heads, val_dim, bias=True)
        self._nk  = nn.LayerNorm(key_dim)
        self._nv  = nn.LayerNorm(val_dim)
        self._nx  = nn.LayerNorm(val_dim)
        self._dp  = nn.Dropout(p=dropout)
    
    def forward(self, q: torch.Tensor) -> torch.Tensor:
        '''
        q: [b x k] Query tensor
        '''
        k = self._nk(self._fk(self._mem))      # [h x n x k]
        k = torch.softmax(k, dim=-1)           # Memory keys

        v = self._fv(self._mem)                # [h x n x v]
        v = torch.relu(self._nv(v))
        v = self._dp(v)

        q = q.unsqueeze(1)                     # [b x 1 x k]
        a = torch.einsum('bqk,hnk->bhn', q, k) # [b x h x n]
        w = torch.softmax(a, dim=-1)           # Attention weights
        v = torch.einsum('bhn,hnv->bhv', w, v) # [b x h x v]
        v = self._dp(v)

        x = v.reshape(v.size(0), -1)           # [b x (h * v)]
        x = self._fx(x)                        # Queried memory
        x = self._nx(x)
        x = torch.relu(x)
        x = self._dp(x)
        return x


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
            val_dim:   int,
            dropout:   float=0.3) -> None:
        super().__init__()

        self._emb = nn.Embedding(inp_types, emb_dim)
        self._gnn = MarketGraphNet(inp_dim, val_dim)
        cat_dim   = inp_dim + emb_dim + val_dim + 1
        self._f1  = nn.Linear(2 * val_dim, 512, bias=True)
        self._f2  = nn.Linear(512, 256, bias=True)
        self._f3  = nn.Linear(256, out_dim, bias=True)

        self._fk  = nn.Linear(cat_dim, key_dim, bias=True)
        self._fv  = nn.Linear(cat_dim, val_dim, bias=True)

        self._mem = MultiHeadMemory(mem_heads, mem_size, mem_dim, key_dim, val_dim, dropout)
        self._n1  = nn.LayerNorm(512)
        self._n2  = nn.LayerNorm(256)
        # self._n3  = nn.LayerNorm(out_dim)

        self._nk  = nn.LayerNorm(key_dim)
        self._nv  = nn.LayerNorm(val_dim)
        self._dp  = nn.Dropout(p=dropout)

    def forward(
            self,
            data:       Data,       # Market graph data
            index:      int,        # Node index of the position of the ticker
            inp_types:  List[int],  # Type of input, trade or market
            log_return: List[float] # Log return of the input
    ) -> torch.Tensor:
        '''
        Forward pass for the PolicyNet. Data contains the features from market data in graph format.

        Parameters:
        data       (torch_geometric.data.Data): Input features tensor.
        index      (int):                       Node feature index.
        inp_type   (List[int]):                 Type of input, trade or market.
        log_return (List[float]):               Log return of the input.
        
        Returns:
        torch.Tensor: Output tensor after passing through the network.
        '''
        p = torch.tensor(inp_types, device=data.x.device).long()
        l = torch.tensor(log_return, device=data.x.device).float().view(-1, 1)
        p = self._emb(p)
        x = data.x[None, index, :]             # Index the feature, [1 x inp]
        x = x.expand(p.size(0), x.size(1))

        z = self._gnn(data)                    # Computes the market representation tensor, [1 x v]
        z = z.expand(p.size(0), z.size(1))
        x = torch.cat((x, p, l, z), dim=-1)    # [n x (inp_dim + emb_dim + 1 + gnn_dim)]
        x = self._dp(x)

        k = self._nk(self._fk(x))              # Computes the market key tensor, [n x k]
        k = torch.softmax(k, dim=-1)
        v = self._nv(self._fv(x))              # Computes the market value tensor, [n x v]
        v = torch.relu(v)
        v = self._dp(v)

        m = self._mem(k)                       # Queries the memory, [n x v]
        x = torch.cat((v, m), dim=-1)          # Concats the market value and memory, [n x 2v]
        x = self._dp(x)

        x = self._f1(x)
        x = torch.relu(self._n1(x))
        x = self._dp(x)

        x = self._f2(x)
        x = torch.relu(self._n2(x))
        x = self._dp(x)

        x = self._f3(x)
        return x
        # x = self._n3(x)
        # return torch.softmax(x, dim=-1)


class CriticNet(nn.Module):
    def __init__(
            self, 
            inp_dim:   int, 
            inp_types: int,
            emb_dim:   int,
            val_dim:   int,
            mem_heads: int,
            mem_size:  int, 
            mem_dim:   int,
            key_dim:   int,
            dropout:   float=0.3) -> None:
        super().__init__()

        self._emb = nn.Embedding(inp_types, emb_dim)
        self._gnn = MarketGraphNet(inp_dim, val_dim)
        cat_dim   = inp_dim + emb_dim + val_dim + 1
        self._f1  = nn.Linear(2 * val_dim, 512, bias=True)
        self._f2  = nn.Linear(512, 256, bias=True)
        self._f3  = nn.Linear(256, 1, bias=True)

        self._fk  = nn.Linear(cat_dim, key_dim, bias=True)
        self._fv  = nn.Linear(cat_dim, val_dim, bias=True)

        self._mem = MultiHeadMemory(mem_heads, mem_size, mem_dim, key_dim, val_dim, dropout)
        self._n1  = nn.LayerNorm(512)
        self._n2  = nn.LayerNorm(256)

        self._nk  = nn.LayerNorm(key_dim)
        self._nv  = nn.LayerNorm(val_dim)
        self._dp  = nn.Dropout(p=dropout)

    def forward(
            self,
            data:       Data,       # Market graph data
            index:      int,        # Node index of the position of the ticker
            inp_types:  List[int],  # Type of input, trade or market
            log_return: List[float] # Log return of the input
    ) -> torch.Tensor:
        
        p = torch.tensor(inp_types, device=data.x.device).long()
        l = torch.tensor(log_return, device=data.x.device).float().view(-1, 1)
        p = self._emb(p)
        x = data.x[None, index, :]             # Index the feature, [1 x inp]
        x = x.expand(p.size(0), x.size(1))

        z = self._gnn(data)                    # Computes the market representation tensor, [1 x v]
        z = z.expand(p.size(0), z.size(1))
        x = torch.cat((x, p, l, z), dim=-1)    # [n x (inp_dim + emb_dim + 1 + gnn_dim)]
        x = self._dp(x)

        k = self._nk(self._fk(x))              # Computes the market key tensor, [n x k]
        k = torch.softmax(k, dim=-1)
        v = self._nv(self._fv(x))              # Computes the market value tensor, [n x v]
        v = torch.relu(v)
        v = self._dp(v)

        m = self._mem(k)                       # Queries the memory, [n x v]
        x = torch.cat((v, m), dim=-1)          # Concats the market value and memory, [n x 2v]
        x = self._dp(x)

        x = self._f1(x)
        x = torch.relu(self._n1(x))
        x = self._dp(x)

        x = self._f2(x)
        x = torch.relu(self._n2(x))
        x = self._dp(x)

        x = self._f3(x)
        return x


import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import GATv2Conv, Linear


class GATNet(nn.Module):
    def __init__(
        self,
        h1_dim=512,
        h2_dim=256,
        out_dim=128,
    ) -> None:
        super().__init__()

        self.conv1 = GATv2Conv(
            in_channels=(-1, -1),
            out_channels=h1_dim,
            add_self_loops=False,
            edge_dim=-1,
        )
        self.conv2 = GATv2Conv(
            in_channels=(-1, -1),
            out_channels=h2_dim,
            add_self_loops=False,
            edge_dim=-1,
        )
        self.conv3 = GATv2Conv(
            in_channels=(-1, -1),
            out_channels=out_dim,
            add_self_loops=False,
            edge_dim=-1,
        )

        self.lin1 = Linear(-1, h1_dim)
        self.lin2 = Linear(-1, h2_dim)
        self.lin3 = Linear(-1, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = self.conv1(x, edge_index) + self.lin1(x)
        x = x.relu()
        x = self.conv2(x, edge_index) + self.lin2(x)
        x = x.relu()
        x = self.conv3(x, edge_index) + self.lin3(x)
        return x

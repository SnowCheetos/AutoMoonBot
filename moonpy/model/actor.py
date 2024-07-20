import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import to_hetero
from torch_geometric.data import HeteroData

from moonpy.model import SAGENet, MemoryClassifier


class Actor(nn.Module):
    def __init__(self, metadata, **kwargs) -> None:
        super().__init__()

        self.sage = SAGENet(**kwargs)  # placeholder params
        self.sage = to_hetero(self.sage, metadata, aggr="sum")
        self.pos_clf = MemoryClassifier(**kwargs)  # placeholder params
        self.ast_clf = MemoryClassifier(**kwargs)  # placeholder params

    def forward(self, data: HeteroData) -> Tensor:
        x = self.sage(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        pos = x["position"]
        ast = x["equity"] # And more, placeholder for now

        # TODO continue

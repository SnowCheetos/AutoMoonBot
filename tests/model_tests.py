import pytest
import torch

from torch_geometric.data import Data
from reinforce.model import PolicyNet
from reinforce.utils import select_action


INP_DIM = 10
OUT_DIM = 10
INP_TPS = 10
EMB_DIM = 10
MEM_HEA = 10
MEM_SIZ = 10
MEM_DIM = 10
KEY_DIM = 10
VAL_DIM = 10
NUM_NDS = 10
DEVICE  = "cpu"

@pytest.fixture
def model():
    return PolicyNet(
        inp_dim   = INP_DIM,
        out_dim   = OUT_DIM,
        inp_types = INP_TPS,
        emb_dim   = EMB_DIM,
        mem_heads = MEM_HEA,
        mem_size  = MEM_SIZ,
        mem_dim   = MEM_DIM,
        key_dim   = KEY_DIM,
        val_dim   = VAL_DIM)

def test_output_shape(model: PolicyNet):
    x = torch.rand((NUM_NDS, INP_DIM))
    a = torch.diag(torch.ones(NUM_NDS-1), 1) + torch.diag(torch.ones(NUM_NDS-2), 2) + torch.eye(NUM_NDS)
    
    data = Data(
        x = x,
        edge_index = torch.nonzero(a).long().t().contiguous()
    )

    output = model(data, 0, [0, 1, 2, 3], [0.1, 0.2, 0.3, 0.4])

    assert output.size(0) == 4, "output contained wrong number of rows"
    assert output.size(1) == OUT_DIM, "output contained wrong number of columns"

    output = model(data, 0, [0, 1, 2, 3, 5], [0.1, 0.2, 0.3, 0.4, 0.5])

    assert output.size(0) == 5, "output contained wrong number of rows"
    assert output.size(1) == OUT_DIM, "output contained wrong number of columns"

    output = model(data, 0, [0], [0.1])

    assert output.size(0) == 1, "output contained wrong number of rows"
    assert output.size(1) == OUT_DIM, "output contained wrong number of columns"

def test_select_action(model: PolicyNet):
    x = torch.rand((NUM_NDS, INP_DIM))
    a = torch.diag(torch.ones(NUM_NDS-1), 1) + torch.diag(torch.ones(NUM_NDS-2), 2) + torch.eye(NUM_NDS)
    
    data = Data(
        x = x,
        edge_index = torch.nonzero(a).long().t().contiguous()
    )

    actions, log_probs = select_action(
        model      = model, 
        state      = data, 
        index      = 0, 
        inp_types  = [0, 1, 2], 
        log_return = [0.1, 0.2, 0.3],
        device     = DEVICE)

    assert len(actions) == 3, "actions contained wrong length"
    assert log_probs.size(0) == 3, "log probabilities contained wrong number of rows"
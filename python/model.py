import torch
import torch.nn as nn
import numpy as np

from typing import Tuple


class PolicyNet(nn.Module):
    def __init__(
            self, 
            input_dim: int,
            output_dim: int, 
            embedding_dim: int) -> None:
        super().__init__()

        self._embedding = nn.Embedding(output_dim, embedding_dim)
        self._f1 = nn.Linear(input_dim + embedding_dim, 256)
        self._f2 = nn.Linear(256, 128)
        self._f3 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, self._embedding(a).squeeze(1)), dim=-1)
        x = torch.relu(self._f1(x))
        x = torch.relu(self._f2(x))
        return torch.softmax(self._f3(x), dim=-1)


def select_action(
        model: nn.Module, 
        state: np.ndarray, 
        action: int, 
        device: str
) -> Tuple[int, torch.Tensor]:
    
    probs = model(
        torch.tensor(
            state,
            dtype=torch.float32,
            device=device),
        torch.tensor(
            [[action]], 
            dtype=torch.long, 
            device=device))
    
    action = np.random.choice(probs.size(-1), p=probs.detach().numpy()[0])
    return action, torch.log(probs[0, action])

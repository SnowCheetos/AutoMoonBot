import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, List


class PolicyNet(nn.Module):
    def __init__(
            self, 
            input_dim:     int,
            output_dim:    int, 
            holding_dim:   int,
            embedding_dim: int) -> None:
        
        super().__init__()

        self._embedding = nn.Embedding(holding_dim, embedding_dim)
        self._f1 = nn.Linear(input_dim + embedding_dim, 512)
        self._f2 = nn.Linear(512, 256)
        self._f3 = nn.Linear(256, 128)
        self._f4 = nn.Linear(128, output_dim)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        x = torch.cat((x, self._embedding(p).squeeze(1)), dim=-1)
        x = torch.relu(self._f1(x))
        x = torch.relu(self._f2(x))
        x = torch.relu(self._f3(x))
        return torch.softmax(self._f4(x), dim=-1)

def select_action(
        model:    nn.Module, 
        state:    np.ndarray, 
        position: int, 
        device:   str) -> Tuple[int, torch.Tensor]:
    
    probs = model(
        torch.tensor(
            state,
            dtype=torch.float32,
            device=device),
        torch.tensor(
            [[position]], 
            dtype=torch.long, 
            device=device))
    
    action = np.random.choice(probs.size(-1), p=probs.detach().cpu().numpy()[0])
    return action, torch.log(probs[0, action])

def compute_discounted_rewards(
        rewards: List[float], 
        gamma:   float) -> List[float]:
    
    discounted_rewards = []
    cumulative = 0
    for reward in reversed(rewards):
        cumulative = reward + gamma * cumulative
        discounted_rewards.insert(0, cumulative)
    return discounted_rewards

def compute_loss(
        log_probs: List[torch.Tensor], 
        rewards:   List[float],
        gamma:     float=0.99,
        device:    str="cpu") -> torch.Tensor:
    
    log_probs = torch.stack(log_probs)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

    discounted_rewards = compute_discounted_rewards(rewards.tolist(), gamma)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)
    normalized_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    policy_gradient = -log_probs * normalized_rewards
    return policy_gradient.sum()

def inference(
        model:    nn.Module,
        state:    np.ndarray,
        position: int,
        device:   str="cpu") -> int:
    
    model.eval()
    probs = model(
        torch.tensor(
            state,
            dtype=torch.float32,
            device=device),
        torch.tensor(
            [[position]], 
            dtype=torch.long, 
            device=device))
    
    return probs.argmax(1)
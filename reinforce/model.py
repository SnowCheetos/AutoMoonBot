import logging
import math
import torch
import torch.nn as nn
import numpy as np

from typing import Tuple, List

class MemoryNet(nn.Module):
    def __init__(self, num_mem: int, mem_dim: int, out_dim: int) -> None:
        super().__init__()
        self._mem = nn.Parameter(torch.randn((num_mem, mem_dim), dtype=torch.float32))
        self._fk  = nn.Linear(mem_dim, out_dim)
        self._fv  = nn.Linear(mem_dim, out_dim)
        
        # Initialize parameters
        nn.init.kaiming_uniform_(self._fk.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self._fv.weight, a=math.sqrt(5))

    def forward(self, k: torch.Tensor) -> torch.Tensor:
        """
        k: [n x d]
        """
        key = torch.softmax(self._fk(self._mem), dim=-1) # [h x d]
        val = self._fv(self._mem)                        # [h x d]

        # Compute attention scores
        att = torch.matmul(k, key.t()) # [n x h]

        # Apply softmax to get attention weights
        att_weights = torch.softmax(att, dim=-1) # [n x h]

        # Compute the weighted sum of the values
        output = torch.matmul(att_weights, val) # [n x d]

        return output


class MultiHeadMemory(nn.Module):
    def __init__(self, num_heads: int, num_mem: int, mem_dim: int, out_dim: int) -> None:
        super().__init__()

        self._mems = [MemoryNet(num_mem, mem_dim, out_dim) for _ in range(num_heads)]
        self._f    = nn.Linear(num_heads * out_dim, out_dim)

    def forward(self, k):
        outputs = []
        for m in self._mems:
            outputs += [m(k)]
        
        x = torch.cat(outputs, dim=-1)

        return self._f(x)


class PolicyNet(nn.Module):
    def __init__(
            self, 
            input_dim:     int, 
            output_dim:    int, 
            position_dim:  int,
            embedding_dim: int, 
            num_mem:       int = 512, 
            mem_dim:       int = 256) -> None:
        super().__init__()

        self._embedding = nn.Embedding(position_dim, embedding_dim)
        self._f1        = nn.Linear(input_dim + embedding_dim + 1, 512)
        self._f2        = nn.Linear(512, 256)
        self._fk        = nn.Linear(256, 128)
        self._fv        = nn.Linear(256, 128)
        self._f4        = nn.Linear(256, output_dim)
        self._mem       = MultiHeadMemory(4, num_mem, mem_dim, 128)
        
        self._d  = nn.Dropout(p=0.5)
        self._n1 = nn.LayerNorm(512)
        self._n2 = nn.LayerNorm(256)
        self._n3 = nn.LayerNorm(256)

    def forward(self, x: torch.Tensor, p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the PolicyNet.

        Parameters:
        x (torch.Tensor): Input features tensor.
        p (torch.Tensor): Position indices tensor.
        g (torch.Tensor): Portfolio features tensor.
        
        Returns:
        torch.Tensor: Output tensor after passing through the network.
        """
        x = torch.cat((x, self._embedding(p).squeeze(1), g), dim=-1)
        x = self._n1(torch.relu(self._f1(x)))
        x = self._d(x)
        x = self._n2(torch.relu(self._f2(x)))
        x = self._d(x)
        
        k = torch.softmax(self._fk(x), dim=-1)
        v = self._fv(x)
        
        q = self._mem(k)
        x = torch.cat((v, q), dim=-1)
        x = self._n3(x)
        x = self._d(x)

        return torch.softmax(self._f4(x), dim=-1)

def select_action(
        model:     nn.Module, 
        state:     np.ndarray, 
        potential: float,
        position:  int, 
        device:    str) -> Tuple[int, torch.Tensor]:
    
    probs = model(
        x=torch.tensor(
            state,
            dtype=torch.float32,
            device=device),
        p=torch.tensor(
            [[position]], 
            dtype=torch.long, 
            device=device),
        g=torch.tensor(
            [[potential]],
            dtype=torch.float32, 
            device=device
        ))

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
        log_probs:  List[torch.Tensor], 
        rewards:    List[float],
        gamma:      float=0.99,
        log_return: float | None=None,
        beta:       float | None=0.5,
        device:     str="cpu") -> torch.Tensor:
    
    log_probs = torch.stack(log_probs)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

    discounted_rewards = compute_discounted_rewards(rewards.tolist(), gamma)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)
    normalized_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    if log_return is not None and beta is not None:
        policy_gradient = -log_probs * (beta * normalized_rewards + (1-beta) * log_return)
    else:
        policy_gradient = -log_probs * normalized_rewards
    
    return policy_gradient.sum()

def inference(
        model:     nn.Module,
        state:     np.ndarray,
        position:  int,
        potential: float,
        device:    str="cpu",
        method:    str="argmax",
        min_prob:  float=0.31) -> Tuple[int, float]:
    
    model.eval()
    min_prob = min(0.34, min_prob)

    if method not in {"argmax", "prob"}:
        logging.error("Method must be in one of [argmax, prob]")
        return 1

    with torch.no_grad():
        probs = model(
            x=torch.tensor(
                state,
                dtype=torch.float32,
                device=device),
            p=torch.tensor(
                [[position]], 
                dtype=torch.long, 
                device=device),
            g=torch.tensor(
                [[potential]],
                dtype=torch.float32, 
                device=device
            ))
    
    if method == "argmax":
        action = probs.argmax(1).item()
    else:
        action = np.random.choice(probs.size(-1), p=probs.detach().cpu().numpy()[0])
        
    prob = probs[0, action].item()
    if method == "prob" and prob < min_prob:
        action = probs.argmax(1).item()
        prob = probs[0, action].item()

    return action, prob
import torch
import logging
import numpy as np
from torch_geometric.data import Data
from typing import List, Tuple
from reinforce.model import PolicyNet

def select_action(
        model:      PolicyNet, 
        state:      Data, 
        index:      int,
        inp_types:  List[int],
        log_return: List[float], 
        device:     str,
        argmax:     bool = False) -> Tuple[List[int], torch.Tensor]:

    probs = model(
        data       = state.to(device),
        index      = index,
        inp_types  = inp_types,
        log_return = log_return)

    actions   = []
    log_probs = []
    if argmax:
        for i in range(probs.size(0)):
            prob_np = probs[i].detach().cpu().numpy()
            action = prob_np.argmax(-1)
            actions.append(action)
            log_probs.append(torch.log(probs[i, action]))
    else:
        for i in range(probs.size(0)):
            prob_np = probs[i].detach().cpu().numpy()
            action = np.random.choice(probs.size(-1), p=prob_np)
            actions.append(action)
            log_probs.append(torch.log(probs[i, action]))

    log_probs = torch.stack(log_probs)
    return actions, log_probs

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
        sharpe:    float,
        gamma:     float=0.99,
        device:    str="cpu") -> torch.Tensor:

    log_probs          = torch.vstack(log_probs)
    discounted_rewards = compute_discounted_rewards(rewards, gamma)
    discounted_rewards = torch.tensor(discounted_rewards, device=device).float()
    # normalized_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    policy_gradient = -log_probs * discounted_rewards.sum() #normalized_rewards
    loss = policy_gradient.sum()

    return loss
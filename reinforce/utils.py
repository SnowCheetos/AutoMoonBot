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
        log_probs:  List[torch.Tensor], 
        rewards:    List[float],
        gamma:      float=0.99,
        device:     str="cpu") -> torch.Tensor:
    
    log_probs = torch.stack(log_probs)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)

    discounted_rewards = compute_discounted_rewards(rewards.tolist(), gamma)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=device)
    # normalized_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

    policy_gradient = -log_probs * discounted_rewards
    return policy_gradient.sum()

# TODO Fix this
def inference(
        model:     PolicyNet, 
        state:     Data, 
        index:     int,
        potential: float,
        position:  int, 
        device:    str,
        method:    str="argmax",
        min_prob:  float=0.31) -> Tuple[int, float] | None:
    
    model.eval()
    min_prob = min(0.34, min_prob)

    if method not in {"argmax", "prob"}:
        logging.error("Method must be in one of [argmax, prob]")
        return None

    with torch.no_grad():
        probs = model(
            data  = state.to(device),
            index = index,
            pos   = torch.tensor(
                        [[position]], 
                        dtype=torch.long, 
                        device=device),
            port  = torch.tensor(
                        [[potential]],
                        dtype=torch.float32, 
                        device=device))
    
    if method == "argmax":
        action = probs.argmax(1).item()
    else:
        action = np.random.choice(probs.size(-1), p=probs.detach().cpu().numpy()[0])
        
    prob = probs[0, action].item()
    if method == "prob" and prob < min_prob:
        action = probs.argmax(1).item()
        prob = probs[0, action].item()

    return (action, prob)
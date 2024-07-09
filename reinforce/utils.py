import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import List, Tuple
from reinforce.model import PolicyNet

def select_action(
        model:       PolicyNet, 
        state:       Data, 
        index:       int,
        inp_types:   List[int],
        log_return:  List[float], 
        device:      str,
        temperature: float = 2.0,
        argmax:      bool = False) -> Tuple[List[int], torch.Tensor]:

    pred = model(
        data       = state.to(device),
        index      = index,
        inp_types  = inp_types,
        log_return = log_return)
    probs = F.softmax(pred / temperature, dim=-1)

    if argmax:
        print(probs)

    actions   = []
    log_probs = []
    if argmax:
        actions = torch.argmax(probs, dim=-1)
        log_probs = torch.log(probs[range(probs.size(0)), actions])
    else:
        action_dist = torch.distributions.Categorical(probs)
        actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions)

    entropy = (-torch.log(probs).exp() * probs).sum(dim=-1).view(-1, 1)
    actions = actions.detach().tolist()
    return actions, log_probs, entropy

def compute_discounted_rewards(
        rewards: List[float], 
        gamma:   float) -> List[float]:
    
    discounted_rewards = []
    cumulative = 0
    for reward in reversed(rewards):
        cumulative = reward + gamma * cumulative
        discounted_rewards.insert(0, cumulative)
    return discounted_rewards

def compute_policy_loss(
        rewards:   List[float],
        log_probs: List[torch.Tensor], 
        entropy:   List[torch.Tensor],
        values:    List[torch.Tensor] | None = None,
        sharpe:    float | None = None,
        beta:      float = 0.5,
        gamma:     float = 0.99,
        device:    str   = "cpu") -> torch.Tensor:

    log_probs          = torch.vstack(log_probs)
    discounted_rewards = compute_discounted_rewards(rewards, gamma)
    discounted_rewards = torch.tensor(discounted_rewards, device=device).float()
    normalized_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-6)

    if values is not None:
        values     = torch.vstack(values).detach()
        advantages = normalized_rewards - values
    else:
        advantages = normalized_rewards

    policy_gradient = (-log_probs * advantages).sum()
    return policy_gradient + beta * torch.vstack(entropy).mean()

def compute_critic_loss(
        rewards:   List[float],
        values:    List[torch.Tensor] | None = None,
        gamma:     float = 0.99,
        device:    str   = "cpu") -> torch.Tensor:
    
    values             = torch.vstack(values)
    discounted_rewards = compute_discounted_rewards(rewards, gamma)
    discounted_rewards = torch.tensor(discounted_rewards, device=device).float()
    normalized_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-6)

    loss = F.mse_loss(values.view(-1,), normalized_rewards, reduction='sum')
    return loss
import logging
import numpy as np
import gymnasium as gym
import torch.optim as optim

from gymnasium import spaces
from typing import Dict, List

from reinforce.sampler import DataSampler
from reinforce.model import PolicyNet, select_action, compute_loss
from reinforce.utils import Position, compute_sharpe_ratio


class TradeEnv(gym.Env):
    def __init__(
            self, 
            state_dim: int, 
            action_dim: int, 
            embedding_dim: int,
            queue_size: int,
            device: str,
            db_path: str,
            return_thresh: float,
            feature_params: Dict[str, List[int] | Dict[str, List[int]]]={
                "sma": np.geomspace(8, 64, 8).astype(int).tolist(),
                "ema": np.geomspace(8, 64, 8).astype(int).tolist(),
                "rsi": np.geomspace(4, 64, 8).astype(int).tolist(),
                "sto": {
                    "window": np.geomspace(8, 64, 4).astype(int).tolist() + np.geomspace(8, 64, 4).astype(int).tolist(),
                    "k": np.linspace(3, 11, 8).astype(int).tolist(),
                    "d": np.linspace(3, 11, 8).astype(int).tolist()
                }
            }) -> None:
        super().__init__()

        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = spaces.Discrete(state_dim)

        self._device = device
        self._sampler = DataSampler(db_path, queue_size, feature_params=feature_params)
        self._policy_net = PolicyNet(state_dim, action_dim, len(Position), embedding_dim).to(device)
        self._return_thresh = return_thresh
        self._position = Position.Cash
        self._portfolio = 1.0
        self._returns = []
        self._log_probs = []
        self._init_close = 0.0
        self._risk_free_rate = 1.0
        self._entry = 0.0
        self._exit = 0.0

    @property
    def model(self):
        return self._policy_net

    @property
    def model_weights(self):
        return self._policy_net.state_dict()

    @property
    def log_probs(self):
        return self._log_probs

    def reset(self):
        self._sampler.reset()
        _, close, state = self._sampler.sample_next()

        while len(state) == 0:
            _, close, state = self._sampler.sample_next()

        self._position = Position.Cash
        self._portfolio = 1.0
        self._returns = []
        self._log_probs = []
        self._init_close = close
        self._risk_free_rate = 1.0
        self._entry = 0.0
        self._exit = 0.0
        return state

    def step(self, action: int):
        end, close, state = self._sampler.sample_next()
        reward, done = 0.0, False

        # Valid buy
        if self._position == Position.Cash and action == 0:
            self._position = Position.Asset
            self._entry = close
            self._exit = 0.0
        
        # Valid sell
        elif self._position == Position.Asset and action == 2:
            self._position = Position.Cash
            self._exit = close
            self._returns += [close / self._entry]
            self._portfolio *= close / self._entry
            if self._portfolio < self._return_thresh:
                done = True
            
            self._risk_free_rate = close / self._init_close
            self._entry = 0.0
            reward += compute_sharpe_ratio(self._returns, self._risk_free_rate)

        action, log_prob = select_action(
            self._policy_net, 
            state, 
            int(self._position.value), 
            self._device)
        
        self._log_probs += [log_prob]

        if end: done = True

        return action, reward, done, False, {}

def train(env: TradeEnv, episodes: int, learning_rate: float=1e-3) -> List[float]:
    logging.info("Training starts")
    env.reset()
    optimizer = optim.SGD(env.model.parameters(), lr=learning_rate)

    reward_history = []
    for e in range(episodes):
        logging.info(f"Episode {e+1}/{episodes} began")
        action, reward, done, _, _ = env.step(1)
        rewards = [reward]
        while not done:
            action, reward, done, _, _ = env.step(action)
            rewards += [reward]

        optimizer.zero_grad()
        loss = compute_loss(env.log_probs, rewards)
        loss.backward()
        optimizer.step()

        reward_history += [sum(rewards)]
        env.reset()
        logging.info(f"Episode {e+1}/{episodes} done, sum reward: {reward_history[-1]:.5f}")

    return reward_history

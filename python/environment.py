import numpy as np
import gymnasium as gym
from gymnasium import spaces

from python.sampler import DataSampler


class TradeEnv(gym.Env):
    def __init__(
            self, 
            state_dim: int, 
            action_dim: int, 
            db_path: str,
            reward_thresh: float) -> None:
        super().__init__()

        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = spaces.Discrete(state_dim)

        self._sampler = DataSampler(db_path)
        self._reward_thresh = reward_thresh

    def reset(self) -> np.ndarray:
        self._sampler.reset()
        return self._sampler.sample_next()

    def step(self, action):
        pass
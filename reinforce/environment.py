import gymnasium as gym
from gymnasium import spaces

from reinforce.sampler import DataSampler
from reinforce.model import PolicyNet, select_action
from reinforce.utils import Position, compute_sharpe_ratio


class TradeEnv(gym.Env):
    def __init__(
            self, 
            state_dim: int, 
            action_dim: int, 
            embedding_dim: int,
            device: str,
            db_path: str,
            return_thresh: float) -> None:
        super().__init__()

        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = spaces.Discrete(state_dim)

        self._device = device
        self._sampler = DataSampler(db_path)
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
    def model_weights(self):
        return self._policy_net.state_dict()

    @property
    def log_probs(self):
        return self._log_probs

    def reset(self):
        self._sampler.reset()
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

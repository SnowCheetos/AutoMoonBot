import logging
import numpy as np
import gymnasium as gym
import torch.optim as optim
import torch.nn as nn

from gymnasium import spaces
from collections import deque
from typing import Dict, List, Any, Optional

from reinforce.sampler import DataSampler
from reinforce.model import PolicyNet, select_action, compute_loss
from utils.trading import Position, Action, Status
from utils.descriptors import compute_sharpe_ratio
from backend.manager import TradeManager


class TradeEnv(gym.Env):
    def __init__(
            self, 
            state_dim:         int, 
            action_dim:        int, 
            embedding_dim:     int,
            queue_size:        int,
            inaction_cost:     float,
            action_cost:       float,
            device:            str,
            db_path:           str,
            return_thresh:     float,
            sharpe_cutoff:     int=30,
            gamma:             float=0.15,
            alpha:             float=1.5,
            feature_params:    Dict[str, List[int] | Dict[str, List[int]]] | None=None,
            beta:              float | None=0.5,
            logger:            Optional[logging.Logger]=None,
            testing:           bool=False,
            max_training_data: int | None=None) -> None:
        
        super().__init__()

        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

        self.action_space = spaces.Discrete(action_dim)
        self.observation_space = spaces.Discrete(state_dim)

        db_max_access = None if not testing else queue_size+2

        self._sampler         = DataSampler(
            db_path           = db_path, 
            queue_size        = queue_size, 
            max_access        = db_max_access, 
            feature_params    = feature_params,
            max_training_data = max_training_data)
        
        self._policy_net   = PolicyNet(
            input_dim      = state_dim, 
            output_dim     = action_dim, 
            position_dim   = len(Position), 
            embedding_dim  = embedding_dim).to(device)

        self._manager = TradeManager(0, alpha, gamma, action_cost)

        self._status         = Status(0, alpha)
        self._device         = device
        self._sharpe_cutoff  = sharpe_cutoff
        self._return_thresh  = return_thresh
        self._position       = Position.Cash
        self._inaction_cost  = inaction_cost
        self._action_cost    = action_cost
        self._alpha          = alpha
        self._beta           = beta
        self._gamma          = gamma
        self._portfolio      = 1.0
        self._returns        = []
        self._log_probs      = []
        self._init_close     = 0.0
        self._risk_free_rate = 1.0
        self._entry          = 0.0
        self._exit           = 0.0
        self._log_return     = 0.0

    @property
    def beta(self):
        return self._beta

    @property
    def sampler(self):
        return self._sampler

    @property
    def model(self):
        return self._policy_net

    @property
    def model_weights(self) -> Dict[str, Any]:
        self._policy_net.eval()
        return self._policy_net.state_dict()

    @property
    def log_return(self) -> float:
        return self._log_return

    @model_weights.setter
    def model_weights(self, state_dict: Dict[str, Any]) -> None:
        self._policy_net.eval()
        self._policy_net.to("cpu")
        self._policy_net.load_state_dict(state_dict)
        self._policy_net.to(self._device)

    @property
    def log_probs(self) -> list:
        return self._log_probs
    
    @property
    def portfolio(self) -> float:
        return self._portfolio
    
    @property
    def init_close(self) -> float:
        return self._init_close

    def reset(self):
        self._sampler.reset()
        _, close, state = self._sampler.sample_next()

        while len(state) == 0:
            _, close, state = self._sampler.sample_next()

        self._manager = TradeManager(
            cov       = self.sampler.coef_of_var, 
            alpha     = self._alpha, 
            gamma     = self._gamma, 
            cost      = self._action_cost)
        
        self._position       = Position.Cash
        self._portfolio      = 1.0
        self._returns        = []
        self._log_probs      = []
        self._init_close     = close
        self._risk_free_rate = 1.0
        self._entry          = 0.0
        self._exit           = 0.0
        self._log_return     = 0.0
        return state

    def step(self, action: int):
        end, price, state = self._sampler.sample_next()
        reward, done = 0, False

        self._risk_free_rate = price / self._init_close
        # Validate buy
        if Action(action) == Action.Buy:
            _ = self._manager.try_buy(price, self.sampler.coef_of_var)
        
        elif Action(action) == Action.Sell:
            gain = self._manager.try_sell(price, self.sampler.coef_of_var)
            if gain > 0:
                if self._manager.portfolio < self._return_thresh:
                    self._logger.info("portfolio threshold hit, episode done")
                    done = True
                
                self._log_return  += np.log(gain)
                sharpe = compute_sharpe_ratio(
                    returns        = self._manager.returns[-self._sharpe_cutoff:], 
                    risk_free_rate = self._risk_free_rate * (1 - self._action_cost))
                
                reward += sharpe + gain

        elif Action(action) == Action.Hold:
            if self._manager.asset or self._manager.partial:
                potential_gain = self._manager.potential_gain(price)
                reward        -= self._inaction_cost * potential_gain * (1 - self._action_cost)
            else:
                if self._manager.cash:
                    prev_exit = self._manager.prev_exit
                    prev_exit = prev_exit if prev_exit > 0 else price
                    reward   -= self._inaction_cost * (price / prev_exit)
                else:
                    reward   -= self._inaction_cost * self._risk_free_rate * (1 - self._action_cost)

        action, log_prob = select_action(
            model     = self._policy_net, 
            state     = state, 
            potential = self._manager.potential_gain(price) - 1,
            position  = int(self._position.value), 
            device    = self._device)
        
        self._log_probs += [log_prob]

        if end: done = True

        return action, reward, done, False, {"price": price}

def train(
        env:            TradeEnv, 
        episodes:       int, 
        learning_rate:  float=1e-3, 
        momentum:       float=0.9,
        weight_decay:   float=0.9,
        max_grad_norm:  float=1.0,
        portfolio_size: int=5) -> float:
    
    env._logger.info("training starts")
    optimizer = optim.SGD(
        env.model.parameters(), 
        lr=learning_rate, 
        momentum=momentum,
        weight_decay=weight_decay)

    buy_and_hold = None
    reward_history = []
    portfolios = deque(maxlen=portfolio_size)

    for e in range(episodes):
        env._logger.info(f"episode {e+1}/{episodes} began")
        _ = env.reset()
        env.model.train()
        action, reward, done, _, close = env.step(1)
        rewards = [reward]
        while not done:
            action, reward, done, _, close = env.step(action)
            rewards += [reward]

        optimizer.zero_grad()
        log_return = env.log_return
        loss = compute_loss(
            log_probs  = env.log_probs, 
            rewards    = rewards, 
            beta       = env.beta,
            log_return = log_return,
            device     = env._device)
        
        loss.backward()
        nn.utils.clip_grad_norm_(env.model.parameters(), max_grad_norm)
        optimizer.step()

        reward_history += [sum(rewards)]
        portfolios += [env._manager.portfolio]

        if not buy_and_hold:
            buy_and_hold = (close["price"] / env.init_close - 1)

        env._logger.info(f"""\n
        episode {e+1}/{episodes} done
        loss:            {' ' if loss.item() > 0 else ''}{loss.item():.4f}
        model portfolio: {'+' if env._manager.portfolio > 1 else ''}{(env._manager.portfolio-1) * 100:.4f}%
        buy and hold:    {'+' if buy_and_hold > 0 else ''}{buy_and_hold * 100:.4f}%
        """)
        env.model.eval()

        avg_port = portfolios[-1] - 1 # np.mean(portfolios) - 1
        if avg_port > buy_and_hold and portfolios[-1] > max(0.0, buy_and_hold) and len(portfolios) == portfolio_size:
            env._logger.info(f"average target reached, last {portfolio_size} averaged {'+' if avg_port > 1 else ''}{avg_port * 100:.4f}%, exiting.")
            return (portfolios[-1] / (buy_and_hold + 1))-1

    env._logger.info(f"training complete, last {portfolio_size} averaged {'+' if avg_port > 1 else ''}{avg_port * 100:.4f}%")
    return (portfolios[-1] / (buy_and_hold + 1))-1

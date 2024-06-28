import logging
import numpy as np
import pandas as pd

import torch
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Data
from typing import Dict, List, Tuple
from backend.manager import Manager
from backend.trade import TradeType, Action
from reinforce.model import PolicyNet, CriticNet
from reinforce.utils import select_action, compute_policy_loss, compute_critic_loss


class Environment:
    '''
    This class is the reinforcement learning environment used for training and testing
    '''
    def __init__(
            self,
            device:  str,
            min_val: float,
            cost:    float,
            dataset: List[Dict[str, pd.DataFrame | Data]] | None) -> None:
        
        self._device  = device
        self._min_val = min_val
        self._dataset = dataset
        self._manager = Manager(trading_cost=cost)

    @property
    def dataset(self) -> List[Dict[str, pd.DataFrame | Data]]:
        return self._dataset
    
    @dataset.setter
    def dataset(self, new_set: List[Data]) -> None:
        self._dataset = new_set

    def _reset(self) -> None:
        self._manager.reset()

    def _step(
            self, 
            data:       Dict[str, pd.DataFrame | Data],
            argmax:     bool,
            policy_net: PolicyNet,
            critic_net: CriticNet | None = None) -> Tuple[bool, List[int], torch.Tensor, float]:
        
        done        = False
        reward      = []
        graph       = data['graph']
        price       = data['price']
        index       = data['index']
        ticker      = data['asset']
        close       = price[ticker]['Price']['Close'].mean(None)
        positions   = self._manager.positions(close)
        uuids       = [position['uuid'] for position in positions]
        types       = [position['type'] for position in positions]
        log_returns = [position['log_return'] for position in positions]
        
        market_log_return = data['log_return']
        if self._manager.liquid:
            uuids.append(self._manager.market_uuid)
            types.append(TradeType.Market.value)
            log_returns.append(market_log_return)

        actions, log_probs = select_action(
            model      = policy_net,
            state      = graph,
            index      = index,
            inp_types  = types,
            log_return = log_returns,
            device     = self._device,
            argmax     = argmax)

        for i, choice in enumerate(actions):
            action     = Action(choice)
            uuid       = uuids[i]
            log_return = log_returns[i]
            if action == Action.Buy:
                if uuid == self._manager.market_uuid:
                    final = self._manager.long(close, np.exp(log_probs[i].item()))
                    if final == Action.Buy:
                        reward.append(np.log(self._manager.value)) # Did buy, add total log return
                    else:
                        reward.append(-market_log_return) # Did not buy, subtract market log return
                else: # TODO Close short position
                    reward.append(-1)

            elif action == Action.Sell:
                if uuid != self._manager.market_uuid:
                    final = self._manager.short(close, np.exp(log_probs[i].item()), uuid)
                    if final == Action.Sell:
                        reward.append(log_return) # Did sell, add trade log return
                    else:
                        reward.append(-log_return) # Did not sell, subtract trade log return
                else: # TODO Short selling
                    reward.append(-1)

            elif action == Action.Hold:
                final = self._manager.hold(close, np.exp(log_probs[i].item()))
                if final == Action.Hold:
                    if uuid == self._manager.market_uuid:
                        reward.append(market_log_return) # Did hold, subtract market log return
                    else:
                        reward.append(-log_return) # Did hold, subtract trade log return
                else:
                    pass # TODO Surprise action
            
            else: # Impossible
                raise NotImplementedError('the selected action is not a valid one')

        if self._manager.value < self._min_val:
            done = True
        
        values = critic_net(
            data       = graph,
            index      = index,
            inp_types  = types,
            log_return = log_returns) if critic_net is not None else None

        return done, values, log_probs, reward

    def train(
            self,
            episodes:   int,
            policy_net: PolicyNet,
            policy_opt: optim.Optimizer,
            critic_net: CriticNet | None = None,
            critic_opt: optim.Optimizer | None = None) -> None:

        policy_net.train()
        if critic_net is not None:
            critic_net.train()

        for episode in range(episodes):
            logging.debug(f'episode {episode+1}/{episodes} started\n')
            self._reset()
            policy_opt.zero_grad()
            if critic_opt is not None:
                critic_opt.zero_grad()

            rewards   = []
            log_probs = []
            values    = []
            for data in self._dataset:
                done, value, log_prob, reward = self._step(
                    policy_net = policy_net,
                    critic_net = critic_net,
                    data       = data,
                    argmax     = False)

                rewards += reward
                log_probs.append(log_prob.view(-1, 1))
                if value is not None:
                    values.append(value)
                if done: 
                    break

            if critic_opt is not None:
                critic_loss = compute_critic_loss(
                    rewards = rewards,
                    values  = values,
                    device  = self._device
                )
                critic_loss.backward()
                clip_grad_norm_(
                    parameters = critic_net.parameters(), 
                    max_norm   = 1.0,
                    error_if_nonfinite=True)
                critic_opt.step()

            sharpe = self._manager.sharpe_ratio
            policy_loss = compute_policy_loss(
                log_probs = log_probs,
                rewards   = rewards,
                values    = values if len(values) > 0 else None,
                sharpe    = sharpe,
                device    = self._device
            )

            policy_loss.backward()
            clip_grad_norm_(
                parameters = policy_net.parameters(), 
                max_norm   = 1.0,
                error_if_nonfinite=True)
            policy_opt.step()

            logging.debug(f'episode {episode+1}/{episodes} done, loss={policy_loss.item():.4f}, returns={self._manager.value:.4f}')
        logging.info('training complete')

    @torch.no_grad()
    def test(
            self,
            policy_net: PolicyNet) -> float:

        self._reset()
        policy_net.eval()
        rewards   = []
        log_probs = []

        for i in range(len(self._dataset)):
            data = self._dataset[i]

            done, _, log_prob, reward = self._step(
                policy_net = policy_net,
                data   = data,
                argmax = True)

            rewards.append(reward)
            log_probs.append(log_prob)

            if done: 
                break
        
        logging.debug(f'testing done, returns={self._manager.value:.4f}')
        return self._manager.value
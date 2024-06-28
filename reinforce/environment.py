import logging
import numpy as np
import pandas as pd

import torch
import torch.optim as optim

from torch_geometric.data import Data
from typing import Dict, List, Tuple
from backend.manager import Manager
from backend.trade import TradeType, Action
from reinforce.model import PolicyNet
from reinforce.utils import select_action, compute_loss


class Environment:
    '''
    This class is the reinforcement learning environment used for training and testing
    '''
    def __init__(
            self,
            device:  str,
            min_val: float,
            dataset: List[Dict[str, pd.DataFrame | Data]] | None) -> None:
        
        self._device  = device
        self._min_val = min_val
        self._dataset = dataset
        self._manager = Manager()

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
            model:  PolicyNet,
            data:   Dict[str, pd.DataFrame | Data],
            argmax: bool = False) -> Tuple[bool, List[int], torch.Tensor, float]:
        
        done        = False
        reward      = 0
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
            model      = model,
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
                        reward += np.log(self._manager.value) # Did buy, add total log return
                    else:
                        reward -= market_log_return # Did not buy, subtract market log return

            elif action == Action.Sell:
                if uuid != self._manager.market_uuid:
                    final = self._manager.short(close, np.exp(log_probs[i].item()), uuid)
                    if final == Action.Sell:
                        reward += log_return # Did sell, add trade log return
                    else:
                        reward -= log_return # Did not sell, subtract trade log return

            elif action == Action.Hold:
                final = self._manager.hold(close, np.exp(log_probs[i].item()))
                if final == Action.Hold:
                    if uuid == self._manager.market_uuid:
                        reward -= market_log_return # Did hold, subtract market log return
                    else:
                        reward -= log_return # Did hold, subtract trade log return
                else:
                    pass # TODO Surprise action

            else: # Impossible
                raise NotImplementedError('the selected action is not a valid one')

        if self._manager.value < self._min_val:
            done = True
        
        return done, actions, log_probs, reward

    def train(
            self,
            episodes:   int,
            policy_net: PolicyNet,
            optimizer:  optim.Optimizer) -> None:

        policy_net.train()
        for episode in range(episodes):
            logging.debug(f'episode {episode+1}/{episodes} started\n')
            self._reset()
            optimizer.zero_grad()

            rewards   = []
            log_probs = []
            for i in range(len(self._dataset)):
                data = self._dataset[i]

                done, _, log_prob, reward = self._step(
                    model  = policy_net,
                    data   = data,
                    argmax = False)

                rewards.append(reward)
                log_probs.append(log_prob.view(-1, 1))

                if done: 
                    break

            loss = compute_loss(
                log_probs = log_probs,
                rewards   = rewards,
                device    = self._device
            )

            loss.backward()
            optimizer.step()
            logging.debug(f'episode {episode+1}/{episodes} done, loss={loss.item():.4f}, returns={self._manager.value:.4f}')
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
                model  = policy_net,
                data   = data,
                argmax = True)

            rewards.append(reward)
            log_probs.append(log_prob)

            if done: 
                break
        
        logging.debug(f'testing done, returns={self._manager.value:.4f}')
        return self._manager.value
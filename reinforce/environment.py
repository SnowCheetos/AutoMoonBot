import logging
import numpy as np
import pandas as pd

import torch
import torch.optim as optim

from torch.nn.utils import clip_grad_norm_
from torch_geometric.data import Data
from typing import Dict, List, Tuple
from automoonbot.manager import Manager
from automoonbot.trade import TradeType, Action
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
        
        self._cost    = cost
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
        potential   = data['potential']
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

        actions, log_probs, entropy = select_action(
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
            prob       = np.exp(log_probs[i].item())
            if action == Action.Buy:
                if uuid == self._manager.market_uuid:
                    final = self._manager.long(close, prob, uuid)
                    
                    if final == Action.Buy:
                        reward.append(potential)
                    
                    else:
                        reward.append(-potential)
                
                else: # TODO Close short position
                    final = self._manager.hold(close, 1, uuid)
                    reward.append(0)

            elif action == Action.Sell:
                if uuid != self._manager.market_uuid:
                    final = self._manager.short(close, prob, uuid)
                    
                    if final == Action.Sell:
                        reward.append(log_return - potential)
                    
                    else:
                        reward.append(potential)
                
                else: # TODO Open short position
                    final = self._manager.hold(close, 1, uuid)
                    reward.append(0)

            elif action == Action.Hold:
                final = self._manager.hold(close, prob, uuid)
                if final == Action.Hold:
                    if uuid == self._manager.market_uuid:
                        reward.append(-potential) # Did hold
                    else:
                        reward.append(potential)
                
                else:
                    reward.append(0)
            
            else: # Impossible
                raise NotImplementedError('the selected action is not a valid one')

        if self._manager.value <= self._min_val:
            done = True
        
        values = critic_net(
            data       = graph,
            index      = index,
            inp_types  = types,
            log_return = log_returns) if critic_net is not None else None

        return done, values, log_probs, reward, entropy

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

        ticker        = self._dataset[0]['asset']
        market_entry  = self._dataset[0]['price'][ticker]['Price']['Close'].mean(None)
        market_exit   = self._dataset[-1]['price'][ticker]['Price']['Close'].mean(None)
        market_return = market_exit / market_entry

        closes = [sample['price'][ticker]['Price']['Close'].mean(None) for sample in self._dataset]
        for i in range(len(self._dataset)):
            max_index = np.argmax(closes[i:])
            distance  = max_index - i
            max_value = closes[max_index]
            potential = np.log(max_value / closes[i]) * (0.99**distance)
            self._dataset[i]['potential'] = potential

        for episode in range(episodes):
            logging.debug(f'episode {episode+1}/{episodes} started\n')
            self._reset()
            policy_opt.zero_grad()
            if critic_opt is not None:
                critic_opt.zero_grad()

            rewards   = []
            log_probs = []
            values    = []
            entropies = []

            for data in self._dataset:
                done, value, log_prob, reward, entropy = self._step(
                    policy_net = policy_net,
                    critic_net = critic_net,
                    data       = data,
                    argmax     = False)

                rewards += reward
                log_probs.append(log_prob.view(-1, 1))
                entropies.append(entropy)
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
                # clip_grad_norm_(
                #     parameters = critic_net.parameters(), 
                #     max_norm   = 1.0,
                #     error_if_nonfinite=True)
                critic_opt.step()

            sharpe = self._manager.sharpe_ratio
            policy_loss = compute_policy_loss(
                log_probs = log_probs,
                rewards   = rewards,
                values    = values if len(values) > 0 else None,
                entropy   = entropies,
                sharpe    = sharpe,
                beta      = -0.5,
                device    = self._device
            )

            policy_loss.backward()
            # clip_grad_norm_(
            #     parameters = policy_net.parameters(), 
            #     max_norm   = 1.0,
            #     error_if_nonfinite=True)
            policy_opt.step()

            logging.debug(f'episode {episode+1}/{episodes} done, loss={policy_loss.item():.4f}, returns={self._manager.value:.4f}')
            if self._manager.value > max(1, market_return):
                break
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

            done, values, log_prob, reward, entropy = self._step(
                policy_net = policy_net,
                data   = data,
                argmax = True)

            rewards.append(reward)
            log_probs.append(log_prob)

            if done: 
                break
        
        logging.debug(f'testing done, returns={self._manager.value:.4f}')
        return self._manager.value
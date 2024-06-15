import logging
import threading

from typing import Dict, List

from backend.buffer import DataBuffer
from reinforce.utils import Position, Action
from reinforce.environment import TradeEnv, train
from reinforce.model import PolicyNet, inference

class Server:
    def __init__(
            self, 
            ticker:         str,
            period:         str,
            interval:       str,
            queue_size:     int,
            state_dim:      int,
            action_dim:     int,
            embedding_dim:  int,
            inaction_cost:  float,
            action_cost:    float,
            device:         str,
            db_path:        str,
            return_thresh:  float,
            feature_params: Dict[str, List[int] | Dict[str, List[int]]]) -> None:
        
        self._position = Position.Cash
        self._device = device

        self._buffer = DataBuffer(
            ticker=ticker, 
            period=period, 
            interval=interval, 
            queue_size=queue_size, 
            feature_params=feature_params)
        
        self._model = PolicyNet(
            input_dim=state_dim, 
            output_dim=action_dim, 
            position_dim=len(Position), 
            embedding_dim=embedding_dim)
        
        self._env = TradeEnv(
            state_dim=state_dim, 
            action_dim=action_dim, 
            embedding_dim=embedding_dim, 
            queue_size=queue_size, 
            inaction_cost=inaction_cost,
            action_cost=action_cost,
            device=device,
            db_path=db_path,
            return_thresh=return_thresh)
        
        self._ready = False
        self._training = False
        self._train_thread = None

    @property
    def busy(self):
        return self._training

    def _update_model(self) -> None:
        if not self._training:
            self._model.load_state_dict(self._env.model_weights)
        else:
            logging.warning("model currently being trained, cannot copy weights")

    def _train(
            self,
            episodes:       int, 
            learning_rate:  float, 
            momentum:       float,
            max_grad_norm:  float,
            portfolio_size: int) -> None:
        
        self._training = True
        
        train(
            env=self._env, 
            episodes=episodes, 
            learning_rate=learning_rate,
            momentum=momentum,
            max_grad_norm=max_grad_norm,
            portfolio_size=portfolio_size)
        
        self._training = False
        if not self._ready:
            self._ready = True

    def train_model(
            self,
            episodes:       int, 
            learning_rate:  float=1e-3, 
            momentum:       float=1e-3,
            max_grad_norm:  float=1.0,
            portfolio_size: int=5):
        
        thread = threading.Thread(
            target=self._train, 
            args=(
                episodes, 
                learning_rate, 
                momentum, 
                max_grad_norm, 
                portfolio_size))
        
        thread.start()
        self._train_thread = thread
    
    def run_inference(self, update: bool=True) -> Action | None:
        state = self._buffer.fetch_state(update)
        if len(state) == 0:
            logging.warning("no data available, not running inference")
            return None

        result = inference(
            self._model,
            state,
            int(self._position.value),
            self._device)
        action = Action(result)

        if self._position == Position.Cash and action == Action.Sell:
            self._position = Position.Asset
            logging.info("sell signal predicted")

        elif self._position == Position.Asset and action == Action.Buy:
            self._position = Position.Cash
            logging.info("buy signal predicted")

        return action
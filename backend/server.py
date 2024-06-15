import logging
import threading

from typing import Dict, List, Optional

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
            return_thresh:  float,
            feature_params: Dict[str, List[int] | Dict[str, List[int]]],
            db_path:        Optional[str] = None,
            logger:         Optional[logging.Logger] = None) -> None:
        
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

        self._position = Position.Cash
        self._device = device

        if not db_path:
            db_path = f"../data/{ticker}.db"

        self._buffer = DataBuffer(
            ticker=ticker, 
            period=period, 
            interval=interval, 
            queue_size=queue_size, 
            db_path=db_path,
            feature_params=feature_params)
        
        self._buffer.write_queue_to_db(flush=True)

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

    def __del__(self):
        if self._training:
            self.join_train_thread()

    @property
    def busy(self):
        return self._training

    def tohlcv(self) -> Dict[str, float]:
        return self._buffer.last_tohlcv()

    def update_model(self) -> bool:
        if not self._training:
            self._model.load_state_dict(self._env.model_weights)
            return True
        else:
            self._logger.warning("model currently being trained, cannot copy weights")
            return False

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
        self._logger.info(f"training started, thread id: {thread}")
        self._train_thread = thread
    
    def join_train_thread(self) -> bool:
        if not self._training:
            self._logger.warning("model not training, nothing to end...")
            return False
        
        self._train_thread.join()
        return True

    def run_inference(self, update: bool=True) -> Action | None:
        state = self._buffer.fetch_state(update)
        if len(state) == 0:
            self._logger.warning("no data available, not running inference")
            return None

        result = inference(
            self._model,
            state,
            int(self._position.value),
            self._device)
        action = Action(result)

        if self._position == Position.Cash and action == Action.Sell:
            self._position = Position.Asset
            self._logger.debug("sell signal predicted")

        elif self._position == Position.Asset and action == Action.Buy:
            self._position = Position.Cash
            self._logger.debug("buy signal predicted")

        return action
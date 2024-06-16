import time
import logging
import threading
from threading import Lock

from collections import deque
from typing import Dict, List, Optional

from backend.buffer import DataBuffer
from reinforce.utils import Position, Action, Status, Signal
from reinforce.environment import TradeEnv, train
from reinforce.model import PolicyNet, inference


class Server:
    def __init__(
            self, 
            ticker:          str,
            period:          str,
            interval:        str,
            queue_size:      int,
            state_dim:       int,
            action_dim:      int,
            embedding_dim:   int,
            inaction_cost:   float,
            action_cost:     float,
            device:          str,
            return_thresh:   float,
            retrain_freq:    int,
            training_params: Dict[str, int | float],
            feature_params:  Dict[str, List[int] | Dict[str, List[int]]],
            db_path:         Optional[str] = None,
            live_data:       bool=False,
            logger:          Optional[logging.Logger] = None) -> None:
        
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

        self._mutex = Lock()
        self._position = Position.Cash
        self._device = device
        self._status = Status(0.005, 1.0025)

        if not db_path:
            db_path = f"../data/{ticker}.db"

        self._buffer = DataBuffer(
            ticker=ticker, 
            period=period, 
            interval=interval, 
            queue_size=queue_size, 
            db_path=db_path,
            feature_params=feature_params,
            logger=logger,
            live_data=live_data)
        
        if live_data:
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

        self._training_params = training_params

        self._ready = False
        self._training = False
        self._train_thread = None

        self._inferencing = False
        self._inference_thread = None

        self._timer_thread  = None
        self._train_counter = retrain_freq
        self._retrain_freq  = retrain_freq

        self._terminate = False
        self._actions_queue = deque(maxlen=5)

    def __del__(self):
        self.join_timer_thread()

        if self._inferencing:
            self.join_inference_thread()

        if self._training:
            self.join_train_thread()

    @property
    def busy(self):
        return self._training

    @property
    def inf_busy(self):
        return self._inferencing

    def consume_queue(self) -> Dict[str, int | float | str] | None:
        with self._mutex:
            if len(self._actions_queue) > 0:
                return self._actions_queue.pop()
        return None

    def _timer_loop(self, interval: int):
        self._logger.info("starting timer thread")
        self.train_model(
            episodes=self._training_params["episodes"],
            learning_rate=self._training_params["learning_rate"],
            momentum=self._training_params["momentum"],
            max_grad_norm=self._training_params["max_grad_norm"],
            portfolio_size=self._training_params["portfolio_size"])

        while not self._terminate:
            if self._train_counter == 0:
                self._logger.info("running scheduled training")
                self.train_model(
                    episodes=self._training_params["episodes"],
                    learning_rate=self._training_params["learning_rate"],
                    momentum=self._training_params["momentum"],
                    max_grad_norm=self._training_params["max_grad_norm"],
                    portfolio_size=self._training_params["portfolio_size"])
                self._train_counter = self._retrain_freq

            time.sleep(interval)
            self._logger.info("running scheduled inference")
            self._inference()
            self._train_counter -= 1

    def start_timer(self, interval: int):
        thread = threading.Thread(
            target=self._timer_loop,
            args=(interval,)
        )
        thread.start()

        with self._mutex:
            self._timer_thread = thread

    def tohlcv(self) -> Dict[str, float]:
        return self._buffer.last_tohlcv()

    def fetch_buffer(self) -> Dict[str, Dict[str, float]]:
        with self._mutex:
            return self._buffer.queue

    def update_model(self) -> bool:
        self._mutex.acquire_lock()
        if not self._training and not self._inferencing:
            self._model.load_state_dict(self._env.model_weights)
            self._mutex.release_lock()
            self._logger.info("model updated successfully")
            return True
        else:
            self._logger.warning("model currently being trained, cannot copy weights")
            self._mutex.release_lock()
            return False

    def _train(
            self,
            episodes:       int, 
            learning_rate:  float, 
            momentum:       float,
            max_grad_norm:  float,
            portfolio_size: int) -> None:
        
        with self._mutex:
            self._training = True
        
        train(
            env=self._env, 
            episodes=episodes, 
            learning_rate=learning_rate,
            momentum=momentum,
            max_grad_norm=max_grad_norm,
            portfolio_size=portfolio_size)
        
        with self._mutex:
            self._training = False
            if not self._ready:
                self._ready = True

        self.update_model()

    def train_model(
            self,
            episodes:       int, 
            learning_rate:  float=1e-3, 
            momentum:       float=1e-3,
            max_grad_norm:  float=1.0,
            portfolio_size: int=5):
        
        with self._mutex:
            if self._training:
                self._logger.info(f"there is another training thread running, wait for it to finish first")
                return

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
        
        with self._mutex:
            self._train_thread = thread
    
    def join_timer_thread(self) -> bool:
        with self._mutex:
            if not self._timer_thread:
                self._logger.warning("no running timer thread, nothing to end...")
                return False
        
        self._timer_thread.join()
        return True

    def join_train_thread(self) -> bool:
        with self._mutex:
            if not self._training:
                self._logger.warning("model not training, nothing to end...")
                return False
        
        self._train_thread.join()
        return True

    def join_inference_thread(self) -> bool:
        with self._mutex:
            if not self._inferencing:
                self._logger.warning("model not inferencing, nothing to end...")
                return False
        
        self._inference_thread.join()
        return True

    def _inference(self) -> None:
        with self._mutex:
            if self._inferencing:
                self._logger.info(f"there is another inferencing thread running, wait for it to finish first")
                return
            
        thread = threading.Thread(
            target=self._void_inference,
            args=(),
        )

        thread.start()
        with self._mutex:
            self._inference_thread = thread

    def _void_inference(self) -> None:
        with self._mutex:
            if not self._ready:
                self._logger.warning("model not ready, not running inference")
                return

        state = self._buffer.fetch_state(False)
        ohlcv = self._buffer.queue["data"][-1]
        if len(state) == 0:
            self._logger.warning("no data available, not running inference")
            return None
        
        with self._mutex:
            if self._inferencing:
                self._logger.info(f"there is another inference thread running, wait for it to finish first")
                return
            
            self._inferencing = True

        self._logger.info(f"running void inferencing, queue size: {len(self._actions_queue)}")
        result = inference(
            self._model,
            state,
            int(self._position.value),
            self._device)
        action = Action(result)

        actual_action = Action.Hold
        with self._mutex:
            if self._position == Position.Cash and action == Action.Buy:
                if self._status.signal == Signal.Idle:
                    self._status.signal      = action.value
                    self._status.take_profit = ohlcv["close"]
                    self._status.stop_loss   = ohlcv["close"]
                elif self._status.signal == Signal.Buy:
                    if self._status.confirm_buy(ohlcv["close"]):
                        self._position = Position.Asset
                        self._logger.debug("buy signal predicted")
                        actual_action = Action.Buy
                        self._status.reset()

            elif self._position == Position.Asset and action == Action.Sell:
                if self._status.signal == Signal.Idle:
                    self._status.signal      = action.value
                    self._status.take_profit = ohlcv["close"]
                    self._status.stop_loss   = ohlcv["close"]
                elif self._status.signal == Signal.Sell:
                    if self._status.confirm_sell(ohlcv["close"]):
                        self._position = Position.Cash
                        self._logger.debug("sell signal predicted")
                        actual_action = Action.Sell
                        self._status.reset()

            take_profit = self._status.take_profit if self._status.take_profit > 0 else None
            stop_loss = self._status.stop_loss if self._status.stop_loss > 0 else None
            self._actions_queue.append({
                "timestamp":   ohlcv["timestamp"],
                "action":      actual_action.name,
                "close":       ohlcv["close"],
                "take_profit": take_profit,
                "stop_loss":   stop_loss
            })
            self._inferencing = False

    def run_inference(self, update: bool=True) -> Action | None:
        state = self._buffer.fetch_state(update)
        ohlcv = self._buffer.queue["data"][-1]

        if len(state) == 0:
            self._logger.warning("no data available, not running inference")
            return None

        with self._mutex:
            if not self._ready:
                self._logger.warning("model not ready, not running inference")
                return
        
        with self._mutex:
            if self._inferencing:
                self._logger.info(f"there is another inference thread running, wait for it to finish first")
                return
            
            self._inferencing = True
        
        self._logger.info(f"running blocking inferencing...")
        result = inference(
            self._model,
            state,
            int(self._position.value),
            self._device)
        action = Action(result)

        actual_action = Action.Hold
        with self._mutex:
            if self._position == Position.Cash and action == Action.Buy:
                if self._status.signal == Signal.Idle:
                    self._status.signal      = action.value
                    self._status.take_profit = ohlcv["close"]
                    self._status.stop_loss   = ohlcv["close"]
                elif self._status.signal == Signal.Buy:
                    if self._status.confirm_buy(ohlcv["close"]):
                        self._position = Position.Asset
                        self._logger.debug("buy signal predicted")
                        actual_action = Action.Buy
                        self._status.reset()

            elif self._position == Position.Asset and action == Action.Sell:
                if self._status.signal == Signal.Idle:
                    self._status.signal      = action.value
                    self._status.take_profit = ohlcv["close"]
                    self._status.stop_loss   = ohlcv["close"]
                elif self._status.signal == Signal.Sell:
                    if self._status.confirm_sell(ohlcv["close"]):
                        self._position = Position.Cash
                        self._logger.debug("sell signal predicted")
                        actual_action = Action.Sell
                        self._status.reset()

            self._inferencing = False
        return actual_action
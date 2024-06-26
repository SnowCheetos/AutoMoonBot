import random
import time
import torch
import logging
import threading

from threading import Lock
from collections import deque
from typing import Dict, List, Optional

from backend.buffer import DataBuffer
from reinforce.model import PolicyNet
from backend.session import Session
from backend.manager import Manager


class Server:
    def __init__(
            self, 
            ticker:            str,
            period:            str,
            interval:          str,
            queue_size:        int,
            state_dim:         int,
            action_dim:        int,
            embedding_dim:     int,
            inaction_cost:     float,
            action_cost:       float,
            device:            str,
            return_thresh:     float,
            retrain_freq:      int,
            training_params:   Dict[str, int | float],
            feature_params:    Dict[str, List[int] | Dict[str, List[int]]],
            db_path:           Optional[str] = None,
            live_data:         bool=False,
            sharpe_cutoff:     int=30,
            full_port:         bool=False,
            gamma:             float=0.1,
            alpha:             float=1.5,
            beta:              float=0.5,
            zeta:              float=0.5,
            leverage:          float=1.0,
            quick_sell:        bool=False,
            num_mem:           int=512,
            mem_dim:           int=128,
            inference_method:  str="prob",
            checkpoint_path:   str="checkpoint",
            logger:            Optional[logging.Logger] = None,
            max_training_data: int | None = None) -> None:
        
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

        self._ticker           = ticker
        self._period           = period
        self._interval         = interval
        self._live_data        = live_data
        self._new_session      = True
        self._mutex            = Lock()
        self._position         = Position.Cash
        self._device           = device
        self._inference_method = inference_method

        if not db_path:
            db_path = f"../data/{ticker}.db"

        self._buffer       = DataBuffer(
            ticker         = ticker, 
            period         = period, 
            interval       = interval, 
            queue_size     = queue_size, 
            db_path        = db_path,
            feature_params = feature_params,
            logger         = logger,
            live_data      = live_data)
        
        if live_data:
            self._buffer.write_queue_to_db(flush=True)

        self._model = PolicyNet(
            input_dim     = state_dim, 
            output_dim    = action_dim, 
            position_dim  = len(Position), 
            embedding_dim = embedding_dim,
            num_mem       = num_mem,
            mem_dim       = mem_dim).to(device)

        self._env = TradeEnv(
            state_dim         = state_dim, 
            action_dim        = action_dim, 
            embedding_dim     = embedding_dim, 
            queue_size        = queue_size, 
            inaction_cost     = inaction_cost,
            action_cost       = action_cost,
            device            = device,
            db_path           = db_path,
            sharpe_cutoff     = sharpe_cutoff,
            return_thresh     = return_thresh,
            testing           = not live_data,
            max_training_data = max_training_data,
            feature_params    = feature_params,
            alpha             = alpha,
            beta              = beta,
            gamma             = gamma,
            zeta              = zeta,
            leverage          = leverage,
            mem_dim           = mem_dim,
            num_mem           = num_mem,
            quick_sell        = quick_sell)

        self._manager   = TradeManager(
            cov         = self._buffer.coef_of_var, 
            alpha       = alpha, 
            gamma       = gamma, 
            cost        = action_cost, 
            full_port   = full_port,
            qk_sell     = quick_sell)

        self._nounce           = 0
        self._beat             = 0
        self._max_access_accum = 0
        self._checkpoint_path  = checkpoint_path
        self._training_params  = training_params
        self._ready            = False
        self._training         = False
        self._train_thread     = None
        self._inferencing      = False
        self._inference_thread = None
        self._timer_thread     = None
        self._train_counter    = retrain_freq
        self._retrain_freq     = retrain_freq
        self._terminate        = False
        self._gamma            = gamma
        self._epsilon          = 0.9
        self._epsilon_decay    = 0.9
        self._data_queue       = deque(maxlen=25)

    def __del__(self):
        self.join_timer_thread()

        if self._inferencing:
            self.join_inference_thread()

        if self._training:
            self.join_train_thread()

    @property
    def new_session(self) -> bool:
        if self._new_session:
            self._new_session = False
            return True
        return False

    @property
    def session_info(self) -> Dict[str, str]:
        return {
            "type":     "new_session",
            "ticker":   self._ticker,
            "period":   self._period if not self._live_data else "live",
            "interval": self._interval
        }

    @property
    def busy(self):
        return self._training

    @property
    def inf_busy(self):
        return self._inferencing

    def _append_action(self, timestamp: str, action: Action, price: float, prob: float, amount: float):
        self._data_queue.append({
            "type":        "action",
            "timestamp":   timestamp,
            "action":      action.name,
            "close":       price,
            "probability": prob,
            "amount":      amount
        })

    def _append_trade(self, timestamp: str, entry: float, exit: float, amount: float):
        self._data_queue.append({
            "type":      "trade",
            "timestamp": timestamp,
            "entry":     entry,
            "exit":      exit,
            "amount":    amount
        })

    def save_model(self, name: str) -> None:
        path = self._checkpoint_path + "/" + name
        with self._mutex:
            state_dict = self._env.model_weights
        torch.save(state_dict, path)

    def load_model(self, path: str) -> None:
        state_dict = torch.load(path)
        with self._mutex:
            self._env.model = state_dict

    def status_report(self) -> Dict[str, int | bool | str]:
        r = {
            "type":      "report",
            "training":  self._training,
            "ready":     self._ready,
            "done":      self._buffer.done
        }
        self._data_queue.append(r)
        return r

    def consume_queue(self) -> List[Dict[str, int | float | str]] | None:
        with self._mutex:
            if len(self._data_queue) > 0:
                d = list(self._data_queue)
                self._data_queue.clear()
                return d
        return None

    def _timer_loop(self, interval: int):
        self._logger.info("starting timer thread")
        self.train_model(
            episodes=self._training_params["episodes"],
            learning_rate=self._training_params["learning_rate"],
            momentum=self._training_params["momentum"],
            weight_decay=self._training_params["weight_decay"],
            max_grad_norm=self._training_params["max_grad_norm"],
            min_episodes=self._training_params["min_episodes"])

        while not self._terminate and not self._buffer.done:
            ohlc = self.tohlcv()
            ohlc["type"]   = "ohlc"
            ohlc["nounce"] = self._nounce
            self._nounce  += 1
            self._data_queue.append(ohlc)
            if self._train_counter == 0:
                self._logger.info("running scheduled training")
                self.train_model(
                    episodes=self._training_params["episodes"],
                    learning_rate=self._training_params["learning_rate"],
                    momentum=self._training_params["momentum"],
                    weight_decay=self._training_params["weight_decay"],
                    max_grad_norm=self._training_params["max_grad_norm"],
                    min_episodes=self._training_params["min_episodes"])
                self._train_counter = self._retrain_freq

            self._buffer.update_queue(False)
            self._logger.info("running scheduled inference")
            if self._ready:
                self._inference()
            else:
                self._logger.warning("model not ready, not running inference")
            self._train_counter -= 1
            time.sleep(interval)
        self._logger.warning("timer loop terminated")

    def start_timer(self, interval: int):
        thread = threading.Thread(
            target=self._timer_loop,
            args=(interval,)
        )
        thread.start()

        with self._mutex:
            self._timer_thread = thread

    def tohlcv(self) -> Dict[str, float]:
        if not self._training:
            if self._max_access_accum > 0:
                self._env.sampler.max_access += self._max_access_accum + 1
                self._max_access_accum = 0
            else:
                self._env.sampler.max_access += 1
        else:
            self._max_access_accum += 1
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
            weight_decay:   float,
            max_grad_norm:  float,
            min_episodes:   int) -> None:
        
        with self._mutex:
            self._training = True
        
        beat = train(
            env=self._env, 
            episodes=episodes, 
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            min_episodes=min_episodes)
        
        with self._mutex:
            self._training = False

        if beat > self._beat: # or self._beat == 0:
            # if self._epsilon < random.random():
            # self._epsilon *= self._epsilon_decay
            self._beat = beat
            self.update_model()

        with self._mutex:
            if not self._ready:
                self._ready = True

    def train_model(
            self,
            episodes:       int, 
            learning_rate:  float=1e-3, 
            momentum:       float=0.9,
            weight_decay:   float=0.9,
            max_grad_norm:  float=1.0,
            min_episodes:   int=10):
        
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
                weight_decay,
                max_grad_norm, 
                min_episodes))
        
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
        data  = self._buffer.queue["data"][-1]
        price = data["close"]
        if len(state) == 0:
            self._logger.warning("no data available, not running inference")
            self._ready = False
            return None
        
        with self._mutex:
            if self._inferencing:
                self._logger.info(f"there is another inference thread running, wait for it to finish first")
                return
            
            self._inferencing = True

        result, prob  = inference(
            model     = self._model,
            state     = state,
            position  = int(self._position.value),
            potential = self._manager.potential_gain(price) - 1,
            device    = self._device,
            method    = self._inference_method)
        
        action        = Action(result)
        actual_action = Action.Hold
        with self._mutex:
            self._risk_free_rate = price / self._buffer.queue["data"][0]["close"]
            # Validate buy
            if action == Action.Buy:
                actual_action = self._manager.try_buy(price, self._buffer.coef_of_var, prob)
            elif action == Action.Sell:
                gain = self._manager.try_sell(price, self._buffer.coef_of_var)
                if gain >= 0:
                    actual_action = Action.Sell
                    trade = self._manager.last_trade
                    self._append_trade(data["timestamp"], trade["entry"], trade["exit"], trade["amount"])

            if action != Action.Hold:
                self._append_action(data["timestamp"], actual_action, price, prob, self._manager.curr_trade.amount)
        self._inferencing = False


class NewServer:
    def __init__(
            self,
            ticker:         str,
            interval:       str,
            buffer_size:    int,
            device:         str,
            feature_config: Dict[str, List[str | int] | str],
            live:           bool = False,
            preload:        bool = True,
            db_path:        str = 'data',
            session_id:     str | None = None,
            market_rep:     List[str] = ['VTI', 'GLD', 'USO']
        ) -> None:
        
        self._manager = Manager()
        self._session = Session(
            ticker         = ticker,
            interval       = interval,
            buffer_size    = buffer_size,
            device         = device,
            feature_config = feature_config,
            live           = live,
            preload        = preload,
            db_path        = db_path,
            session_id     = session_id,
            market_rep     = market_rep
        )

        self._session.start()
import time
import torch
import logging
import threading
import numpy as np
import pandas as pd

from collections import deque
from typing import Callable, Dict, List
from torch.optim import SGD
from torch_geometric.data import Data
from backend.loader import DataLoader
from backend.manager import Manager
from backend.trade import Action, TradeType
from reinforce.model import PolicyNet
from reinforce.environment import Environment
from reinforce.utils import select_action


class Session:
    '''
    This class holds the data loader as well as training and testing environment.
    '''
    def __init__(
            self,
            ticker:         str,
            interval:       str,
            buffer_size:    int,
            device:         str,
            feature_config: Dict[str, List[str | int] | str],
            combile_models: bool       = True,
            live:           bool       = False,
            preload:        bool       = True,
            db_path:        str        = 'data',
            session_id:     str | None = None,
            inf_interval:   int | None = None,
            trn_interval:   int | None = None,
            market_rep:     List[str]  = ['VTI', 'GLD', 'USO']) -> None:
        
        if not session_id:
            session_id = ticker + f'_{int(time.time())}'

        self._loader = DataLoader(
            session_id     = session_id,
            tickers        = [ticker] + market_rep,
            db_path        = db_path,
            interval       = interval,
            preload        = preload,
            buffer_size    = buffer_size,
            feature_config = feature_config)
        
        self._manager = Manager()

        self._environment = Environment(
            device  = device,
            min_val = 0.5,
            dataset = None)

        self._infer_net = PolicyNet(
            inp_dim=81,
            out_dim=3,
            inp_types=len(TradeType),
            emb_dim=10,
            mem_dim=10,
            mem_size=10,
            mem_heads=2,
            key_dim=10,
            val_dim=50
        ).to(device)
        self._valid_net = PolicyNet(
            inp_dim=81,
            out_dim=3,
            inp_types=len(TradeType),
            emb_dim=10,
            mem_dim=10,
            mem_size=10,
            mem_heads=2,
            key_dim=10,
            val_dim=50
        ).to(device)
        self._train_net = PolicyNet(
            inp_dim=81,
            out_dim=3,
            inp_types=len(TradeType),
            emb_dim=10,
            mem_dim=10,
            mem_size=10,
            mem_heads=2,
            key_dim=10,
            val_dim=50
        ).to(device)
        if combile_models:
            self._compile_models()

        self._train_eps = 100
        self._train_opt = SGD(
            params       = self._train_net.parameters(),
            lr           = 1e-3,
            momentum     = 0.9,
            weight_decay = 0.9)

        self._ticker        = ticker
        self._device        = device
        self._live          = live
        self._inf_interval  = inf_interval if not live else 10 #TODO Convert interval to seconds
        self._trn_interval  = trn_interval if not live else 10 #TODO Convert interval to seconds
        self._buffer_size   = buffer_size
        self._dataset       = deque(maxlen=buffer_size)
        self._actions_queue = deque()

        self._inference_timer  = None
        self._retrain_timer    = None
        self._inference_thread = None
        self._retrain_thread   = None
        self._thread_lock      = threading.Lock()
        self._stop_event       = threading.Event()

    def __del__(self):
        self._stop_timers()

    @property
    def dataset(self) -> List[Dict[str, pd.DataFrame | Data | float | str | int]]:
        return list(self._dataset)

    def start(self) -> None:
        self._fill_dataset()
        self._environment.dataset = self.dataset
        self._start_inference_timer()
        self._start_retrain_timer()

    def _compile_models(self) -> None:
        logging.info('compiling models...')
        self._infer_net = torch.compile(self._infer_net)
        self._valid_net = torch.compile(self._valid_net)
        self._train_net = torch.compile(self._train_net)
        logging.info('model compilation complete')

    def _start_inference_timer(self) -> None:
        self._inference_timer = threading.Thread(
            target = self._timer,
            args   = (
                self._inf_interval, 
                self._inference,))
        self._inference_timer.start()
    
    def _start_retrain_timer(self) -> None:
        self._retrain_timer = threading.Thread(
            target = self._timer,
            args   = (
                self._trn_interval, 
                self._retrain,))
        self._retrain_timer.start()

    def _stop_timers(self) -> None:
        for thread in (self._inference_timer, self._retrain_timer):
            if thread and thread.is_alive():
                self._stop_event.set()
                thread.join()

    def _inference(self) -> None:
        if self._inference_thread and self._inference_thread.is_alive():
            logging.warning('another thread is running inference, wait for it to finish first')
            return
        
        if not self._infer_net:
            logging.warning('model is not ready for inference')
            return
        
        self._inference_thread = threading.Thread(
            target = self._inference_,
            args   = ())
        self._inference_thread.start()

    @torch.no_grad()
    def _inference_(self) -> None:
        #with self._thread_lock:
        data = self._fetch_next(cache=True)
        
        graph      = data['graph']
        price      = data['price']
        index      = data['index']
        ticker     = data['asset']
        close      = price[ticker]['Price']['Close'].mean(None)
        log_return = data['log_return']

        with self._thread_lock:
            market_uuid = self._manager.market_uuid
            positions   = self._manager.positions(close)
            liquid      = self._manager.liquid
        
        uuids       = [position['uuid'] for position in positions]
        types       = [position['type'] for position in positions]
        log_returns = [position['log_return'] for position in positions]

        if liquid:
            uuids.append(market_uuid)
            types.append(TradeType.Market.value)
            log_returns.append(log_return)

        actions, log_probs = select_action(
            model      = self._infer_net,
            state      = graph,
            index      = index,
            inp_types  = types,
            log_return = log_returns,
            device     = self._device,
            argmax     = True)

        for i, choice in enumerate(actions):
            action = Action(choice)
            uuid   = uuids[i]
            final  = Action.Hold
            if action == Action.Buy:
                if uuid == market_uuid:
                    with self._thread_lock:
                        final = self._manager.long(close, np.exp(log_probs[i].item()))

            elif action == Action.Sell:
                if uuid != market_uuid:
                    with self._thread_lock:
                        final = self._manager.short(close, np.exp(log_probs[i].item()), uuid)

            elif action == Action.Hold:
                with self._thread_lock:
                    final = self._manager.hold(close, np.exp(log_probs[i].item()))

            else: # Impossible
                raise NotImplementedError('the selected action is not a valid one')
        
            with self._thread_lock:
                self._actions_queue.append(final)

    def _retrain(self) -> None:
        if self._retrain_thread and self._retrain_thread.is_alive():
            logging.warning('another thread is running retrain, wait for it to finish first')
            return
        
        self._retrain_thread = threading.Thread(
            target = self._retrain_,
            args   = ())
        self._retrain_thread.start()

    def _retrain_(self) -> None:
        logging.info('training starts')
        self._environment.train(
            episodes   = self._train_eps,
            policy_net = self._train_net,
            optimizer  = self._train_opt)
        
        train_score = self._environment.test(self._train_net)
        valid_score = self._environment.test(self._valid_net)

        if train_score > valid_score:
            weights = self._train_net.state_dict()
            with self._thread_lock:
                self._valid_net.load_state_dict(weights)
                self._infer_net.load_state_dict(weights)
            logging.info('model updated')

    def _timer(
            self, 
            interval: int, 
            callback: Callable) -> None:
        
        while not self._stop_event.is_set():
            time.sleep(interval)
            callback()

    def _build_graph(
            self, 
            features:       pd.DataFrame, 
            corr:           pd.DataFrame, 
            corr_threshold: float = 0.5,
            cache:          bool  = False) -> Dict[str, pd.DataFrame | Data | float | str | int]:
        
        cmat = corr.to_numpy()
        cmat[cmat < corr_threshold] = 0
        
        edge_index = np.nonzero(cmat)
        # edge_attrs = cmat[edge_index][None,:] # Not using for now
        edge_index = np.stack(edge_index)

        c1 = features.columns.get_level_values('Type') != 'Price'
        c2 = features.columns.get_level_values('Type') != 'SMA'
        df = features.iloc[-1:, (c1) & (c2)].sort_index(axis=1)
        df = df.stack(level=0, future_stack=True).reset_index(level=0).sort_index(axis=1).drop(columns=['level_0'])
        
        data = {
            'asset':      self._ticker,
            'index':      df.index.get_loc(self._ticker),
            'graph':      Data(
                x = torch.from_numpy(df.values).float(),
                edge_index = torch.from_numpy(edge_index).long().contiguous()),
            'price':      features.iloc[-1:, features.columns.get_level_values('Type') == 'Price'],
            'log_return': np.log(features[self._ticker]['Price']['Close'].dropna().mean(1)).diff(2).iloc[-1]
        }
        
        if cache:
            self._dataset.append(data)
        return data
    
    def _fetch_next(
            self,
            cache: bool = True) -> Dict[str, pd.DataFrame | Data | float | str | int] | None:
        
        if self._live:
            success = self._loader.update_db()
            if not success:
                logging.error(f'no new data available at this time for {self._ticker} from Yahoo Finance API')
                return None

        success = self._loader.load_row()
        if not success:
            logging.error(f'reached last row in db for {self._ticker}')
            return None
        
        features, corr = self._loader.features
        return self._build_graph(
            features = features,
            corr     = corr,
            cache    = cache)
    
    def _fill_dataset(self):
        n = len(self._dataset)
        for i in range(n, self._buffer_size):
            logging.info(f'filling dataset, {i+1}/{self._buffer_size} done')
            _ = self._fetch_next(True)
        logging.info(f'dataset filled')

import time
import torch
import logging
import threading
import numpy as np
import pandas as pd

from collections import deque
from typing import Dict, List
from torch_geometric.data import Data
from backend.loader import DataLoader
from reinforce.environment import Environment


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
            live:           bool         = False,
            preload:        bool         = True,
            db_path:        str          = 'data',
            session_id:     str | None   = None,
            timer_interval: float | None = None,
            market_rep:     List[str]    = ['VTI', 'GLD', 'USO']) -> None:
        
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
        
        self._environment = Environment(
            device  = device,
            min_val = 0.5,
            dataset = None)

        self._ticker         = ticker
        self._device         = device
        self._live           = live
        self._timer_interval = timer_interval if not live else 10 #TODO Convert interval to seconds
        self._buffer_size    = buffer_size
        self._dataset        = deque(maxlen=buffer_size)

    @property
    def dataset(self) -> List[Dict[str, pd.DataFrame | Data | float | str | int]]:
        return list(self._dataset)

    def start(self) -> None:
        self._fill_dataset()
        self._environment.dataset = self.dataset

    def _start_timer(self):
        thread = threading.Thread(
            group  = None,
            target = 0,
            args   = (0,)
        )

    def _timer(self):
        pass

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

import time
import torch
import numpy as np
import pandas as pd

from typing import Dict, List
from torch_geometric.data import Data
from backend.loader import DataLoader


class Session:
    def __init__(
            self,
            ticker:         str,
            interval:       str,
            buffer_size:    int,
            device:         str,
            feature_config: Dict[str, List[str | int] | str],
            live:           bool       = False,
            db_path:        str        = "data",
            session_id:     str | None = None,
            market_rep:     List[str]  = []) -> None:
        
        if not session_id:
            session_id = ticker + f'_{int(time.time())}'

        self._loader = DataLoader(
            session_id     = session_id,
            tickers        = [ticker] + market_rep,
            db_path        = db_path,
            interval       = interval,
            buffer_size    = buffer_size,
            feature_config = feature_config)
        
        self._device = device
        self._live   = live

    def _build_graph(self, features: pd.DataFrame, corr: pd.DataFrame, corr_threshold: float=0.5) -> Data:
        cmat = corr.to_numpy()
        cmat[cmat < corr_threshold] = 0
        
        edge_index = np.nonzero(cmat)
        # edge_attrs = cmat[edge_index][None,:] # Not using for now
        edge_index = np.stack(edge_index)

        c1 = features.columns.get_level_values('Type') != 'Price'
        c2 = features.columns.get_level_values('Type') != 'SMA'
        df = features.iloc[-1:, (c1) & (c2)].sort_index(axis=1)
        df = df.stack(level=0, future_stack=True).reset_index(level=0).sort_index(axis=1).drop(columns=['level_0'])
        
        return Data(
            x          = torch.from_numpy(df.values).float(),
            edge_index = torch.from_numpy(edge_index).long().contiguous())
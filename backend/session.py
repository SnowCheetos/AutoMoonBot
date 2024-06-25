import time
import numpy as np
import pandas as pd

from typing import Dict, List
from loader import DataLoader


class Session:
    def __init__(
            self,
            ticker:         str,
            interval:       str,
            buffer_size:    int,
            feature_config: Dict[str, List[str | int] | str],
            live:           bool       = False,
            db_path:        str        = "data",
            session_id:     str | None = None,
            market_rep:     List[str]  = []) -> None:
        
        if not session_id:
            session_id = ticker + f'_{int(time.time())}'

        self._loader = DataLoader(
            session_id     = session_id,
            tickers        = ticker + market_rep,
            db_path        = db_path,
            interval       = interval,
            buffer_size    = buffer_size,
            feature_config = feature_config)
        
        self._live = live

    def build_graph(self, features: pd.DataFrame, corr: pd.DataFrame, corr_threshold: float=0.5):
        cmat = corr.to_numpy() - corr_threshold
        edge_indices = ...
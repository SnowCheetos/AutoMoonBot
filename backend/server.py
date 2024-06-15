from typing import Dict, List

from backend.buffer import DataBuffer
from reinforce.environment import TradeEnv, train
from reinforce.model import PolicyNet, inference

class Server:
    def __init__(
            self, 
            ticker: str,
            period: str,
            interval: str,
            queue_size: int,
            feature_dim: int,
            feature_params: Dict[str, List[int] | Dict[str, List[int]]]) -> None:
        
        self._buffer = DataBuffer(ticker, period, interval, queue_size, feature_params)
        self._model = PolicyNet

    def run_inference(self):
        pass
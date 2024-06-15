import numpy as np
from enum import Enum
from typing import List

class Position(Enum):
    Cash  = 0
    Asset = 1

def compute_sharpe_ratio(returns: List[float], risk_free_rate: float) -> float:
    returns = np.asarray(returns)
    excess_returns = returns - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)
    if std_excess_return == 0:
        return 0
    
    return mean_excess_return / std_excess_return
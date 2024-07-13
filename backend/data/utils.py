import dateparser
import numpy as np
from enum import Enum, auto
from typing import List
from dateutil.relativedelta import relativedelta

class Tense(Enum):
    Past = auto()
    Present = auto()
    Future = auto()


class Aspect(Enum):
    Simple = auto()
    Perfect = auto()
    Continuous = auto()
    PerfectContinuous = auto()

def get_all_months(start_date: str, end_date: str) -> List[str]:
    start = dateparser.parse(start_date)
    end = dateparser.parse(end_date)

    months = []
    current = start
    while current <= end:
        months.append(current.strftime("%Y-%m"))
        current += relativedelta(months=1)
    
    return months

def compute_time_decay(
    start: float, end: float, shift: float = 7, alpha: float = 0.5
) -> float:
    if start > end:
        return 0

    sigmoid = lambda x, alpha: 1 / (1 + np.exp(-alpha * x))

    time_delta = max(end - start, 1e-3)
    log_time = np.log(time_delta)
    shifted = log_time - shift
    return 1 - sigmoid(shifted, alpha)
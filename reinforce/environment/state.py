from trading import Category
from dataclasses import dataclass

@dataclass
class State:
    category: Category
    potential: float
    price: float
    log_return: float
    pid: str | None = None
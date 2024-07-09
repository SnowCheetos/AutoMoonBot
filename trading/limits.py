from trading import Category


class StopLoss:
    def __init__(
        self,
        category:  Category,
        price:     float,
        variance:  float,
        tolerance: float
    ) -> None:
        
        self._risk = tolerance * variance


class TakeProfit:
    def __init__(
        self,
        category:  Category,
        price:     float,
        variance:  float,
        tolerance: float
    ) -> None:
        
        self._risk = tolerance * variance
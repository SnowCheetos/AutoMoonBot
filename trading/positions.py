
from trading import Category, Condition


class Position:
    '''
    Represents a position, can be options, shares, shorts
    '''
    def __init__(
        self, 
        category   : Category,
        price      : float,
        size       : float,
        cost       : float,
        variance   : float,
        expiration : float | None = None
    ) -> None:
        
        self.category  = category
        self.condition = Condition.CREATED

        self._price      = price
        self._size       = size
        self._cost       = cost
        self._variance   = variance
        self._expiration = expiration
        self._entry      = None
        self._exit       = None
        self._value      = None

    def value(
        self, 
        price: float
    ) -> float:
        if self._entry is None:
            return 0
        else:
            return self._size * (price / self._entry)

    def update(
        self, 
        price:    float,
        variance: float,
    ) -> None:
        self._price    = price
        self._variance = variance

    def open(
        self, 
        prob:  float
    ) -> None:
        if self.condition == Condition.CREATED:
            self.condition = Condition.OPENED
            self._entry    = self._price
        
        else:
            pass

    def close(
        self, 
        prob:  float
    ) -> None:
        if self.condition == Condition.OPENED:
            self.condition = Condition.CLOSED
            self._exit     = self._price
        
        elif self.condition == Condition.CREATED:
            self.condition = Condition.CANCELLED
        
        else:
            pass
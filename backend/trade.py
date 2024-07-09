import time
import uuid
import numpy as np
from enum import Enum
from typing import Dict, List


class Action(Enum):
    '''
    An abstraction to the actions allowed in the environment
    '''
    Buy:  int = 0
    Hold: int = 1
    Sell: int = 2


class TradeStatus(Enum):
    '''
    An abstraction to the states a trade can have
    '''
    Opened: int = 0
    Closed: int = 1


class TradeType(Enum):
    '''
    An abstraction to the types of trades
    '''
    Market:    int = 0
    Long:      int = 1
    Short:     int = 2
    LongCall:  int = 3
    LongPut:   int = 4
    ShortCall: int = 5
    ShortPut:  int = 6


class Trade:
    '''
    This class implements the attributes and methods that goes into a single position
    '''
    def __init__(
            self, 
            uuid:       str,
            trade_type: TradeType,
            cost:       float) -> None:

        self._uuid    = uuid
        self._type    = trade_type
        self._cost    = cost
        self._status  = TradeStatus.Opened
        self._opened  = None
        self._closed  = None
        self._entry   = 0
        self._exit    = 0
        self._holding = 0

    @property
    def type(self) -> TradeType:
        return self._type

    @property
    def uuid(self) -> str:
        return self._uuid

    @property
    def status(self) -> TradeStatus:
        return self._status
    
    @property
    def gain(self) -> float | None:
        if self._closed:
            return (self._exit / self._entry) * self._holding
        else:
            return None
    
    def open(
            self, 
            price:  float, 
            amount: float) -> None:

        assert price > 0, 'asset prices must be strictly positive'

        if self._type == TradeType.Long:
            self._status  = TradeStatus.Opened
            self._opened  = time.time()
            self._entry   = price
            self._holding = amount
        else:
            raise NotImplementedError('short and options are not implemented yet')

    def close(
            self, 
            price: float) -> float | None:
        
        if self._status == TradeStatus.Opened:
            self._status  = TradeStatus.Closed
            self._closed  = time.time()
            self._exit    = price
            return (self._exit / self._entry - self._cost) * self._holding
        else:
            return None
    
    def value(
            self,
            price:    float, 
            absolute: bool = False) -> float | None:
        
        if self._status == TradeStatus.Opened:
            return (price / self._entry) * (1 if absolute else self._holding)
        else:
            return None


class Account:
    def __init__(self, cost: float) -> None:
        self._cost = cost
        self._market_uuid = str(uuid.uuid4())
        self._opened: Dict[str, Trade] = {}
        self._closed: Dict[str, Trade] = {}

    @property
    def market_uuid(self) -> str:
        return self._market_uuid

    @property
    def history(self) -> Dict[str, Trade]:
        return self._closed

    def reset(self):
        self._opened = {}
        self._closed = {}

    def open_positions(
            self, 
            price: float) -> List[Dict[str, str | int | float]]:
        
        result = []
        for trade in self._opened.values():
            result.append({
                'uuid':       trade.uuid,
                'type':       trade.type.value,
                'log_return': np.log(trade.value(price, True))
            })
        return result

    def total_value(
            self, 
            price: float) -> float:
        
        values = 0
        for trade in self._opened.values():
            value   = trade.value(price, False)
            values += value if value else 0
        return values

    def open(
            self, 
            trade_type: TradeType,
            price:      float,
            amount:     float) -> None:
        
        trade = Trade(
            uuid       = str(uuid.uuid4()),
            trade_type = trade_type,
            cost       = self._cost)
        
        trade.open(price, amount)
        self._opened[trade.uuid] = trade
    
    def close(
            self, 
            price:    float, 
            trade_id: str) -> float | None:
        
        trade = self._opened.pop(trade_id, None)
        if trade is None: 
            return None

        value = trade.close(price)
        self._closed[trade_id] = trade
        return value
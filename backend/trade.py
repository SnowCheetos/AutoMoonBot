import time
import uuid
import numpy as np
from enum import Enum
from typing import Dict
from collections import deque


class TradeStatus(Enum):
    Opened: int = 0
    Closed: int = 1


class TradeType(Enum):
    Market:    int = 0
    Long:      int = 1
    Short:     int = 2
    LongCall:  int = 3
    LongPut:   int = 4
    ShortCall: int = 5
    ShortPut:  int = 6


class Trade:
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
    def uuid(self) -> str:
        return self._uuid

    @property
    def status(self) -> TradeStatus:
        return self._status
    
    def open(self, price: float, amount: float) -> None:
        if self._type == TradeType.Long:
            self._status  = TradeStatus.Opened
            self._opened  = time.time()
            self._entry   = price
            self._holding = amount
        else:
            raise NotImplementedError('short and options are not implemented yet')

    def close(self, price: float) -> float | None:
        if self._status == TradeStatus.Opened:
            self._status  = TradeStatus.Closed
            self._closed  = time.time()
            self._exit    = price
            self._holding = 0
            return (self._exit / self._entry - self._cost) * self._holding
        else:
            return None
    
    def value(self, price: float) -> float | None:
        if self._status == TradeStatus.Opened:
            return (price / self._entry) * self._holding
        else:
            return None


class Account:
    def __init__(self) -> None:
        self._opened = deque()
        self._closed = deque()

    def reset(self):
        self._opened = deque()
        self._closed = deque()

    def value(
            self, 
            price:    float,
            logscale: bool = False) -> Dict[str, float | None]:
        
        values = {}
        for trade in self._opened:
            val = trade.value(price)
            if val is not None:
                if logscale:
                    val = np.log(val)
                values[trade.uuid] = val
        return values

    def open(
            self, 
            trade_type: TradeType,
            price:      float,
            amount:     float,
            cost:       float) -> str:
        
        trade = Trade(
            uuid       = str(uuid.uuid4()),
            trade_type = trade_type,
            cost       = cost)
        
        trade.open(price, amount)
        self._opened.append(trade)
        return trade.uuid
    
    def close(
            self, 
            price: float, 
            uid:   str | None) -> float:
        
        if uid is None and len(self._opened) > 0:
            # If no trade specified, close the oldest trade
            trade = self._opened.popleft()
        else:
            trade = None
            for t in self._opened:
                if t.uuid == uid:
                    trade = t
            if trade is None:
                return 1

        gain  = trade.close(price)
        self._closed.append(trade)
        
        if gain is None:
            return 1
        else:
            return gain
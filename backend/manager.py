from typing import Dict, List

import numpy as np
from backend.trade import Account, TradeType, Action


class Manager:
    '''
    This class is an account manager that abstracts a lot of the trading functionalities.
    '''
    def __init__(
            self, 
            risk_free_rate:  float = 1.05,
            min_probability: float = 0.34,
            max_trade_size:  float = 0.5,
            min_balance:     float = 0,
            trading_cost:    float = 0.005) -> None:
        
        self._value           = 1
        self._balance         = 1
        self._risk_free_rate  = risk_free_rate
        self._min_probability = min_probability
        self._max_trade_size  = max_trade_size
        self._min_balance     = min_balance
        self._account         = Account(cost=trading_cost)

    @property
    def liquid(self) -> bool:
        return self._balance > self._min_balance

    @property
    def value(self) -> float:
        return self._value

    @property
    def market_uuid(self) -> float:
        return self._account.market_uuid

    @property
    def sharpe_ratio(self) -> float:
        history = self._account.history
        gains = []
        for trade in history.values():
            gains.append(trade.gain - 1)
        if len(gains) < 2:
            return 0.5
        excess_return = self.value - self._risk_free_rate
        return excess_return / (np.std(gains) + 1e-9)

    def reset(self) -> None:
        self._value    = 1
        self._balance  = 1
        self._account.reset()

    def positions(
            self, 
            price: float) ->  List[Dict[str, str | int | float]]:
        
        return self._account.open_positions(price)

    def _update_value(
            self, 
            price: float) -> None:
        
        self._value = self._account.total_value(price) + self._balance

    def long(
            self, 
            price: float, 
            prob:  float) -> Action:
        
        if self._balance > self._min_balance and prob > self._min_probability:
            amount = min(self._max_trade_size, prob)
            self._account.open(TradeType.Long, price, amount)
            self._balance -= amount
            self._update_value(price)
            return Action.Buy
        else:
            self._update_value(price)
            return Action.Hold

    def short(
            self, 
            price:    float, 
            prob:     float,
            trade_id: str) -> Action:
        
        if prob > self._min_probability:
            value = self._account.close(price, trade_id)
            if value is not None:
                self._balance += value
                self._update_value(price)
                return Action.Sell
        return Action.Hold

    def hold(
            self, 
            price: float, 
            prob:  float) -> Action:
        
        self._update_value(price)
        return Action.Hold
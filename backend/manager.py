from typing import Dict, List
from backend.trade import Account, TradeType, Action


class Manager:
    '''
    This class is an account manager that abstracts a lot of the trading functionalities.
    '''
    def __init__(
            self, 
            min_probability: float = 0.34,
            max_trade_size:  float = 0.5,
            min_balance:     float = 0) -> None:
        
        self._value           = 1
        self._balance         = 1
        self._min_probability = min_probability
        self._max_trade_size  = max_trade_size
        self._min_balance     = min_balance
        self._account         = Account()

    @property
    def liquid(self) -> bool:
        return self._balance > self._min_balance

    @property
    def value(self) -> float:
        return self._value

    @property
    def market_uuid(self) -> float:
        return self._account.market_uuid

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
            self._account.open(TradeType.Long, price, amount, 0)
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
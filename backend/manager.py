from collections import deque
from typing import Dict
from utils.trading import Position, Trade, Action, Signal
from backend.trade import Account

class TradeManager:
    '''
    To be replaced by Manager
    '''
    def __init__(
            self,
            cov:       float,
            alpha:     float,
            gamma:     float,
            cost:      float = 0,
            full_port: bool  = False,
            leverage:  float = 0,
            qk_sell:   bool = False) -> None:
        
        self._trade      = Trade(cov, alpha, gamma, cost, full_port, leverage, qk_sell)
        self._position   = Position.Cash
        self._full_port  = full_port
        self._portfolio  = 1.0
        self._returns    = []
        self._prev_exit  = 0.0
        self._trade_hist = deque(maxlen=10)

    @property
    def position(self) -> Position:
        return self._position

    @property
    def cash(self) -> bool:
        return self._position == Position.Cash
    
    @property
    def partial(self) -> bool:
        return self._position == Position.Partial
    
    @property
    def asset(self) -> bool:
        return self._position == Position.Asset

    @property
    def signal(self) -> Signal:
        return self._trade.signal

    @property
    def portfolio(self):
        return self._portfolio

    @property
    def returns(self):
        return self._returns

    @property
    def prev_exit(self):
        return self._prev_exit

    @property
    def curr_trade(self) -> Trade:
        return self._trade

    @property
    def last_trade(self) -> Dict[str, float] | None:
        return self._trade_hist[-1]

    def log_trade(self):
        self._trade_hist.append(self._trade.data)

    def try_buy(self, price: float, cov: float, amount: float | None=None) -> Action:
        self._trade.risk = cov
        if self._trade.opened:
            if self.partial and not self._full_port:
                # double down
                action = self._trade.double(price)
                if action == Action.Double: 
                    self._position = Position.Asset
                    self._trade.hold()
                    return Action.Double
                elif action == Action.Sell:
                    self._portfolio = Position.Cash
                    self._portfolio = 0
                    self._trade.hold()
                    return Action.Sell
        else:
            # new trade
            success = self._trade.open(price, amount)
            if success: 
                self._position = Position.Partial
                self._trade.hold()
                return Action.Buy
        return Action.Hold

    def try_sell(self, price: float, cov: float) -> float:
        self._trade.risk = cov
        if self._trade.opened:
            gain = self._trade.close(price)
            if gain > 0: 
                self._position   = Position.Cash
                self._portfolio *= gain
                self._returns   += [gain]
                self._prev_exit  = price
                self.log_trade()
                self._trade.hold()
            return gain
        return -1

    def hold(self, cov: float):
        self._trade.risk = cov
        if self._trade.opened:
            self._trade.hold()

    def potential_gain(self, price: float) -> float:
        return self._trade.potential_gain(price)


class Manager:
    def __init__(
            self, 
            min_balance: float = 0) -> None:
        
        self._value       = 1
        self._balance     = 1
        self._min_balance = min_balance
        self._account     = Account()
        self._position    = Position.Cash

    @property
    def value(self) -> float:
        return self._value

    @property
    def position(self) -> Position:
        return self._position

    def reset(self) -> None:
        self._value    = 1
        self._balance  = 1
        self._position = Position.Cash
        self._account.reset()

    def _update_value(self, price: float) -> None:
        values = self._account.value(price)
        if len(values) > 0:
            self._value = sum(values.values()) + self._balance

    def long(self, price: float, prob: float) -> Action:
        if self._balance > self._min_balance:
            ...
        else:
            self._update_value(price)
        return Action.Hold

    def short(self, price: float, prob: float) -> Action:
        if self._position == Position.Asset:
            ...
        elif self._position == Position.Partial:
            ...
        else:
            self._update_value(price)
        return Action.Hold

    def hold(self, price: float, prob: float) -> Action:
        self._update_value(price)
        return Action.Hold
from typing import Dict
from collections import deque

from utils import Position, Trade, Action, Signal


class TradeManager:
    def __init__(
            self,
            cov:       float,
            alpha:     float,
            gamma:     float,
            cost:      float = 0,
            full_port: bool  = False,
            leverage:  float = 0) -> None:
        
        self._trade      = Trade(cov, alpha, gamma, cost, full_port, leverage)
        self._position   = Position.Cash
        self._full_port  = full_port
        self._portfolio  = 1.0
        self._returns    = []
        self._prev_exit  = 0.0
        self._trade_hist = deque(maxlen=10)

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
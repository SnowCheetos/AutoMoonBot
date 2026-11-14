"""PGPerceptron's trade bookkeeping, ported literally.

``Status``/``Trade``/``TradeManager`` from the original ``utils/trading.py``
and ``backend/manager.py``. The logic is reproduced as-is, quirks included,
because it *is* the strategy: an order only executes when the price at a
repeated request has moved outside a band (``gamma * coef_of_var`` up, or
``alpha`` times that down, for a buy) around the price at the previous
request, which acts as a volatility-scaled confirmation filter. Positions
are cash, half (``Partial``) or doubled (``Asset``).

Only one change: leverage above 1 is rejected, since that path of the
original set the portfolio value to 0 (``self._portfolio = 0``) on a
losing double-down.
"""

import math
from collections import deque
from dataclasses import dataclass
from enum import IntEnum


class Position(IntEnum):
    Cash = 0
    Asset = 1
    Partial = 2


class Action(IntEnum):
    Buy = 0
    Hold = 1
    Sell = 2
    Double = 3


class Signal(IntEnum):
    Buy = 0
    Idle = 1
    Sell = 2


class Status:
    __slots__ = ("alpha", "risk", "signal", "stop_loss", "take_profit")

    def __init__(self, risk: float, alpha: float) -> None:
        self.signal = Signal.Idle
        self.take_profit = math.nan
        self.stop_loss = math.nan
        self.risk = risk
        self.alpha = alpha

    def reset(self) -> None:
        self.take_profit = math.nan
        self.stop_loss = math.nan
        self.signal = Signal.Idle

    def set_signal(self, action: int) -> None:
        signal = Signal(action)
        if abs(signal - self.signal) == 2:  # buy <-> sell
            self.reset()
        else:
            self.signal = signal

    def set_band(self, close: float) -> None:
        if self.signal == Signal.Buy:
            self.take_profit = close * (1 - self.alpha * self.risk)
            self.stop_loss = close * (1 + self.risk)
        elif self.signal == Signal.Sell:
            self.take_profit = close * (1 + self.risk)
            self.stop_loss = close * (1 - self.alpha * self.risk)

    def confirm_buy(self, close: float) -> bool:
        return self.signal == Signal.Buy and (close >= self.stop_loss or close <= self.take_profit)

    def confirm_sell(self, close: float) -> bool:
        return self.signal == Signal.Sell and (close <= self.stop_loss or close >= self.take_profit)


@dataclass(frozen=True)
class TradeRecord:
    entry: float
    exit: float
    amount: float


class Trade:
    __slots__ = ("amount", "cost", "entry", "gamma", "leverage", "opened", "status")

    def __init__(
        self, cov: float, alpha: float, gamma: float, cost: float, leverage: float
    ) -> None:
        self.opened = False
        self.cost = cost
        self.gamma = gamma
        self.entry = 0.0
        self.amount = 0.0
        self.leverage = leverage
        self.status = Status(cov * gamma, alpha)

    def set_risk(self, cov: float) -> None:
        self.status.risk = self.gamma * cov

    def _rearm(self, action: int, price: float) -> None:
        self.status.set_signal(action)
        self.status.set_band(price)

    def open(self, price: float, amount: float | None) -> bool:
        amt = 0.5 if not amount else min(0.5, amount)
        if self.status.confirm_buy(price):
            self.opened, self.entry, self.amount = True, price, amt
            return True
        self._rearm(Signal.Buy, price)
        return False

    def double(self, price: float) -> bool:
        if self.status.confirm_buy(price):
            self.entry = 0.5 * (self.entry + price)
            self.amount *= 2
            return True
        self._rearm(Signal.Buy, price)
        return False

    def hold(self) -> None:
        self.status.set_signal(Signal.Idle)

    def close(self, price: float) -> float:
        if self.status.confirm_sell(price):
            self.opened = False
            gain = (price / self.entry - 1) * (self.leverage - self.cost)
            return max(0.0, gain * self.amount + 1)
        self._rearm(Signal.Sell, price)
        return -1.0

    def potential_gain(self, price: float) -> float:
        if self.opened:
            gain = (price / self.entry - 1) * (self.leverage - self.cost)
            return max(0.0, gain * self.amount + 1)
        return 0.0


class TradeManager:
    def __init__(
        self, cov: float, alpha: float, gamma: float, cost: float = 0.0, leverage: float = 0.0
    ) -> None:
        if not 0.0 <= leverage <= 1.0:
            raise ValueError("leverage must be in [0, 1]")
        self.trade = Trade(cov, alpha, gamma, cost, leverage)
        self.position = Position.Cash
        self.portfolio = 1.0
        self.returns: list[float] = []
        self.prev_exit = 0.0
        self.history: deque[TradeRecord] = deque(maxlen=10)

    @property
    def last_trade(self) -> TradeRecord:
        return self.history[-1]

    def try_buy(self, price: float, cov: float, amount: float | None = None) -> Action:
        self.trade.set_risk(cov)
        if self.trade.opened:
            if self.position == Position.Partial and self.trade.double(price):
                self.position = Position.Asset
                self.trade.hold()
                return Action.Double
        elif self.trade.open(price, amount):
            self.position = Position.Partial
            self.trade.hold()
            return Action.Buy
        return Action.Hold

    def try_sell(self, price: float, cov: float) -> float:
        self.trade.set_risk(cov)
        if not self.trade.opened:
            return -1.0
        gain = self.trade.close(price)
        if gain > 0:
            self.position = Position.Cash
            self.portfolio *= gain
            self.returns.append(gain)
            self.prev_exit = price
            self.history.append(TradeRecord(self.trade.entry, price, self.trade.amount))
            self.trade.hold()
        return gain

    def potential_gain(self, price: float) -> float:
        return self.trade.potential_gain(price)

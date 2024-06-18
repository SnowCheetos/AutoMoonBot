from enum import Enum

class Position(Enum):
    Cash    = 0
    Asset   = 1
    Partial = 2

class Action(Enum):
    Buy    = 0
    Hold   = 1
    Sell   = 2
    Double = 3

class Signal(Enum):
    Buy  = 0
    Idle = 1
    Sell = 2

class Status:
    def __init__(
            self, 
            max_risk: float, 
            alpha:    float) -> None:
        
        self._signal      = Signal.Idle
        self._take_profit = float("nan")
        self._stop_loss   = float("nan")
        self._max_risk    = max_risk
        self._alpha       = alpha

    @property
    def signal(self):
        return self._signal

    @property
    def take_profit(self):
        return self._take_profit
    
    @property
    def stop_loss(self):
        return self._stop_loss
    
    def reset(
            self, 
            max_risk: float | None=None,
            alpha:    float | None=None):
        self._take_profit = float("nan")
        self._stop_loss   = float("nan")
        self._signal      = Signal.Idle
        
        if max_risk:
            self._max_risk = max_risk
        if alpha:
            self._alpha = alpha

    @signal.setter
    def signal(self, action: int):
        signal = Signal(action)
        if abs(signal.value - self._signal.value) == 2:
            self.reset()
            return
        else:
            self._signal = signal
            return

    @take_profit.setter
    def take_profit(self, close: float):
        if self._signal == Signal.Buy:
            self._take_profit = close * (1 - self._alpha * self._max_risk)
        elif self._signal == Signal.Sell:
            self._take_profit = close * (1 + self._max_risk)

    @stop_loss.setter
    def stop_loss(self, close: float):
        if self._signal == Signal.Buy:
            self._stop_loss = close * (1 + self._max_risk)
        elif self._signal == Signal.Sell:
            self._stop_loss = close * (1 - self._alpha * self._max_risk)

    def confirm_buy(self, close: float) -> bool:
        if self._signal != Signal.Buy:
            return False
        if close >= self._stop_loss or close <= self._take_profit:
            return True
        return False
    
    def confirm_sell(self, close: float) -> bool:
        if self._signal != Signal.Sell:
            return False
        if close <= self._stop_loss or close >= self._take_profit:
            return True
        return False
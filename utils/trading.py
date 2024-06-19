from enum import Enum
from typing import Dict

class Position(Enum):
    """
    Enum that holds the possible position states
    """
    Cash    = 0 # holding all cash
    Asset   = 1 # holding all assets
    Partial = 2 # holding 50/50

class Action(Enum):
    """
    Enum that holds the possible actions
    """
    Buy    = 0 # buy 50%
    Hold   = 1 # do nothing
    Sell   = 2 # sell asset
    Double = 3 # all in

class Signal(Enum):
    """
    Enum that holds the possible interpreted signals
    """
    Buy    = 0 # buy 50%
    Idle   = 1 # do nothing
    Sell   = 2 # sell all

class Status:
    """
    Holds the current trade conditions
    """
    def __init__(
            self, 
            risk:  float, # allowed price movement for entry / exit
            alpha: float  # take profit / stop loss ratio
        ) -> None:
        
        self._signal      = Signal.Idle  # current signal
        self._take_profit = float("nan") # current take profit amount
        self._stop_loss   = float("nan") # current stop loss amount
        self._risk        = risk         # risk level
        self._alpha       = alpha        # tp:sl ratio

    @property
    def signal(self):
        return self._signal

    @property
    def take_profit(self):
        return self._take_profit
    
    @property
    def stop_loss(self):
        return self._stop_loss
    
    @property
    def risk(self) -> float:
        return self._risk

    @risk.setter
    def risk(self, r: float):
        self._risk = r

    def reset(
            self, 
            risk:  float | None=None,
            alpha: float | None=None):
        self._take_profit = float("nan")
        self._stop_loss   = float("nan")
        self._signal      = Signal.Idle
        if risk:
            self._risk = risk
        if alpha:
            self._alpha = alpha

    @signal.setter
    def signal(self, action: int):
        signal = Signal(action)
        if abs(signal.value - self._signal.value) == 2:
            # buy to sell or sell to buy
            self.reset()
        else:
            self._signal = signal

    @take_profit.setter
    def take_profit(self, close: float):
        if self._signal == Signal.Buy:
            self._take_profit = close * (1 - self._alpha * self._risk)
        elif self._signal == Signal.Sell:
            self._take_profit = close * (1 + self._risk)

    @stop_loss.setter
    def stop_loss(self, close: float):
        if self._signal == Signal.Buy:
            self._stop_loss = close * (1 + self._risk)
        elif self._signal == Signal.Sell:
            self._stop_loss = close * (1 - self._alpha * self._risk)

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

class Trade:
    """
    Class represents a single trade
    """
    def __init__(
            self,
            cov:       float,
            alpha:     float,
            gamma:     float,
            cost:      float=0,
            full_port: bool=False) -> None:
        
        self._open      = False
        self._cost      = cost
        self._alpha     = alpha
        self._gamma     = gamma
        self._full_port = full_port
        self._entry     = 0
        self._exit      = 0
        self._amount    = 0
        self._status    = Status(cov * gamma, alpha)

    @property
    def data(self) -> Dict[str, float]:
        return {
            "entry":  self._entry,
            "exit":   self._exit,
            "amount": self._amount
        }

    @property
    def opened(self) -> bool:
        return self._open

    @property
    def status(self) -> Status:
        return self._status

    @property
    def signal(self) -> Signal:
        return self._status.signal
    
    @property
    def amount(self) -> float:
        return self._amount

    @property
    def risk(self) -> float:
        return self.status.risk
    
    @signal.setter
    def signal(self, action: int):
        self._status.signal = action

    @risk.setter
    def risk(self, cov: float):
        self.status.risk = self._gamma * cov

    def open(self, price: float, amount: float | None=None) -> bool:
        """
        Tries to open the trade, amount as portfolio percentage from 0 to 1
        """
        if not amount: 
            amount = 0.5
        else:
            amount = min(0.5, amount)
        if self.status.confirm_buy(price):
            self._open   = True
            self._entry  = price
            self._amount = amount if not self._full_port else 1.0
            return True
        
        self.signal             = 0
        self.status.take_profit = price
        self.status.stop_loss   = price
        return False

    def double(self, price: float) -> bool:
        """
        Double down
        """
        if self.status.confirm_buy(price) and not self._full_port:
            self._entry  = 0.5 * (self._entry + price)
            self._amount *= 2
            return True
        
        if not self._full_port:
            self.signal             = 0
            self.status.take_profit = price
            self.status.stop_loss   = price
        return False

    def hold(self):
        """
        Set signal to hold
        """
        self.signal = 1

    def close(self, price: float) -> float:
        """
        Tries to close
        """
        if self.status.confirm_sell(price):
            self._open = False
            gain       = price / self._entry - self._cost
            net_gain   = (gain - 1) * self._amount + 1
            return max(0, net_gain)
        self.signal             = 2
        self.status.take_profit = price
        self.status.stop_loss   = price
        return -1
    
    def potential_gain(self, price: float) -> float:
        """
        Computes the potential gain
        """
        if self.opened:
            gain     = price / self._entry - self._cost
            net_gain = (gain - 1) * self._amount + 1
            return max(0, net_gain)
        return 0
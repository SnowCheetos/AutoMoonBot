import pytest

from backend import TradeManager
from utils.trading import Action, Signal

@pytest.fixture
def manager():
    return TradeManager(1, 1, 1, 0)

def test_manager(manager: TradeManager):
    action = manager.try_buy(1, 0)

    assert action         == Action.Hold, "manager did not hold"
    assert manager.signal == Signal.Buy,  "manager signal is not buy"

    action = manager.try_buy(1, 0)

    assert action         == Action.Buy,  "manager did not buy"
    assert manager.signal == Signal.Idle, "manager signal is not idle"

    gain = manager.try_sell(2, 0)

    assert gain           <  0,           "manager sold early"
    assert manager.signal == Signal.Sell,  "manager signal is not sell"

    gain        = manager.try_sell(2, 0)
    actual_gain = 1.5

    assert gain           == actual_gain,  "manager did not sell"
    assert manager.signal == Signal.Idle,  "manager signal is not idle"

    action = manager.try_buy(1, 0)

    assert action         == Action.Hold, "manager did not hold"
    assert manager.signal == Signal.Buy,  "manager signal is not buy"

    action = manager.try_buy(1, 0)

    assert action         == Action.Buy,  "manager did not buy"
    assert manager.signal == Signal.Idle, "manager signal is not idle"

    action = manager.try_buy(1, 0)

    assert action         == Action.Hold, "manager bought early"
    assert manager.signal == Signal.Buy,  "manager signal is not idle"

    action = manager.try_buy(1, 0)

    assert action         == Action.Double, "manager did not double"
    assert manager.signal == Signal.Idle,    "manager signal is not idle"

    gain = manager.try_sell(2, 0)

    assert gain           <  0,           "manager sold early"
    assert manager.signal == Signal.Sell,  "manager signal is not sell"

    gain        = manager.try_sell(2, 0)
    actual_gain = 2.0

    assert gain           == actual_gain,  "manager did not sell"
    assert manager.signal == Signal.Idle,  "manager signal is not idle"
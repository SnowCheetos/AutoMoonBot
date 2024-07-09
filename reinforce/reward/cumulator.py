from typing import List
from reinforce.environment import State
from trading import Portfolio, Category, Status, Action


class Cumulator:
    def __init__(self, price: float, gamma: float) -> None:
        self._portfolio = Portfolio(price, 0)
        self._rewards = []
        self._gamma = gamma

    @property
    def discounted_rewards(self) -> List[float]:
        discounted_rewards = []
        cumulative = 0
        for reward in reversed(self._rewards):
            cumulative = reward + self._gamma * cumulative
            discounted_rewards.insert(0, cumulative)
        return discounted_rewards

    def reset(self, price: float) -> None:
        self._portfolio.reset(price)
        self._rewards = []

    def step(self, state: State, action: Action) -> bool:
        self._portfolio.update(state.price)
        if action == Action.ENTER:
            if state.category == Category.MARKET:
                status = self._portfolio.open(Category.LONG, 1)
                if status == Status.SUCCESS:
                    self._rewards.append(state.potential)
                elif status == Status.NOFUNDS:
                    self._rewards.append(-state.potential)
                else:
                    pass
            elif state.category == Category.LONG:
                self._rewards.append(-state.potential)
            else:
                pass

        elif action == Action.EXIT:
            if state.category == Category.MARKET:
                self._rewards.append(-state.potential)
            elif state.category == Category.LONG:
                status = self._portfolio.close(state.pid, 1)
                if status == Status.SUCCESS:
                    self._rewards.append(state.log_return)
                else:
                    pass
            else:
                pass

        elif action == Action.IDLE:
            if state.category == Category.MARKET:
                self._rewards.append(-state.potential)
            elif state.category == Category.LONG:
                self._rewards.append(-state.log_return)
            else:
                pass

        else:
            raise NotImplementedError("the selected action is not a valid one")
        
        if self._portfolio.value() < 0.1:
            return False
        else:
            return True

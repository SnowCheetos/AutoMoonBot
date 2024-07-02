from reinforce import Cumulator

class Environment:
    def __init__(self) -> None:
        self._cumulator = Cumulator(0, 0.99)

    def reset(self):
        pass

    def step(self):
        pass
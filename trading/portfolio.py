from trading import Positions, Category, Status


class Portfolio:
    def __init__(self, price: float, cost: float) -> None:
        self._positions = Positions(price)
        self._balance = 1.0
        self._price = price
        self._cost = cost

    def reset(self, price: float) -> None:
        self._positions.reset(price)
        self._balance = 1.0
        self._price = price

    def update(self, price: float) -> None:
        self._price = price
        self._positions.update(price)

    def value(self) -> float:
        return self._balance + self._positions.value()

    def open(self, category: Category, prob: float) -> Status:
        size = prob
        cost = size * self._price * (1 + self._cost)
        if cost > self._balance:
            return Status.NOFUNDS
        else:
            status = self._positions.open(category, size)
            if status == Status.SUCCESS:
                self._balance -= cost
            return status

    def close(self, pid: str, prob: float) -> Status:
        status = self._positions.close(pid)
        if status == Status.SUCCESS:
            value = self._positions.fetch_closed(pid)
            self._balance += (1 - self._cost) * value
        return status
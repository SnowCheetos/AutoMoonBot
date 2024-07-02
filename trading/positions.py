import uuid
from collections import defaultdict
from trading import Category, Condition, Action, Status


class Position:
    def __init__(self, category: Category, price: float, size: float) -> None:

        self.category = category
        self.condition = Condition.CREATED
        self._price = price
        self._size = size
        self._entry = None
        self._exit = None
        self._value = None

    def value(self) -> float:
        if self._entry is None:
            return 0
        else:
            return self._size * (self._price / self._entry)

    def update(self, price: float) -> None:
        self._price = price

    def open(self) -> Status:
        if self.condition == Condition.CREATED:
            self.condition = Condition.OPENED
            self._entry = self._price
            return Status.SUCCESS
        else:
            return Status.INVALID

    def close(self) -> Status:
        if self.condition != Condition.CLOSED:
            self.condition = Condition.CLOSED
            self._exit = self._price
            return Status.SUCCESS
        else:
            return Status.INVALID


class Positions:
    def __init__(self, price: float) -> None:
        self._price = price
        self._opened = defaultdict(Position)
        self._closed = defaultdict(Position)

    def reset(self, price: float) -> None:
        self._price = price
        self._opened.clear()
        self._closed.clear()

    def update(self, price: float) -> None:
        self._price = price
        for position in self._opened.values():
            position.update(price)

    def value(self) -> float:
        return sum([position.value() for position in self._opened.values()])

    def fetch_opened(self, pid: str) -> Position | None:
        return self._opened.pop(pid, None)

    def fetch_closed(self, pid: str) -> Position | None:
        return self._closed.pop(pid, None)

    def open(self, category: Category, size: float) -> Status:
        pid = str(uuid.uuid4())
        position = Position(category, self._price, size)
        status = position.open()
        self._opened[pid] = position
        return status

    def close(self, pid: str) -> Status:
        position = self._opened.pop(pid, None)
        status = position.close()
        self._closed[pid] = position
        return status

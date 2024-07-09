from enum import Enum


class Category(Enum):
    MARKET: int = 0
    LONG: int = 1
    SHORT: int = 2


class Condition(Enum):
    CREATED: int = 0
    OPENED: int = 1
    CLOSED: int = 2
    EXPIRED: int = 3


class Action(Enum):
    ENTER: int = 0
    IDLE: int = 1
    EXIT: int = 2


class Status(Enum):
    SUCCESS: int = 0
    NOFUNDS: int = 1
    INVALID: int = 2

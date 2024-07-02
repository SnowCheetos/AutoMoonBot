from enum import Enum
from dataclasses import dataclass


class Category(Enum):
    MARKET : int = 0
    LONG   : int = 1
    SHORT  : int = 2

class Condition(Enum):
    CREATED   : int = 0
    CANCELLED : int = 1
    OPENED    : int = 2
    CLOSED    : int = 3
    EXPIRED   : int = 4

class Action(Enum):
    ENTER : int = 0
    HOLD  : int = 1
    EXIT  : int = 2
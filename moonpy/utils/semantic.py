from enum import Enum, auto

class Tense(Enum):
    Past = auto()
    Present = auto()
    Future = auto()

class Aspect(Enum):
    Simple = auto()
    Perfect = auto()
    Continuous = auto()
    PerfectContinuous = auto()
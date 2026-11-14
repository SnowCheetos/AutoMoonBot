"""Baselines: simple rules and a faithful re-implementation of PGPerceptron,
evaluated out of sample with the same harness as every new model."""

from automoonbot.baselines.rules import (
    BuyAndHold,
    Cash,
    CrossSectionalMomentum,
    EqualWeight,
    InverseVolatility,
    TrendFollowing,
)

__all__ = [
    "BuyAndHold",
    "Cash",
    "CrossSectionalMomentum",
    "EqualWeight",
    "InverseVolatility",
    "TrendFollowing",
]

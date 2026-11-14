"""Faithful, leak-free re-implementation of PGPerceptron (SnowCheetos/PGPerceptron)."""

from automoonbot.baselines.pgperceptron.agent import Config, PGPerceptron, variants
from automoonbot.baselines.pgperceptron.descriptors import (
    FEATURE_PARAMS,
    descriptors,
    feature_dim,
    reference_descriptors,
)

__all__ = [
    "FEATURE_PARAMS",
    "Config",
    "PGPerceptron",
    "descriptors",
    "feature_dim",
    "reference_descriptors",
    "variants",
]

"""Walk-forward splits with purging and embargo.

For decision steps indexed ``0..T-1``, a fold trains on steps whose labels
are *fully realized* before the test window starts (purge), skips an
embargo after each test window before reusing data, and never trains on the
future: training uses only steps before the test window.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class Fold:
    train: npt.NDArray[np.int64]
    test: npt.NDArray[np.int64]


def walk_forward(
    n_steps: int,
    *,
    test_size: int,
    min_train: int,
    max_label_horizon: int,
    embargo: int = 0,
    train_window: int | None = None,
    step: int | None = None,
) -> Iterator[Fold]:
    """Yield expanding (or rolling, with ``train_window``) folds.

    A training step ``t`` has labels realized at bar ``t + 1 + h``; it is
    kept only if ``t + 1 + max_label_horizon < test_start``, so no training
    label overlaps the test period. ``embargo`` additionally drops the steps
    right before the test window (serial correlation of features).
    """
    if min(test_size, min_train) < 1 or max_label_horizon < 1 or embargo < 0:
        raise ValueError("invalid walk-forward parameters")
    stride = step or test_size
    test_start = min_train + max_label_horizon + 1 + embargo
    while test_start < n_steps:
        test_end = min(test_start + test_size, n_steps)
        last_train = test_start - max_label_horizon - 2 - embargo
        first_train = 0 if train_window is None else max(0, last_train + 1 - train_window)
        train = np.arange(first_train, last_train + 1, dtype=np.int64)
        if train.size >= min_train:
            yield Fold(train=train, test=np.arange(test_start, test_end, dtype=np.int64))
        test_start += stride

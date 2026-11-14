"""Multi-horizon prediction targets.

Targets are forward **total** log returns measured on the same price the
simulator trades at: from the open of bar ``t + 1`` (where a decision made
after bar ``t`` executes) to the open of bar ``t + 1 + h``, with splits and
dividends accounted for. They are labels only: they must never be used as
features, and a label for decision step ``t`` and horizon ``h`` only becomes
known at ``times[t + 1 + h]`` - which is what the walk-forward purge uses.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from automoonbot.backtest.panel import MarketPanel


@dataclass(frozen=True)
class Labels:
    horizons: tuple[int, ...]
    #: ``[T, S, H]`` forward total log returns (NaN where undefined).
    values: npt.NDArray[np.float64]
    #: ``[T, S, H]`` validity mask.
    mask: npt.NDArray[np.bool_]
    #: ``[T, H]`` index of the bar whose open realizes the label.
    realized_at: npt.NDArray[np.int64]


def forward_returns(panel: MarketPanel, horizons: Sequence[int]) -> Labels:
    hs = tuple(int(h) for h in horizons)
    if not hs or min(hs) < 1 or len(set(hs)) != len(hs):
        raise ValueError("horizons must be distinct positive integers")
    t, s = panel.shape
    # Wealth index of holding one share bought at an open, per bar open:
    # w[k] = cumulative split factor * open[k] + accrued dividends.
    o = np.where(panel.tradable, panel.open, np.nan)
    cum_split = np.cumprod(panel.split, axis=0)
    # Dividends are per pre-split share at their ex-date, i.e. per
    # cum_split[k-1] shares of the original; accrue them cumulatively.
    shares_before = np.vstack([np.ones((1, s)), cum_split[:-1]])
    divs = np.cumsum(panel.dividend * shares_before, axis=0)
    wealth = cum_split * o + divs  # value at open k of 1 original share (+ cash)
    out = np.full((t, s, len(hs)), np.nan)
    realized = np.full((t, len(hs)), -1, dtype=np.int64)
    for j, h in enumerate(hs):
        start = np.arange(t) + 1
        end = start + h
        ok = end < t
        a, b = start[ok], end[ok]
        # One share bought at open a (in original-share units, cum_split[a]
        # of them) is worth cum_split[b] * o[b] at open b, plus dividends with
        # ex-dates in (a, b]; the dividend at a itself went to the seller.
        with np.errstate(invalid="ignore", divide="ignore"):
            out[ok, :, j] = np.log((wealth[b] - divs[a]) / (cum_split[a] * o[a]))
        realized[ok, j] = b
    # Positions frozen by a halt at the end open would not be realizable.
    mask = np.isfinite(out)
    return Labels(hs, out, mask, realized)

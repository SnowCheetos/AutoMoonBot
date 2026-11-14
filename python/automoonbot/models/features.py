"""Causal per-security input features for the learned models.

Every feature at row ``t`` uses bars ``<= t`` only. Cross-sectional
normalization uses the same row's cross-section (all securities' values at
``t``), which is known at the decision; nothing is normalized with
statistics from the future, as a full-sample z-score would be.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from automoonbot.backtest.panel import MarketPanel
from automoonbot.baselines.rules import trailing_sum, trailing_vol

F32 = npt.NDArray[np.float32]
F64 = npt.NDArray[np.float64]

#: Names of the channels produced by :func:`price_features`, in order.
CHANNELS: tuple[str, ...] = (
    "ret",  # close-to-close total log return
    "gap",  # overnight: log(open / previous close), total return
    "intraday",  # log(close / open)
    "range",  # log(high / low)
    "clv",  # close location in the day's range, in [-1, 1]
    "volume",  # log(volume / trailing 21-day mean volume)
    "vol21",  # trailing 21-day volatility of ret
    "ret21",  # trailing 21-day total return
)


@dataclass(frozen=True)
class FeaturePanel:
    """``values[T, S, C]`` (float32, NaN where undefined) and ``ok[T, S]``:
    a bar exists at ``t`` and every channel is finite."""

    values: F32
    ok: npt.NDArray[np.bool_]
    channels: tuple[str, ...] = CHANNELS


def _robust_cross_section(x: F64, ok: npt.NDArray[np.bool_], clip: float = 5.0) -> F64:
    """Per-row median/MAD standardization over the valid securities,
    clipped to ``[-clip, clip]``."""
    v = np.where(ok, x, np.nan)
    # Rows with no valid security give NaN statistics (and a warning).
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(v, axis=1, keepdims=True)
        mad = np.nanmedian(np.abs(v - med), axis=1, keepdims=True) * 1.4826
    scale = np.where(np.isfinite(mad) & (mad > 0), mad, 1.0)
    z = (v - np.nan_to_num(med)) / scale
    return np.clip(z, -clip, clip)


def price_features(panel: MarketPanel, *, normalize: bool = True) -> FeaturePanel:
    r = panel.log_returns()
    tradable = panel.tradable
    with np.errstate(invalid="ignore", divide="ignore"):
        prev = np.vstack([np.full((1, panel.shape[1]), np.nan), panel.close[:-1]])
        # Overnight total return: split/dividend-adjusted previous close.
        gap = np.log((panel.open * panel.split + panel.dividend) / prev)
        intraday = np.log(panel.close / panel.open)
        rng = np.log(panel.high / panel.low)
        span = panel.high - panel.low
        clv = np.where(span > 0, (2 * panel.close - panel.high - panel.low) / span, 0.0)
        # Volume in original-share units, so splits do not look like spikes.
        cum_split = np.cumprod(panel.split, axis=0)
        vol_shares = np.where(tradable, panel.volume / cum_split, np.nan)
        vol_total, vol_n = trailing_sum(vol_shares, 21, 10)
        mean_vol = vol_total / vol_n
        volume = np.log1p(vol_shares) - np.log1p(mean_vol)
    vol21 = trailing_vol(r, 21, 10)
    ret21, _ = trailing_sum(r, 21, 15)
    raw = [r, gap, intraday, rng, clv, volume, vol21, ret21]
    stack = np.stack(raw, axis=-1)
    ok = tradable & np.isfinite(stack).all(axis=-1)
    if normalize:
        stack = np.stack([_robust_cross_section(x, ok) for x in raw], axis=-1)
    out = np.where(ok[..., None], stack, np.nan).astype(np.float32)
    return FeaturePanel(out, ok)

"""Rule-based baselines every learned model must beat out of sample.

All signals are computed vectorized over the whole panel from causal
transforms (row ``t`` depends on bars ``<= t`` only) and sliced at the
requested steps; :func:`automoonbot.backtest.evaluate.check_causal` verifies
this in the test suite. Between rebalances the strategies emit ``NaN``
(hold), so weights drift with prices and no turnover is spent keeping them
constant. A rebalance happens on steps divisible by ``rebalance`` and on the
first step of every :meth:`targets` call, so a fold never starts in cash.

Only ``investable[t]`` (a bar exists at ``t`` and the security is in the
point-in-time universe) decides eligibility; securities with no bar at the
decision are skipped rather than guessed at.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from automoonbot.backtest.panel import MarketPanel

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def _prefix(x: F64) -> F64:
    return np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])


def trailing_sum(x: F64, window: int, min_count: int) -> tuple[F64, F64]:
    """Inclusive trailing sum over rows ``[t - window + 1, t]`` ignoring NaN,
    and the count of valid rows; the sum is NaN below ``min_count``."""
    if window < 1 or min_count < 1:
        raise ValueError("window and min_count must be >= 1")
    ok = ~np.isnan(x)
    cs, cn = _prefix(np.where(ok, x, 0.0)), _prefix(ok.astype(np.float64))
    hi = np.arange(1, x.shape[0] + 1)
    lo = np.maximum(0, hi - window)
    n = cn[hi] - cn[lo]
    total = cs[hi] - cs[lo]
    return np.where(n >= min_count, total, np.nan), n


def trailing_vol(r: F64, window: int, min_count: int | None = None) -> F64:
    """Inclusive trailing sample standard deviation of returns."""
    m = min_count if min_count is not None else max(2, window // 2)
    s1, n = trailing_sum(r, window, m)
    s2, _ = trailing_sum(r * r, window, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (s2 - s1 * s1 / n) / (n - 1)
    return np.sqrt(np.maximum(var, 0.0))


def _schedule(steps: I64, every: int) -> npt.NDArray[np.bool_]:
    if every < 1:
        raise ValueError("rebalance period must be >= 1")
    due: npt.NDArray[np.bool_] = np.asarray(steps % every == 0)
    if due.size:
        due[0] = True
    return due


def _emit(weights: F64, steps: I64, every: int) -> F64:
    """Rows of ``weights`` at rebalance steps, NaN (hold) elsewhere."""
    out = np.full((steps.size, weights.shape[1]), np.nan)
    due = _schedule(steps, every)
    out[due] = weights[steps[due]]
    return out


class _Rule:
    name = "rule"

    def fit(self, panel: MarketPanel, train: I64) -> None:
        """Rules have no parameters to fit."""

    def weights(self, panel: MarketPanel) -> F64:
        raise NotImplementedError

    rebalance = 1

    def targets(self, panel: MarketPanel, steps: I64) -> F64:
        steps = np.asarray(steps, dtype=np.int64)
        return _emit(self.weights(panel), steps, self.rebalance)


@dataclass
class Cash(_Rule):
    name: str = "cash"

    def targets(self, panel: MarketPanel, steps: I64) -> F64:
        return np.zeros((np.asarray(steps).size, panel.shape[1]))


@dataclass
class BuyAndHold(_Rule):
    """Equal weight across securities investable at the first step, then hold.
    Delisted holdings go to cash and stay there."""

    name: str = "buy_and_hold"

    def targets(self, panel: MarketPanel, steps: I64) -> F64:
        steps = np.asarray(steps, dtype=np.int64)
        out = np.full((steps.size, panel.shape[1]), np.nan)
        if steps.size:
            ok = panel.investable[steps[0]]
            out[0] = np.where(ok, 1.0 / max(int(ok.sum()), 1), 0.0)
        return out


@dataclass
class EqualWeight(_Rule):
    rebalance: int = 21
    name: str = "equal_weight"

    def weights(self, panel: MarketPanel) -> F64:
        ok = panel.investable
        n = ok.sum(axis=1, keepdims=True)
        return np.where(ok, 1.0 / np.maximum(n, 1), 0.0)


@dataclass
class InverseVolatility(_Rule):
    window: int = 63
    rebalance: int = 21
    name: str = "inverse_volatility"

    def weights(self, panel: MarketPanel) -> F64:
        vol = trailing_vol(panel.log_returns(), self.window)
        with np.errstate(divide="ignore"):
            inv = np.where(panel.investable & (vol > 0), 1.0 / vol, 0.0)
        inv = np.nan_to_num(inv, nan=0.0)
        total = inv.sum(axis=1, keepdims=True)
        return np.divide(inv, total, out=np.zeros_like(inv), where=total > 0)


@dataclass
class CrossSectionalMomentum(_Rule):
    """Long the top ``quantile`` by total return from ``t - lookback`` to
    ``t - skip`` (the classic 12-1 month momentum), equal weight."""

    lookback: int = 252
    skip: int = 21
    quantile: float = 0.2
    rebalance: int = 21
    name: str = "cross_sectional_momentum"

    def weights(self, panel: MarketPanel) -> F64:
        if not 0 <= self.skip < self.lookback or not 0.0 < self.quantile <= 1.0:
            raise ValueError("need 0 <= skip < lookback and 0 < quantile <= 1")
        span = self.lookback - self.skip
        total, _ = trailing_sum(panel.log_returns(), span, max(1, int(0.8 * span)))
        score = np.full_like(total, np.nan)
        score[self.skip :] = total[: total.shape[0] - self.skip]
        score[~panel.investable] = np.nan
        valid = np.isfinite(score)
        # Stable sort on -score: ties broken by security index, deterministic.
        order = np.argsort(np.where(valid, -score, np.inf), axis=1, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.broadcast_to(np.arange(score.shape[1]), order.shape), 1)
        n = valid.sum(axis=1)
        k = np.where(n > 0, np.maximum(1, np.ceil(self.quantile * n)), 1)[:, None]
        return np.where(valid & (ranks < k), 1.0 / k, 0.0)


@dataclass
class TrendFollowing(_Rule):
    """Time-series momentum: long each security whose trailing ``lookback``
    total return is positive, sized to ``vol_target`` annualized volatility
    per position (split across eligible securities), gross capped at 1."""

    lookback: int = 252
    vol_window: int = 63
    vol_target: float = 0.15
    periods_per_year: float = 252.0
    rebalance: int = 5
    name: str = "trend_following"

    def weights(self, panel: MarketPanel) -> F64:
        r = panel.log_returns()
        total, _ = trailing_sum(r, self.lookback, max(1, int(0.8 * self.lookback)))
        vol = trailing_vol(r, self.vol_window)
        eligible = panel.investable & np.isfinite(total) & np.isfinite(vol) & (vol > 0)
        n = np.maximum(eligible.sum(axis=1, keepdims=True), 1)
        per_period = self.vol_target / math.sqrt(self.periods_per_year)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = np.where(eligible & (total > 0), per_period / (vol * n), 0.0)
        gross = w.sum(axis=1, keepdims=True)
        return np.where(gross > 1.0, w / np.maximum(gross, 1e-300), w)

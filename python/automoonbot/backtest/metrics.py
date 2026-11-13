"""Risk-first performance metrics.

All functions take per-period **log** returns of the strategy's equity
(the simulator's ``log_return``). Annualization uses ``periods_per_year``.

The headline objective for this project is log-growth subject to tail risk,
so :func:`summary` leads with log-growth, CVaR and drawdown; Sharpe-type
ratios are reported with their statistical significance (probabilistic and
deflated Sharpe) because a raw Sharpe from a search over many configurations
is mostly selection bias.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np
import numpy.typing as npt

_N = NormalDist()
EULER_GAMMA = 0.5772156649015329

F64 = npt.NDArray[np.float64]


def _clean(r: npt.ArrayLike) -> F64:
    x = np.asarray(r, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("returns must be one-dimensional")
    if np.isnan(x).any():
        raise ValueError("returns contain NaN")
    return x


def equity_curve(log_returns: npt.ArrayLike, start: float = 1.0) -> F64:
    x = _clean(log_returns)
    return start * np.exp(np.concatenate([[0.0], np.cumsum(x)]))


def max_drawdown(log_returns: npt.ArrayLike) -> tuple[float, int]:
    """Largest peak-to-trough loss (as a positive fraction) and the longest
    time under water, in periods."""
    x = _clean(log_returns)
    if x.size == 0:
        return 0.0, 0
    if np.isneginf(x).any():
        return 1.0, int(x.size)
    cum = np.concatenate([[0.0], np.cumsum(x)])
    peak = np.maximum.accumulate(cum)
    dd = 1.0 - np.exp(cum - peak)
    under = cum < peak
    # Longest run of consecutive underwater points, vectorized.
    idx = np.arange(under.size)
    last_above = np.maximum.accumulate(np.where(under, -1, idx).astype(np.int64))
    longest = int(np.max(idx - last_above)) if under.any() else 0
    return float(np.max(dd)), longest


def cvar(log_returns: npt.ArrayLike, alpha: float = 0.95) -> float:
    """Expected shortfall of *simple* period returns at level ``alpha``,
    returned as a positive loss fraction (historical, with the fractional
    tail atom weighted exactly)."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    x = _clean(log_returns)
    if x.size == 0:
        return 0.0
    simple = np.sort(np.expm1(x))
    tail = (1.0 - alpha) * simple.size
    k = math.floor(tail)
    total = simple[:k].sum()
    if tail > k:
        total += (tail - k) * simple[k]
    return float(-total / tail)


def sharpe(log_returns: npt.ArrayLike) -> float:
    """Per-period Sharpe ratio of simple returns (no risk-free adjustment)."""
    x = np.expm1(_clean(log_returns))
    if x.size < 2:
        return float("nan")
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else float("nan")


def probabilistic_sharpe(log_returns: npt.ArrayLike, benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > ``benchmark``), accounting for sample length,
    skewness and kurtosis (Bailey & Lopez de Prado, 2012)."""
    x = np.expm1(_clean(log_returns))
    n = x.size
    if n < 3:
        return float("nan")
    sr = sharpe(log_returns)
    if math.isnan(sr):
        return float("nan")
    z = (x - x.mean()) / x.std(ddof=0)
    skew = float((z**3).mean())
    kurt = float((z**4).mean())
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if denom <= 0:
        return float("nan")
    return _N.cdf((sr - benchmark) * math.sqrt(n - 1) / math.sqrt(denom))


def expected_max_sharpe(n_trials: int, trial_sharpe_variance: float) -> float:
    """Expected maximum per-period Sharpe among ``n_trials`` unskilled trials."""
    if n_trials < 1 or trial_sharpe_variance < 0:
        raise ValueError("need n_trials >= 1 and a non-negative variance")
    if n_trials == 1:
        return 0.0
    sd = math.sqrt(trial_sharpe_variance)
    return sd * (
        (1 - EULER_GAMMA) * _N.inv_cdf(1 - 1 / n_trials)
        + EULER_GAMMA * _N.inv_cdf(1 - 1 / (n_trials * math.e))
    )


def deflated_sharpe(
    log_returns: npt.ArrayLike, n_trials: int, trial_sharpe_variance: float
) -> float:
    """Probability that the strategy's Sharpe beats the best of ``n_trials``
    unskilled configurations (Bailey & Lopez de Prado, 2014)."""
    return probabilistic_sharpe(log_returns, expected_max_sharpe(n_trials, trial_sharpe_variance))


@dataclass(frozen=True)
class Summary:
    periods: int
    log_growth: float
    log_growth_annual: float
    cagr: float
    volatility_annual: float
    cvar_95: float
    max_drawdown: float
    max_underwater_periods: int
    calmar: float
    sharpe_annual: float
    sortino_annual: float
    probabilistic_sharpe: float
    hit_rate: float
    ruined: bool

    def as_dict(self) -> dict[str, float | int | bool]:
        return asdict(self)


def summary(log_returns: npt.ArrayLike, periods_per_year: float = 252.0) -> Summary:
    x = _clean(log_returns)
    n = x.size
    ruined = bool(np.isneginf(x).any())
    if n == 0:
        raise ValueError("no returns")
    growth = float(x.sum())
    per_year = growth / n * periods_per_year
    simple = np.expm1(x)
    vol = float(simple.std(ddof=1) * math.sqrt(periods_per_year)) if n > 1 else float("nan")
    down = simple[simple < 0]
    dsd = math.sqrt(float((down**2).sum()) / n) if n else float("nan")
    mdd, under = max_drawdown(x)
    sr = sharpe(x) if not ruined else float("nan")
    cagr = math.expm1(per_year) if not ruined else -1.0
    return Summary(
        periods=n,
        log_growth=growth,
        log_growth_annual=per_year,
        cagr=cagr,
        volatility_annual=vol,
        cvar_95=cvar(x, 0.95) if not ruined else 1.0,
        max_drawdown=mdd,
        max_underwater_periods=under,
        calmar=cagr / mdd if mdd > 0 else float("inf"),
        sharpe_annual=sr * math.sqrt(periods_per_year),
        sortino_annual=float(simple.mean() / dsd * math.sqrt(periods_per_year))
        if dsd > 0
        else float("nan"),
        probabilistic_sharpe=probabilistic_sharpe(x) if not ruined else 0.0,
        hit_rate=float((x > 0).mean()),
        ruined=ruined,
    )

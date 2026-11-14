"""Walk-forward evaluation of strategies through the execution simulator.

A :class:`Strategy` is fitted on each fold with a panel *truncated* before
the fold's first test decision, so fitting cannot see the test period at all.
It then emits target weights for the fold's test steps; the weights for step
``t`` may use bars ``<= t`` only. That second property cannot be enforced by
construction for arbitrary code, so :func:`check_causal` verifies it by
recomputing targets on truncated panels, and :func:`evaluate` can spot-check
it on every fold (``audit``).

All folds' test windows must tile one contiguous out-of-sample span, which is
simulated as a single path: the model is swapped at fold boundaries, and
positions carry over, as they would in production.

Because every configuration tried makes the best result look better than it
is, :func:`compare` reports the deflated Sharpe ratio across all trials
evaluated on the same span.
"""

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from automoonbot._moonrs import Simulator
from automoonbot.backtest.metrics import Summary, deflated_sharpe, sharpe, summary
from automoonbot.backtest.panel import MarketPanel
from automoonbot.backtest.walkforward import Fold

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@runtime_checkable
class Strategy(Protocol):
    """Maps point-in-time market data to target portfolio weights.

    ``targets(panel, steps)`` returns ``float64[len(steps), S]``: row ``k`` is
    the weight vector decided after bar ``steps[k]`` closes and executed at
    the next open. ``steps`` are consecutive, and stateful strategies start
    from a flat book at ``steps[0]``. It may read rows ``<= steps[k]`` of
    ``open, high, low, close, volume, tradable, split, dividend, times,
    decision_time``, but never ``listed`` (which encodes future delistings).
    ``NaN`` holds the current position. The call must be deterministic.
    """

    name: str

    def fit(self, panel: MarketPanel, train: I64) -> None: ...

    def targets(self, panel: MarketPanel, steps: I64) -> F64: ...


@dataclass(frozen=True)
class Trial:
    name: str
    params: Mapping[str, object]
    #: Bar index each return ends on.
    periods: I64
    log_returns: F64
    turnover: F64
    costs: F64
    summary: Summary
    folds: int
    periods_per_year: float


@dataclass
class CausalityError(AssertionError):
    strategy: str
    step: int
    full: F64 = field(repr=False)
    truncated: F64 = field(repr=False)

    def __str__(self) -> str:
        return (
            f"{self.strategy}: targets at step {self.step} change when future bars are "
            f"removed ({self.full} vs {self.truncated})"
        )


def _same(a: F64, b: F64) -> bool:
    return bool(np.array_equal(a, b, equal_nan=True))


def check_causal(
    strategy: Strategy,
    panel: MarketPanel,
    steps: I64,
    *,
    probes: Iterable[int] | None = None,
    full: F64 | None = None,
) -> None:
    """Raise :class:`CausalityError` unless the targets at every probed step
    are identical when all bars after that step are removed."""
    steps = np.asarray(steps, dtype=np.int64)
    if full is None:
        full = strategy.targets(panel, steps)
    probe = range(steps.size) if probes is None else probes
    for k in probe:
        t = int(steps[k])
        trunc = strategy.targets(panel.truncate(t + 1), steps[: k + 1])
        if not _same(full[k], trunc[-1]):
            raise CausalityError(strategy.name, t, full[k], trunc[-1])


def _check_folds(folds: Sequence[Fold], n: int) -> None:
    if not folds:
        raise ValueError("no folds")
    for a, b in itertools.pairwise(folds):
        if b.test[0] != a.test[-1] + 1:
            raise ValueError("fold test windows must be contiguous and non-overlapping")
    for f in folds:
        if f.test.size == 0 or np.any(np.diff(f.test) != 1):
            raise ValueError("each test window must be a consecutive range")
        if f.train.size and f.train[-1] >= f.test[0]:
            raise ValueError("training steps must precede the test window")
    if folds[-1].test[-1] + 1 >= n:
        raise ValueError("the last test step needs a following bar to execute on")


def evaluate(
    panel: MarketPanel,
    strategy: Strategy,
    folds: Sequence[Fold],
    *,
    params: Mapping[str, object] | None = None,
    capital: float = 1.0,
    periods_per_year: float = 252.0,
    audit: int = 0,
    seed: int = 0,
    sim_kwargs: Mapping[str, object] | None = None,
) -> Trial:
    """Fit and simulate ``strategy`` over all folds as one out-of-sample path.

    ``audit`` causality probes are drawn per fold (``0`` disables). The last
    test step is dropped if it is the panel's final bar (nothing to execute
    on).
    """
    t_bars, s = panel.shape
    # The final bar has no next open to execute on; drop it from the tests.
    folds = [Fold(f.train, f.test[f.test < t_bars - 1]) for f in folds if f.test[0] < t_bars - 1]
    _check_folds(folds, t_bars)
    rng = np.random.default_rng(seed)
    rows: list[F64] = []
    for f in folds:
        strategy.fit(panel.truncate(int(f.test[0])), f.train)
        w = np.asarray(strategy.targets(panel, f.test), dtype=np.float64)
        if w.shape != (f.test.size, s):
            raise ValueError(f"{strategy.name}: targets must be [{f.test.size}, {s}]")
        if np.isinf(w).any():
            raise ValueError(f"{strategy.name}: infinite target weight")
        if audit:
            k = min(audit, f.test.size)
            check_causal(
                strategy,
                panel,
                f.test,
                probes=sorted(rng.choice(f.test.size, size=k, replace=False).tolist()),
                full=w,
            )
        rows.append(w)
    targets = np.ascontiguousarray(np.vstack(rows))
    sim = Simulator(
        panel.market(),
        start=int(folds[0].test[0]),
        capital=capital,
        periods_per_year=periods_per_year,
        **dict(sim_kwargs or {}),  # type: ignore[arg-type]
    )
    out = sim.run(targets)
    lr = out["log_return"]
    costs = out["commission"] + out["spread_cost"] + out["impact_cost"] + out["borrow_cost"]
    equity_open = out["equity_open"]
    with np.errstate(invalid="ignore", divide="ignore"):
        cost_frac = np.where(equity_open > 0, costs / equity_open, 0.0)
    return Trial(
        name=strategy.name,
        params=dict(params or {}),
        periods=out["period"].astype(np.int64),
        log_returns=lr,
        turnover=out["turnover"],
        costs=cost_frac,
        summary=summary(lr, periods_per_year),
        folds=len(folds),
        periods_per_year=periods_per_year,
    )


def compare(trials: Sequence[Trial]) -> list[dict[str, object]]:
    """One row per trial, best log-growth first, with the deflated Sharpe
    ratio computed against all ``len(trials)`` trials (which must share the
    same out-of-sample periods)."""
    if not trials:
        return []
    ref = trials[0].periods
    for t in trials[1:]:
        if not np.array_equal(t.periods, ref):
            raise ValueError("trials must be evaluated on identical periods")
    srs = np.array([sharpe(t.log_returns) for t in trials])
    finite = srs[np.isfinite(srs)]
    var = float(finite.var(ddof=1)) if finite.size > 1 else 0.0
    out: list[dict[str, object]] = []
    for t, sr in zip(trials, srs, strict=True):
        ruined = bool(np.isneginf(t.log_returns).any())
        dsr = (
            deflated_sharpe(t.log_returns, len(trials), var)
            if math.isfinite(sr) and not ruined
            else float("nan")
        )
        out.append(
            {
                "name": t.name,
                **{f"param.{k}": v for k, v in t.params.items()},
                **t.summary.as_dict(),
                "turnover_annual": float(t.turnover.sum() / t.summary.periods * t.periods_per_year),
                "cost_annual": float(t.costs.sum() / t.summary.periods * t.periods_per_year),
                "deflated_sharpe": dsr,
            }
        )
    out.sort(key=lambda r: float(r["log_growth"]), reverse=True)  # type: ignore[arg-type]
    return out

"""The standard Stage 4 comparison: every baseline and the three PGPerceptron
configurations on one out-of-sample path, with deflated Sharpe ratios.

The walk-forward cadence follows the original PGPerceptron server: retrain
every ``retrain_every`` bars on (at most) the last ``train_window`` bars.
Rule baselines are evaluated on the same folds, so every trial covers the
identical out-of-sample periods.
"""

from collections.abc import Mapping, Sequence

from automoonbot.backtest.evaluate import Strategy, Trial, compare, evaluate
from automoonbot.backtest.panel import MarketPanel
from automoonbot.backtest.walkforward import walk_forward
from automoonbot.baselines.pgperceptron.agent import variants
from automoonbot.baselines.rules import (
    BuyAndHold,
    Cash,
    CrossSectionalMomentum,
    EqualWeight,
    InverseVolatility,
    TrendFollowing,
)


def standard_strategies(
    n_securities: int, periods_per_year: float, *, seed: int = 0, **pgp: object
) -> list[Strategy]:
    out: list[Strategy] = [
        Cash(),
        BuyAndHold(),
        TrendFollowing(periods_per_year=periods_per_year),
    ]
    if n_securities > 1:
        out += [EqualWeight(), InverseVolatility(), CrossSectionalMomentum()]
    out += variants(seed=seed, **pgp)
    return out


def run(
    panel: MarketPanel,
    *,
    periods_per_year: float,
    retrain_every: int = 50,
    train_window: int = 1000,
    min_train: int = 1000,
    strategies: Sequence[Strategy] | None = None,
    sim_kwargs: Mapping[str, object] | None = None,
    audit: int = 2,
    seed: int = 0,
) -> tuple[list[Trial], list[dict[str, object]]]:
    folds = list(
        walk_forward(
            panel.shape[0],
            test_size=retrain_every,
            min_train=min_train,
            max_label_horizon=1,
            train_window=train_window,
        )
    )
    strats = strategies or standard_strategies(panel.shape[1], periods_per_year, seed=seed)
    trials = [
        evaluate(
            panel,
            s,
            folds,
            params=getattr(s, "params", None),
            periods_per_year=periods_per_year,
            audit=audit,
            seed=seed,
            sim_kwargs=sim_kwargs,
        )
        for s in strats
    ]
    return trials, compare(trials)

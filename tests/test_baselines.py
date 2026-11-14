"""Tests for the evaluation harness, rule baselines and the PGPerceptron port."""

import numpy as np
import pytest

from automoonbot.backtest import (
    CausalityError,
    MarketPanel,
    Simulator,
    check_causal,
    compare,
    evaluate,
    walk_forward,
)
from automoonbot.baselines import (
    BuyAndHold,
    Cash,
    CrossSectionalMomentum,
    EqualWeight,
    InverseVolatility,
    TrendFollowing,
)

from helpers import HOUR, synthetic


# ------------------------------------------------------------------ panel
def test_total_return_index_ignores_splits_and_is_causal() -> None:
    p = synthetic(200, 2)
    tri = p.total_return_index()
    r = np.diff(np.log(tri[:, 0]))
    assert np.abs(r).max() < 0.1  # the split is not a -69% bar
    np.testing.assert_allclose(p.truncate(120).total_return_index(), tri[:120])


def test_simulator_hold_keeps_shares() -> None:
    p = synthetic(10, 1)
    sim = Simulator(p.market(), commission_bps=0, half_spread_bps=0, impact_coef=0)
    sim.step(np.array([0.5]))
    q = sim.shares().copy()
    out = sim.run(np.full((3, 1), np.nan))
    np.testing.assert_array_equal(sim.shares(), q)
    assert out["turnover"].sum() == 0
    with pytest.raises(ValueError):
        sim.step(np.array([np.inf]))


# ------------------------------------------------------------------ rules
RULES = [
    Cash(),
    BuyAndHold(),
    EqualWeight(rebalance=7),
    InverseVolatility(window=20, rebalance=5),
    CrossSectionalMomentum(lookback=40, skip=5, quantile=0.5, rebalance=10),
    TrendFollowing(lookback=30, vol_window=20, rebalance=3),
]


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.name)
def test_rules_are_causal(rule: object) -> None:
    p = synthetic(160, 4, halts=True)
    steps = np.arange(60, 150)
    check_causal(rule, p, steps)  # type: ignore[arg-type]
    w = rule.targets(p, steps)  # type: ignore[attr-defined]
    assert w.shape == (steps.size, 4)
    assert not np.isinf(w).any()
    finite = np.nan_to_num(w, nan=0.0)
    assert (finite >= 0).all() and (finite.sum(axis=1) <= 1 + 1e-12).all()


def test_rule_schedules() -> None:
    p = synthetic(100, 3)
    steps = np.arange(33, 60)
    bh = BuyAndHold().targets(p, steps)
    assert np.allclose(bh[0], 1 / 3) and np.isnan(bh[1:]).all()
    ew = EqualWeight(rebalance=10).targets(p, steps)
    due = np.flatnonzero(~np.isnan(ew[:, 0]))
    assert list(steps[due]) == [33, 40, 50]


def test_momentum_picks_the_winner() -> None:
    t = 80
    c = np.ones((t, 3)) * 10
    c[:, 2] *= np.exp(np.arange(t) * 0.01)  # steady winner
    c[:, 0] *= np.exp(-np.arange(t) * 0.01)
    times = np.arange(t, dtype=np.int64) * HOUR
    p = MarketPanel.from_arrays(["a", "b", "c"], times, c, c, np.ones_like(c))
    w = CrossSectionalMomentum(lookback=20, skip=2, quantile=0.3, rebalance=5).targets(
        p, np.array([40])
    )
    np.testing.assert_array_equal(w[0], [0, 0, 1])


def test_check_causal_catches_peeking() -> None:
    class Peek:
        name = "peek"

        def fit(self, panel: MarketPanel, train: object) -> None: ...

        def targets(self, panel: MarketPanel, steps: np.ndarray) -> np.ndarray:
            nxt = np.roll(panel.close[:, 0], -1)  # tomorrow's close
            up = (nxt > panel.close[:, 0]).astype(float)
            return up[steps][:, None]

    p = synthetic(50, 1)
    with pytest.raises(CausalityError):
        check_causal(Peek(), p, np.arange(10, 40))  # type: ignore[arg-type]


# ---------------------------------------------------------------- harness
def test_evaluate_and_compare() -> None:
    p = synthetic(300, 3)
    folds = list(walk_forward(300, test_size=40, min_train=80, max_label_horizon=1))
    trials = [evaluate(p, r, folds, audit=3) for r in RULES]
    assert all(t.periods[0] == folds[0].test[0] + 1 for t in trials)
    assert all(t.periods[-1] == 299 for t in trials)
    rows = compare(trials)
    assert len(rows) == len(RULES)
    assert [r["log_growth"] for r in rows] == sorted((r["log_growth"] for r in rows), reverse=True)
    cash = next(t for t in trials if t.name == "cash")
    assert np.all(cash.log_returns == 0)
    bad = [folds[0], folds[2]]
    with pytest.raises(ValueError, match="contiguous"):
        evaluate(p, Cash(), bad)


def test_fit_never_sees_the_test_window() -> None:
    seen: list[int] = []

    class Spy(Cash):
        def fit(self, panel: MarketPanel, train: object) -> None:
            seen.append(panel.shape[0])

    p = synthetic(200, 1)
    folds = list(walk_forward(200, test_size=30, min_train=50, max_label_horizon=2))
    evaluate(p, Spy(), folds)
    assert seen == [int(f.test[0]) for f in folds]


def test_rules_respect_membership() -> None:
    from automoonbot.baselines import EqualWeight

    p = synthetic(60, 4)
    member = np.ones(p.shape, dtype=bool)
    member[:, 2] = False
    w = EqualWeight(rebalance=1).targets(p.with_membership(member), np.arange(30, 40))
    assert (w[:, 2] == 0).all() and np.allclose(w.sum(axis=1), 1.0)
    with pytest.raises(ValueError):
        p.with_membership(member[:10])

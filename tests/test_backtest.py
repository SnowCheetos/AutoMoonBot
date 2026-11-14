"""Tests for panels, the simulator bindings, labels, walk-forward and metrics."""

import math

import numpy as np
import pytest
from automoonbot.backtest import MarketPanel, Simulator, forward_returns, summary, walk_forward
from automoonbot.backtest.metrics import (
    cvar,
    deflated_sharpe,
    expected_max_sharpe,
    max_drawdown,
    probabilistic_sharpe,
)

DAY = 86_400 * 10**9


def panel(open_: list[list[float]], close: list[list[float]], **kw: object) -> MarketPanel:
    o, c = np.array(open_, float), np.array(close, float)
    t = np.arange(o.shape[0], dtype=np.int64) * DAY
    return MarketPanel.from_arrays(
        [f"s{i}" for i in range(o.shape[1])], t, o, c, np.full(o.shape, 1e9), window=2, **kw
    )


def test_panel_lags_and_listing() -> None:
    nan = np.nan
    p = panel(
        [[10, nan], [11, 20], [nan, 21], [12, 22], [13, nan]],
        [[10, nan], [11, 20], [nan, 21], [12, 22], [13, nan]],
    )
    np.testing.assert_array_equal(p.listed[:, 0], [True] * 5)
    np.testing.assert_array_equal(p.listed[:, 1], [False, True, True, True, False])
    np.testing.assert_array_equal(p.tradable[:, 0], [True, True, False, True, True])
    assert p.close[2, 0] == 11  # carried over the halt
    # adv at t only uses bars < t: nothing known at t=0 or t=1 (needs 2 obs).
    assert p.adv_shares[0, 0] == 0 and p.adv_shares[1, 0] == 0
    assert p.adv_shares[2, 0] == 1e9


def test_simulator_buy_and_hold_and_report() -> None:
    p = panel([[10], [11], [12]], [[10], [11.5], [12.6]])
    sim = Simulator(
        p.market(),
        capital=1000.0,
        commission_bps=0,
        half_spread_bps=0,
        impact_coef=0,
        max_participation=1.0,
    )
    out = sim.run(np.ones((2, 1)))
    assert out["equity"].shape == (2,)
    # No volume history before bar 2, so the participation cap blocks the
    # first trade; the position is opened at the open of bar 2.
    assert out["clipped"][0] == 1 and out["equity"][0] == 1000
    assert out["equity"][-1] == pytest.approx(1000 / 12 * 12.6)
    assert sim.done
    with pytest.raises(ValueError):
        sim.step(np.ones(1))


def test_simulator_validates() -> None:
    p = panel([[10], [11]], [[10], [11]])
    sim = Simulator(p.market())
    with pytest.raises(ValueError):
        sim.step(np.array([np.inf]))
    with pytest.raises(ValueError):
        Simulator(p.market(), start=5)
    with pytest.raises(ValueError):
        Simulator(p.market(), half_spread_bps=-1.0)


def test_labels_total_return_and_timing() -> None:
    # 2:1 split at bar 3 with a 1.0 dividend per pre-split share at bar 3.
    o = [[10.0], [10.0], [10.0], [5.0], [5.0]]
    split = np.ones((5, 1))
    split[3] = 2.0
    div = np.zeros((5, 1))
    div[3] = 1.0
    p = panel(o, o, split=split, dividend=div)
    lab = forward_returns(p, [1, 2])
    # Decision at t=1 executes at open 2 (10.0); h=1 realizes at open 3:
    # 2 shares * 5.0 + 1.0 dividend = 11 per share bought at 10.
    assert lab.values[1, 0, 0] == pytest.approx(math.log(1.1))
    assert lab.realized_at[1, 0] == 3
    # Decision at t=2 buys at open 3, after the ex-date: no dividend.
    assert lab.values[2, 0, 0] == pytest.approx(0.0)
    assert not lab.mask[3, 0, 0] and not lab.mask[2, 0, 1]
    with pytest.raises(ValueError):
        forward_returns(p, [0])


def test_walk_forward_never_overlaps() -> None:
    h, emb = 5, 2
    folds = list(walk_forward(200, test_size=20, min_train=50, max_label_horizon=h, embargo=emb))
    assert folds
    for f in folds:
        # Every training label is realized before the first test decision.
        assert f.train.max() + 1 + h < f.test.min() - emb
        assert f.train.min() == 0
    rolling = list(
        walk_forward(200, test_size=20, min_train=50, max_label_horizon=h, train_window=60)
    )
    assert all(f.train.size <= 60 for f in rolling)


def test_metrics() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.0005, 0.01, 2520)
    s = summary(r)
    assert s.periods == 2520 and not s.ruined
    assert s.log_growth == pytest.approx(r.sum())
    assert 0 < s.max_drawdown < 1
    assert s.cvar_95 > 0
    # CVaR of a known distribution: 20 returns, worst 1 at alpha=0.95.
    x = np.log1p(np.array([-0.1] + [0.01] * 19))
    assert cvar(x, 0.95) == pytest.approx(0.1)
    assert max_drawdown(np.log([0.5, 2.0, 0.5]))[0] == pytest.approx(0.5)
    ruined = summary(np.array([0.01, -np.inf]))
    assert ruined.ruined and ruined.max_drawdown == 1.0
    # More trials raise the bar.
    assert expected_max_sharpe(100, 0.01) > expected_max_sharpe(10, 0.01) > 0
    psr = probabilistic_sharpe(r)
    assert deflated_sharpe(r, 1000, 0.002) < psr


def test_panel_from_store(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import datetime as dt

    import pyarrow as pa
    from automoonbot.data import ParquetStore
    from automoonbot.data.schemas import TS

    store = ParquetStore(tmp_path)
    days = [dt.date(2024, 1, d) for d in (2, 3, 4, 5)]
    close_ns = [
        int(dt.datetime(d.year, d.month, d.day, 21, tzinfo=dt.UTC).timestamp()) * 10**9
        for d in days
    ]
    n = len(days)
    bars = pa.table(
        {
            "security_id": ["AAA"] * n,
            "start": pa.array([c - 6 * 3600 * 10**9 for c in close_ns]).cast(TS),
            "end": pa.array(close_ns).cast(TS),
            "open": [10.0, 10.0, 5.0, 5.0],
            "high": [10.0, 10.0, 5.0, 5.0],
            "low": [10.0, 10.0, 5.0, 5.0],
            "close": [10.0, 10.0, 5.0, 5.0],
            "volume": [1e6] * n,
            "vwap": pa.array([None] * n, pa.float64()),
            "known_time": pa.array([c + 900 * 10**9 for c in close_ns]).cast(TS),
            "source": ["test"] * n,
        }
    )
    store.write("bars", bars)
    store.write(
        "corporate_actions",
        pa.table(
            {
                "security_id": ["AAA"],
                "action": ["split"],
                "ex_date": pa.array([days[2]], pa.date32()),
                "value": [2.0],
                "known_time": pa.array([close_ns[0]]).cast(TS),
                "source": ["test"],
            }
        ),
    )
    p = MarketPanel.from_store(store, ["AAA"], window=2)
    assert p.shape == (4, 1)
    np.testing.assert_array_equal(p.split[:, 0], [1, 1, 2, 1])
    np.testing.assert_array_equal(p.decision_time, np.array(close_ns) + 900 * 10**9)
    # The split is value-neutral, so the trailing volatility stays zero.
    assert np.all(p.sigma[:, 0] == 0)

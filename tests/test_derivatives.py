"""Options, futures and margin through the Python bindings."""

from dataclasses import replace

import numpy as np
import pytest

from automoonbot.backtest import MarketPanel, Simulator, simulate

FREE = dict(
    commission_bps=0.0,
    half_spread_bps=0.0,
    impact_coef=0.0,
    borrow_rate=0.0,
    margin_rate=0.0,
    option_commission=0.0,
    future_commission=0.0,
    option_half_spread_bps=0.0,
    option_min_half_spread=0.0,
)


def liquid(p: MarketPanel) -> MarketPanel:
    """Unlimited volume from the first bar (the trailing ADV needs history)."""
    return replace(p, adv_shares=np.full(p.shape, 1e12))


def stock_and_call() -> MarketPanel:
    """Stock 100 -> 120 over 6 bars; a 110 call expiring at bar 4 that does
    not trade on bar 2 (but stays listed)."""
    t = np.arange(6) * 86_400 * 10**9
    stock = [100.0, 100.0, 104.0, 110.0, 120.0, 121.0]
    call = [2.0, 2.0, np.nan, 3.0, 10.0, np.nan]
    px = np.column_stack([stock, call])
    vol = np.full_like(px, 1e9)
    p = liquid(MarketPanel.from_arrays(["S", "C"], t, px, px, vol))
    return p.with_instruments(
        [
            {"kind": "equity"},
            {"kind": "option", "underlying": 0, "right": "call", "strike": 110.0, "expiry": 4},
        ]
    )


def test_options_stay_listed_to_expiry() -> None:
    p = stock_and_call()
    assert p.listed[:, 1].tolist() == [True, True, True, True, True, False]
    assert not p.tradable[2, 1] and p.close[2, 1] == 2.0  # carried forward
    assert np.isnan(p.close[5, 1])


def test_exercise_through_simulate() -> None:
    p = stock_and_call()
    targets = np.array([[0.0, 1.0], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    sim = Simulator(p.market(), start=0, capital=10_000.0, **FREE)
    out = sim.run(targets, units=True)
    assert out["exercised"].tolist() == [0.0, 0.0, 0.0, 1.0]
    # Paid 200 premium and 11,000 at the strike for 100 shares worth 12,000.
    assert out["equity"][-1] == pytest.approx(10_000.0 - 200.0 + 1_000.0)
    assert sim.shares().tolist() == [100.0, 0.0]
    trial = simulate(p, targets, 0, name="call", capital=10_000.0, sim_kwargs=FREE, units=True)
    assert trial.log_returns.sum() == pytest.approx(np.log(10_800.0 / 10_000.0))


def test_margin_models_and_reports() -> None:
    t = np.arange(4) * 86_400 * 10**9
    px = np.array([[10.0], [10.0], [6.0], [6.0]])
    p = liquid(MarketPanel.from_arrays(["S"], t, px, px, np.full_like(px, 1e12)))
    sim = Simulator(
        p.market(),
        start=0,
        capital=1_000.0,
        long_only=False,
        max_gross=3.0,
        max_weight=3.0,
        margin="reg_t",
        **FREE,
    )
    r = sim.step(np.array([3.0]))
    assert r["margin_scale"] == pytest.approx(2.0 / 3.0)
    assert r["margin_requirement"] == pytest.approx(0.25 * 2_000.0)
    r = sim.step(np.array([np.nan]))
    assert r["margin_call"] == 1.0
    r = sim.step(np.array([2.0]))
    assert r["liquidated"] == 1.0
    with pytest.raises(ValueError, match="unknown margin model"):
        Simulator(p.market(), margin="spam")
    with pytest.raises(ValueError, match="unknown parameter"):
        Simulator(p.market(), margin="reg_t", margin_params={"initial": 0.5})
    pm = Simulator(p.market(), margin="portfolio", margin_params={"equity_range": 0.2, "points": 4})
    assert pm.step(np.array([1.0]))["margin_requirement"] > 0


def test_instrument_dicts_are_strict() -> None:
    p = stock_and_call()
    bad = dict(p.instruments[1])  # type: ignore[index]
    bad["strik"] = 1.0
    with pytest.raises(ValueError, match="unknown key"):
        p.with_instruments([{"kind": "equity"}, bad]).market()
    with pytest.raises(ValueError, match="missing strike"):
        p.with_instruments(
            [{"kind": "equity"}, {"kind": "option", "underlying": 0, "right": "call", "expiry": 4}]
        ).market()
    with pytest.raises(ValueError, match="right"):
        p.with_instruments(
            [
                {"kind": "equity"},
                {"kind": "option", "underlying": 0, "right": "c", "strike": 1.0, "expiry": 4},
            ]
        ).market()


def test_futures_and_truncation() -> None:
    t = np.arange(5) * 86_400 * 10**9
    px = np.array([[5000.0], [5010.0], [5030.0], [5000.0], [4990.0]])
    p = liquid(MarketPanel.from_arrays(["ES"], t, px, px, np.full_like(px, 1e6))).with_instruments(
        [
            {
                "kind": "future",
                "multiplier": 50.0,
                "expiry": 3,
                "initial_margin": 0.06,
                "maintenance_margin": 0.05,
            }
        ]
    )
    sim = Simulator(
        p.market(), capital=100_000.0, long_only=False, max_gross=10.0, max_weight=10.0, **FREE
    )
    out = sim.run(np.array([[2.0], [np.nan], [np.nan]]), units=True)
    # Bought at 5010 (open = close here), settled at 5000 at expiry.
    assert out["variation_margin"].sum() == pytest.approx(2 * 50 * (5000.0 - 5010.0))
    assert out["expired"].tolist() == [0.0, 0.0, 1.0]
    # Truncation keeps the instruments and slices the per-bar arrays.
    cut = p.with_half_spread(np.full(p.shape, 0.25)).truncate(3)
    assert cut.instruments == p.instruments and cut.half_spread is not None
    assert cut.half_spread.shape == (3, 1)

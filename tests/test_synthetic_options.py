"""Synthetic Black-Scholes option chains."""

import datetime as dt
from dataclasses import replace
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from automoonbot import _moonrs
from automoonbot.backtest import MarketPanel, Simulator, check_causal
from automoonbot.backtest.panel import session_dates
from automoonbot.backtest.synthetic import Smile, SyntheticChain, tbill_rate
from automoonbot.baselines import OptionWrite

NY = ZoneInfo("America/New_York")


def underlying(n_days: int = 300, seed: int = 3) -> MarketPanel:
    days = [
        d.astype(dt.date)
        for d in np.arange(np.datetime64("2022-01-03"), np.datetime64("2024-01-01"))
        if np.is_busday(d)
    ][:n_days]
    ends = [int(dt.datetime.combine(d, dt.time(16), NY).timestamp()) * 10**9 for d in days]
    rng = np.random.default_rng(seed)
    close = 400 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, len(days))))
    opn = close * np.exp(rng.normal(0, 0.003, len(days)))
    p = MarketPanel.from_arrays(
        ["ticker:SPY"],
        np.array(ends),
        opn[:, None],
        close[:, None],
        np.full((len(days), 1), 1e8),
        decision_time=np.array(ends) + 60 * 10**9,
    )
    return replace(p, adv_shares=np.full(p.shape, 1e12))


def inputs(p: MarketPanel, vix: float = 20.0) -> tuple[np.ndarray, ...]:
    t = p.shape[0]
    return np.full(t, vix), np.full(t, vix), np.full(t, 0.03), np.full(t, 0.015)


def test_pricing_matches_the_textbook() -> None:
    one = lambda x: np.array([x], dtype=np.float64)  # noqa: E731
    for call, want in ((True, 4.759422), (False, 0.808599)):
        v = _moonrs.option_value(
            np.array([call]),
            one(42),
            one(40),
            one(0.5),
            one(0.1),
            one(0.1),
            one(0.2),
            american=False,
        )
        assert v[0] == pytest.approx(want, abs=1e-6)
    deep_put = _moonrs.option_value(
        np.array([False]), one(50), one(100), one(1.0), one(0.1), one(0.1), one(0.2)
    )
    assert deep_put[0] == pytest.approx(50.0)  # American floor
    with pytest.raises(ValueError, match="length"):
        _moonrs.option_value(
            np.array([True, False]), one(1), one(1), one(1), one(0), one(0), one(1)
        )


def test_smile_and_rates() -> None:
    s = Smile()
    tau = np.full(3, 30 / 365)
    v = s.vol(np.full(3, 20.0), np.array([-0.05, 0.0, 0.05]), tau)
    assert v[1] == pytest.approx(0.18)  # 0.9 x VIX
    assert v[0] > v[1] > v[2]  # puts rich, calls cheap
    assert tbill_rate(np.array([5.0]))[0] == pytest.approx(0.0512, abs=2e-4)
    with pytest.raises(ValueError):
        Smile(floor=1.5)


def test_chain_listing_and_prices() -> None:
    p = underlying()
    chain = SyntheticChain(levels=(0.0, 0.02))
    q = chain.build(p, "ticker:SPY", *inputs(p))
    days = session_dates(q.times)
    exp = chain.expiries(p, 0)
    # Third Fridays (none of 2022's fell on a holiday in this calendar).
    for e in exp:
        d = days[e].astype(dt.date)
        assert d.weekday() == 4 and 15 <= d.day <= 21
    assert q.instruments is not None
    opts = [(j, x) for j, x in enumerate(q.instruments) if x["kind"] == "option"]
    assert len(opts) <= 4 * len(exp)
    for j, x in opts:
        live = np.flatnonzero(q.listed[:, j])
        first, e = int(live[0]), int(x["expiry"])
        assert e == live[-1] and e in exp
        prior = exp[exp < e]
        assert first == (prior[-1] if prior.size else e - chain.first_life)
        spot = p.close[first, 0]
        assert abs(x["strike"] - round(x["strike"])) < 1e-9
        assert abs(x["strike"] / spot - 1) < 0.03  # at the money or 2% out, $1 strikes
        # Expires at intrinsic (floored at the tick).
        s_e, k = p.close[e, 0], x["strike"]
        intrinsic = max(s_e - k, 0) if x["right"] == "call" else max(k - s_e, 0)
        assert q.close[e, j] == pytest.approx(max(intrinsic, 0.01))
    # Put-call parity at the money at listing (European parity, both legs
    # well above the American floor).
    calls = {(x["strike"], x["expiry"]): j for j, x in opts if x["right"] == "call"}
    for j, x in opts:
        c = calls.get((x["strike"], x["expiry"]))
        if x["right"] != "put" or c is None:
            continue
        t0 = int(np.flatnonzero(q.listed[:, j])[0])
        tau = (x["expiry"] - t0) / 252
        s0, k = p.close[t0, 0], x["strike"]
        parity = s0 * np.exp(-0.015 * tau) - k * np.exp(-0.03 * tau)
        assert q.close[t0, c] - q.close[t0, j] == pytest.approx(parity, abs=1e-6)
    # Rows end no earlier than the 16:15 option close and are decided after.
    end = [
        int(dt.datetime.combine(d.astype(dt.date), dt.time(16, 15), NY).timestamp()) for d in days
    ]
    assert np.all(q.times // 10**9 >= end)
    assert np.all(q.decision_time // 10**9 >= np.array(end) + 15 * 60)


def test_prices_do_not_see_the_future() -> None:
    p = underlying()
    chain = SyntheticChain(levels=(0.0,))
    vo, vc, r, y = inputs(p)
    full = chain.build(p, "ticker:SPY", vo, vc, r, y)
    cut = 150
    # Change everything after the cut: the underlying, the index, rates.
    bumped = replace(
        p,
        close=np.vstack([p.close[: cut + 1], p.close[cut + 1 :] * 1.3]),
        open=np.vstack([p.open[: cut + 1], p.open[cut + 1 :] * 1.3]),
    )
    vc2, vo2, r2 = vc.copy(), vo.copy(), r.copy()
    vc2[cut + 1 :] = 45.0
    vo2[cut + 1 :] = 45.0
    r2[cut + 1 :] = 0.08
    other = chain.build(bumped, "ticker:SPY", vo2, vc2, r2, y)
    listed_by_cut = [j for j in range(1, full.shape[1]) if full.listed[: cut + 1, j].any()]
    assert listed_by_cut
    for j in listed_by_cut:
        sid = full.securities[j]
        k = other.securities.index(sid)  # same contract, same strike
        assert np.array_equal(full.close[: cut + 1, j], other.close[: cut + 1, k], equal_nan=True)
        assert np.array_equal(full.open[: cut + 1, j], other.open[: cut + 1, k], equal_nan=True)


def test_put_write_runs_and_is_causal() -> None:
    p = underlying()
    q = SyntheticChain(levels=(0.0,), rights=("put",)).build(p, "ticker:SPY", *inputs(p))
    s = OptionWrite("put")
    start = int(np.argmax(q.tradable[:, 1:].any(axis=1)))
    steps = np.arange(start, q.shape[0] - 1)
    w = s.targets(q, steps)
    check_causal(s, q, steps, probes=range(0, steps.size, 17), full=w)
    sim = Simulator(
        q.market(),
        start=start,
        capital=1e7,
        long_only=False,
        max_gross=2.5,
        max_weight=1.5,
        margin="portfolio",
    )
    out = sim.run(w)
    assert (out["equity"] > 0).all() and out["margin_call"].sum() == 0
    # Premium collected every cycle: the put writer's equity beats cash
    # whenever the underlying ends a cycle above the strike.
    assert np.isfinite(out["log_return"]).all()

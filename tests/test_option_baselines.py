"""Covered-call and put-write baselines on a synthetic, Black-Scholes-priced
monthly chain."""

import math
from dataclasses import replace

import numpy as np
import pytest

from automoonbot.backtest import MarketPanel, Simulator, check_causal, simulate
from automoonbot.baselines import OptionWrite

T = 130
EXPIRIES = (25, 50, 75, 100, 125)
STRIKES = tuple(float(k) for k in range(80, 135, 5))
LISTED_FOR = 45  # bars a contract trades before expiration
VOL = 0.2


def bs(right: str, s: float, k: float, tau: float) -> float:
    if tau <= 0:
        return max(s - k if right == "call" else k - s, 0.0)
    d1 = (math.log(s / k) + 0.5 * VOL**2 * tau) / (VOL * math.sqrt(tau))
    d2 = d1 - VOL * math.sqrt(tau)
    n = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    call = s * n(d1) - k * n(d2)
    return call if right == "call" else call - s + k


def chain() -> MarketPanel:
    rng = np.random.default_rng(7)
    stock = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, T)))
    cols: list[np.ndarray] = [stock]
    inst: list[dict[str, object]] = [{"kind": "equity"}]
    for e in EXPIRIES:
        for right in ("call", "put"):
            for k in STRIKES:
                px = np.full(T, np.nan)
                for t in range(max(0, e - LISTED_FOR), min(e, T - 1) + 1):
                    px[t] = max(bs(right, stock[t], k, (e - t) / 252), 0.01)
                cols.append(px)
                inst.append(
                    {"kind": "option", "underlying": 0, "right": right, "strike": k, "expiry": e}
                )
    close = np.column_stack(cols)
    names = ["ticker:XYZ"] + [f"option:{i}" for i in range(1, close.shape[1])]
    times = np.arange(T, dtype=np.int64) * 86_400 * 10**9
    p = MarketPanel.from_arrays(names, times, close, close, np.full(close.shape, 1e6))
    p = replace(p, adv_shares=np.full(p.shape, 1e12))
    return p.with_instruments(inst)


SIM = dict(long_only=False, max_gross=2.5, max_weight=1.5, margin="portfolio")


def test_choice_is_causal_and_by_the_rules() -> None:
    p = chain()
    steps = np.arange(5, T - 1)
    for s in (OptionWrite("call", otm=0.02), OptionWrite("put")):
        w = s.targets(p, steps)
        check_causal(s, p, steps, probes=range(0, steps.size, 7), full=w)
        writes = np.flatnonzero((w[:, 1:] < 0).any(axis=1))
        assert writes.size == len(EXPIRIES)  # one write per monthly cycle
        for k in writes:
            t = int(steps[k])
            j = int(np.flatnonzero(w[k] < 0)[0])
            x = p.instruments[j]  # type: ignore[index]
            assert x["right"] == s.right and 15 <= x["expiry"] - t <= 50
            assert w[k, 0] == (1.0 if s.right == "call" else 0.0)
            # The next write comes at the first decision on/after expiry.
            later = writes[writes > k]
            if later.size:
                assert steps[later[0]] == x["expiry"]


def test_covered_call_stays_covered() -> None:
    p = chain()
    steps = np.arange(5, T - 1)
    s = OptionWrite("call")
    w = s.targets(p, steps)
    sim = Simulator(p.market(), start=5, capital=1e7, **SIM)  # type: ignore[arg-type]
    for row in w:
        sim.step(row)
        if not np.isnan(row[0]) and (row[1:] < 0).any():
            q = sim.shares()
            shares, calls = q[0], q[1:].sum()
            assert calls == -round(shares / 100)  # whole contracts, same open
    trial = simulate(p, w, 5, name=s.name, capital=1e7, sim_kwargs=SIM)
    assert np.isfinite(trial.log_returns).all()
    # Writing calls caps the upside: never more than the stock plus premia.
    hold = np.full_like(w, np.nan)
    hold[0] = 0.0
    hold[0, 0] = 1.0
    stock = simulate(p, hold, 5, name="hold", capital=1e7, sim_kwargs=SIM)
    assert trial.summary.volatility_annual < stock.summary.volatility_annual


def test_put_write_is_cash_secured() -> None:
    p = chain()
    steps = np.arange(5, T - 1)
    w = OptionWrite("put").targets(p, steps)
    out = Simulator(p.market(), start=5, capital=1e7, **SIM).run(w)  # type: ignore[arg-type]
    assert out["margin_call"].sum() == 0 and out["liquidated"].sum() == 0
    assert (out["equity"] > 0).all()


def test_validation() -> None:
    with pytest.raises(ValueError):
        OptionWrite("straddle")
    with pytest.raises(ValueError):
        OptionWrite(coverage=1.5)
    t = np.arange(3, dtype=np.int64) * 86_400 * 10**9
    px = np.ones((3, 1))
    with pytest.raises(ValueError, match="instruments"):
        OptionWrite().targets(MarketPanel.from_arrays(["S"], t, px, px, px), np.arange(2))

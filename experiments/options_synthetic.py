"""Covered calls and put writes on SPY since 1996 with synthetic options:
Black-Scholes prices at the volatility the market was charging (VIX plus a
skew), against buy-and-hold, T-bills and the Cboe S&P 500 PutWrite index.

    python experiments/options_synthetic.py --store data/synth --cache data/http --out results/synth

The store is filled from Yahoo (SPY bars and actions, ^VIX, ^IRX, ^PUT)
unless ``--no-ingest``. Three ways to price the options show what the
volatility input does:

* ``vix_skew``: VIX-based at-the-money volatility with an S&P 500-like skew
  (the main case);
* ``vix_flat``: VIX-based, no skew;
* ``realized_flat``: the naive model, SPY's own trailing 21-day volatility
  and no skew. It leaves out the volatility risk premium.

Everything runs on one out-of-sample path with causality probes, with cash
earning the 13-week T-bill rate, under portfolio margin, and under three
cost assumptions.
"""

import argparse
import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from automoonbot.backtest import Fold, MarketPanel, Trial, compare, simulate, walk_forward_targets
from automoonbot.backtest.panel import session_dates
from automoonbot.backtest.synthetic import (
    Smile,
    SyntheticChain,
    aligned_bars,
    tbill_rate,
    trailing_dividend_yield,
)
from automoonbot.baselines import OptionWrite
from automoonbot.baselines.rules import trailing_vol
from automoonbot.data import ParquetStore
from automoonbot.data.providers.yahoo import YahooProvider

F64 = npt.NDArray[np.float64]
SPY = "ticker:SPY"
SIM: dict[str, object] = {
    "long_only": False,
    "max_gross": 2.5,
    "max_weight": 1.5,
    "margin": "portfolio",
}
STOCK = {"half_spread_bps": 0.5, "commission_bps": 0.2}
COSTS: dict[str, dict[str, float]] = {
    "frictionless": {
        "half_spread_bps": 0.0,
        "commission_bps": 0.0,
        "impact_coef": 0.0,
        "option_commission": 0.0,
        "option_half_spread_bps": 0.0,
        "option_min_half_spread": 0.0,
    },
    # Penny-wide SPY options: 1% of premium, at least half a cent.
    "tight": STOCK,
    # Pre-penny-pilot spreads: 3% of premium, at least 5 cents.
    "wide": STOCK | {"option_half_spread_bps": 300.0, "option_min_half_spread": 0.05},
}
WRITERS = (("call", 0.0), ("call", 0.02), ("put", 0.0), ("put", 0.02))


@dataclass(frozen=True)
class Inputs:
    panel: MarketPanel
    vix_open: F64
    vix_close: F64
    realized: F64
    rate: F64
    dividend_yield: F64
    put_index: F64


def ffill(x: F64) -> F64:
    idx = np.where(np.isfinite(x), np.arange(x.size), -1)
    idx = np.maximum.accumulate(idx)
    return np.where(idx >= 0, x[np.maximum(idx, 0)], np.nan)


def ingest(store: ParquetStore, cache: str | None) -> None:
    y = YahooProvider("1d", cache_dir=cache)
    y.ingest(store, "bars", ["SPY", "^VIX", "^IRX", "^PUT"])
    y.ingest(store, "corporate_actions", ["SPY"])


def load(store: ParquetStore) -> Inputs:
    panel = MarketPanel.from_store(store, [SPY], align="session")
    vix_o, vix_c = aligned_bars(store, "ticker:^VIX", panel)
    _, irx = aligned_bars(store, "ticker:^IRX", panel)
    _, put = aligned_bars(store, "ticker:^PUT", panel)
    rate = np.nan_to_num(ffill(tbill_rate(irx)), nan=0.0)
    lr = panel.log_returns()
    realized = trailing_vol(lr, 21)[:, 0] * np.sqrt(252) * 100
    panel = panel.with_rate(rate)
    return Inputs(
        panel,
        vix_o,
        vix_c,
        ffill(realized),
        rate,
        trailing_dividend_yield(panel, 0),
        put,
    )


def priced(inp: Inputs, model: str, right: str, otm: float, atm_ratio: float) -> MarketPanel:
    if model == "realized_flat":
        smile = Smile(atm_ratio=1.0, slope=0.0, curvature=0.0)
        # The open uses the volatility known at the previous close.
        index_c = inp.realized
        index_o = np.concatenate([[np.nan], inp.realized[:-1]])
    elif model == "vix_flat":
        smile = Smile(atm_ratio=atm_ratio, slope=0.0, curvature=0.0)
        index_o, index_c = inp.vix_open, inp.vix_close
    elif model == "vix_skew":
        smile, index_o, index_c = Smile(atm_ratio=atm_ratio), inp.vix_open, inp.vix_close
    else:
        raise ValueError(model)
    chain = SyntheticChain(levels=(otm,), rights=(right,), smile=smile)
    return chain.build(inp.panel, SPY, index_o, index_c, inp.rate, inp.dividend_yield)


@dataclass(frozen=True)
class WalkPath:
    start: int
    steps: npt.NDArray[np.int64]
    fold: list[Fold]
    expiries: npt.NDArray[np.int64]
    days: npt.NDArray[np.datetime64]
    #: Month-end rows from the start (the start row first).
    month_ends: npt.NDArray[np.int64]


def path(inp: Inputs, start_date: dt.date) -> WalkPath:
    p = inp.panel
    days = session_dates(p.times)
    exp = SyntheticChain().expiries(p, 0)
    start = int(exp[np.searchsorted(days[exp], np.datetime64(start_date))])
    steps = np.arange(start, p.shape[0] - 1, dtype=np.int64)
    rows = np.arange(start, p.shape[0])
    month = days[rows].astype("datetime64[M]")
    ends = rows[np.r_[month[1:] != month[:-1], True]]
    return WalkPath(
        start, steps, [Fold(np.arange(0, dtype=np.int64), steps)], exp, days, np.r_[start, ends]
    )


def yearly(trial: Trial, years: npt.NDArray[np.int64]) -> dict[int, float]:
    y = years[trial.periods]
    return {int(k): float(np.expm1(trial.log_returns[y == k].sum())) for k in np.unique(y)}


def levels(trial: Trial, pth: WalkPath, n: int) -> F64:
    """Equity (start = 1) at each row of the path, NaN elsewhere."""
    eq = np.full(n, np.nan)
    eq[pth.start] = 1.0
    eq[trial.periods] = np.exp(np.cumsum(trial.log_returns))
    return eq


def monthly(level: F64, pth: WalkPath) -> tuple[F64, npt.NDArray[np.datetime64]]:
    v = level[pth.month_ends]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.diff(np.log(v))
    return r, pth.days[pth.month_ends[1:]].astype("datetime64[M]")


def put_write(inp: Inputs, pth: WalkPath, atm_ratio: float, model: str = "vix_skew") -> Trial:
    q = priced(inp, model, "put", 0.0, atm_ratio)
    w = walk_forward_targets(q, OptionWrite("put"), pth.fold)[0]
    return simulate(
        q, w, pth.start, name="put_write", capital=1e6, sim_kwargs={**SIM, **COSTS["frictionless"]}
    )


def calibrate(inp: Inputs, pth: WalkPath, until: np.datetime64) -> float:
    """The at-the-money/VIX ratio at which the frictionless at-the-money put
    write earns the Cboe PutWrite index's mean monthly log return on the
    months before ``until`` (bisection; the return rises with the ratio)."""
    put, months = monthly(inp.put_index, pth)
    mask = (months < until) & np.isfinite(put)
    n = inp.panel.shape[0]

    def gap(ratio: float) -> float:
        ours, _ = monthly(levels(put_write(inp, pth, ratio), pth, n), pth)
        return float(np.mean(ours[mask] - put[mask]))

    lo, hi = 0.6, 1.2
    if not gap(lo) < 0 < gap(hi):
        raise ValueError("the calibration bracket does not straddle the index")
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if gap(mid) < 0 else (lo, mid)
    return 0.5 * (lo + hi)


def tracking(inp: Inputs, pth: WalkPath, trial: Trial, until: np.datetime64) -> dict[str, Any]:
    """Monthly fit of a strategy to the Cboe PutWrite index, before and
    after ``until``."""
    ours, months = monthly(levels(trial, pth, inp.panel.shape[0]), pth)
    put, _ = monthly(inp.put_index, pth)
    out: dict[str, Any] = {}
    for label, mask in (("calibration", months < until), ("out_of_sample", months >= until)):
        m = mask & np.isfinite(put) & np.isfinite(ours)
        a, b = ours[m], put[m]
        out[label] = {
            "months": int(m.sum()),
            "mean_annual": float(12 * a.mean()),
            "index_mean_annual": float(12 * b.mean()),
            "vol_annual": float(np.sqrt(12) * a.std(ddof=1)),
            "index_vol_annual": float(np.sqrt(12) * b.std(ddof=1)),
            "tracking_error": float(np.sqrt(12) * (a - b).std(ddof=1)),
            "correlation": float(np.corrcoef(a, b)[0, 1]),
        }
    return out


def period_stats(trial: Trial, bills: Trial, pth: WalkPath, until: np.datetime64) -> dict[str, Any]:
    """CAGR, volatility, max drawdown and Sharpe in excess of T-bills,
    before and after ``until``."""
    months = pth.days[trial.periods].astype("datetime64[M]")
    out: dict[str, Any] = {}
    for label, mask in (("calibration", months < until), ("out_of_sample", months >= until)):
        lr, rf = trial.log_returns[mask], bills.log_returns[mask]
        ex = np.expm1(lr) - np.expm1(rf)
        cum = np.cumsum(lr)
        out[label] = {
            "cagr": float(np.expm1(lr.mean() * 252)),
            "vol": float(lr.std(ddof=1) * np.sqrt(252)),
            "max_drawdown": float(
                1 - np.exp(np.min(cum - np.maximum.accumulate(np.r_[0.0, cum])[1:]))
            ),
            "excess_sharpe": float(ex.mean() / ex.std(ddof=1) * np.sqrt(252))
            if ex.std() > 0
            else 0.0,
        }
    return out


def run(
    inp: Inputs, start_date: dt.date, until: dt.date, audit: int, capital: float = 1e6
) -> dict[str, Any]:
    p = inp.panel
    pth = path(inp, start_date)
    cut = np.datetime64(until, "M")
    years = pth.days.astype("datetime64[Y]").astype(np.int64) + 1970
    ratio = calibrate(inp, pth, cut)
    print(f"calibrated at-the-money/VIX ratio: {ratio:.4f}", flush=True)

    # Buy-and-hold tops up to fully invested on the writers' roll dates, so
    # the participation cap on the first day cannot leave it in cash.
    hold = np.full((pth.steps.size, 1), np.nan)
    hold[np.isin(pth.steps, pth.expiries)] = 1.0
    hold[0] = 1.0
    base = {"buy_and_hold": (p, hold), "t_bills": (p, np.zeros((pth.steps.size, 1)))}
    out: dict[str, Any] = {
        "first_session": str(pth.days[pth.start + 1]),
        "last_session": str(pth.days[-1]),
        "sessions": int(pth.steps.size),
        "calibration_until": str(cut),
        "atm_ratio": ratio,
        "tracking": {
            m: tracking(inp, pth, put_write(inp, pth, ratio, m), cut)
            for m in ("vix_skew", "vix_flat", "realized_flat")
        },
        "models": {},
    }
    for model in ("vix_skew", "vix_flat", "realized_flat"):
        runs = dict(base)
        for right, otm in WRITERS:
            q = priced(inp, model, right, otm, ratio)
            s = OptionWrite(right, otm=otm)
            runs[s.name] = (q, walk_forward_targets(q, s, pth.fold, audit=audit)[0])
        res: dict[str, Any] = {}
        for label, extra in COSTS.items():
            trials = [
                simulate(q, w, pth.start, name=name, capital=capital, sim_kwargs={**SIM, **extra})
                for name, (q, w) in runs.items()
            ]
            by = {t.name: t for t in trials}
            rows = compare(trials)
            for r in rows:
                t = by[str(r["name"])]
                r["yearly"] = yearly(t, years)
                r["periods"] = period_stats(t, by["t_bills"], pth, cut)
            res[label] = rows
        out["models"][model] = res
        print(f"{model}: done", flush=True)
    return out


def table(rows: list[dict[str, Any]], period: str | None = None) -> str:
    head = "| strategy | CAGR | vol | max DD | excess Sharpe |\n|---|---:|---:|---:|---:|\n"
    order = ["buy_and_hold", "t_bills"] + [OptionWrite(r, otm=o).name for r, o in WRITERS]
    by = {r["name"]: r for r in rows}
    lines = []
    for n in order:
        if period is None:
            r = by[n]
            x = {
                "cagr": r["cagr"],
                "vol": r["volatility_annual"],
                "max_drawdown": r["max_drawdown"],
            }
            x["excess_sharpe"] = float("nan")
        else:
            x = by[n]["periods"][period]
        lines.append(
            f"| {n} | {x['cagr']:+.1%} | {x['vol']:.1%} | {x['max_drawdown']:.1%} | "
            f"{x['excess_sharpe']:.2f} |"
        )
    return head + "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="1996-09-01")
    ap.add_argument("--calibrate-until", default="2011-01-01")
    ap.add_argument("--audit", type=int, default=20)
    ap.add_argument("--no-ingest", action="store_true")
    args = ap.parse_args()
    store = ParquetStore(args.store)
    if not args.no_ingest:
        ingest(store, args.cache)
    res = run(
        load(store),
        dt.date.fromisoformat(args.start),
        dt.date.fromisoformat(args.calibrate_until),
        args.audit,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps(res["tracking"], indent=2))
    for model, by_cost in res["models"].items():
        for label, rows in by_cost.items():
            for period in ("calibration", "out_of_sample"):
                print(f"\n## {model}, {label} costs, {period}\n")
                print(table(rows, period))


if __name__ == "__main__":
    main()

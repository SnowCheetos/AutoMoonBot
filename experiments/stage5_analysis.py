"""Stage 5 diagnostics: where does the graph transformer's return come from?

Rebuilds the out-of-sample weights from the per-fold checkpoints of a
``stage5_sp500_daily.py`` run (checking that they reproduce the reported
returns), then reports:

* year-by-year log-growth against equal weight;
* alpha and beta against the equal-weight portfolio of the same universe;
* what the model holds: invested fraction, number of names, concentration,
  and the cross-sectional percentiles of its holdings' trailing volatility,
  12-1 momentum, 5-day return and dollar volume;
* control strategies that harvest the obvious tilts on the same
  (survivorship-biased) universe: high-volatility and low-volatility
  quintiles, and short-term (5-day) reversal.

Usage::

    python experiments/stage5_analysis.py --store data/d1 --universe sp500.csv --run results/s5
"""

import argparse
import json
import pickle
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from stage5_sp500_daily import add_universe_args, build_universe

from automoonbot.backtest import MarketPanel, simulate, walk_forward
from automoonbot.baselines.rules import _Rule, trailing_sum, trailing_vol
from automoonbot.models import GraphTransformerStrategy, TrainConfig
from automoonbot.models.transformer import GraphTransformer

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


def _quantile_ew(score: F64, ok: npt.NDArray[np.bool_], q: float, top: bool) -> F64:
    """Equal weight on the top (or bottom) ``q`` fraction of ``score`` per row."""
    s = np.where(ok & np.isfinite(score), score, np.nan)
    with np.errstate(invalid="ignore"):
        lo, hi = np.nanquantile(s, [q, 1 - q], axis=1, keepdims=True)
    pick = (s >= hi) if top else (s <= lo)
    n = pick.sum(axis=1, keepdims=True)
    return np.where(pick, 1.0 / np.maximum(n, 1), 0.0)


@dataclass
class VolQuintile(_Rule):
    top: bool = True
    window: int = 63
    rebalance: int = 21
    name: str = "high_vol_quintile"

    def weights(self, panel: MarketPanel) -> F64:
        vol = trailing_vol(panel.log_returns(), self.window)
        return _quantile_ew(vol, panel.investable, 0.2, self.top)


@dataclass
class Reversal(_Rule):
    """Equal weight on the worst 5-day performers (short-term reversal)."""

    lookback: int = 5
    rebalance: int = 1
    name: str = "reversal_5d"

    def weights(self, panel: MarketPanel) -> F64:
        r5, _ = trailing_sum(panel.log_returns(), self.lookback, self.lookback)
        return _quantile_ew(r5, panel.investable, 0.2, top=False)


def pct_rank(x: F64, ok: npt.NDArray[np.bool_]) -> F64:
    """Row-wise percentile rank in [0, 1] among valid entries."""
    v = np.where(ok & np.isfinite(x), x, np.nan)
    order = np.argsort(np.where(np.isnan(v), np.inf, v), axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(v.shape[0])[:, None]
    ranks[rows, order] = np.arange(v.shape[1])[None, :]
    n = np.isfinite(v).sum(axis=1, keepdims=True)
    return np.where(np.isfinite(v), ranks / np.maximum(n - 1, 1), np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_universe_args(ap)
    ap.add_argument("--run", required=True)
    ap.add_argument("--min-train-years", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--capital", type=float, default=1e6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    run = Path(args.run)

    panel, graph, _ = build_universe(args)
    t_bars, _ = panel.shape
    folds = list(
        walk_forward(
            t_bars,
            test_size=252,
            min_train=252 * args.min_train_years,
            max_label_horizon=21,
            embargo=5,
        )
    )

    # ---- rebuild the graph transformer's weights from its checkpoints
    gt = GraphTransformerStrategy(graph, train=TrainConfig(), bf16=True)
    layout = gt.layout(panel)
    gt.net = GraphTransformer(replace(gt.cfg, n_edge_types=layout.n_edge_types))
    rows = []
    t0 = time.time()
    for i, f in enumerate(folds):
        test = f.test[f.test < t_bars - 1]
        gt._update_vocab(layout, int(panel.decision_time[f.train[-1]]))
        gt.net.load_state_dict(torch.load(run / f"model_fold{i + 1}.pt"))
        rows.append(gt.targets(panel, test))
    w_gt = np.vstack(rows)
    start = int(folds[0].test[0])
    steps = np.arange(start, start + w_gt.shape[0])
    print(f"rebuilt weights in {time.time() - t0:.0f}s", flush=True)

    with open(run / "trials.pkl", "rb") as fh:
        reported = {t.name: t for t in pickle.load(fh)["default"]}
    mine = simulate(panel, w_gt, start, name="gt", capital=args.capital)
    ref = reported["graph_transformer"]
    gap = float(np.max(np.abs(mine.log_returns - ref.log_returns)))
    print(f"reproduction: max |daily return difference| = {gap:.2e}")

    # ---- controls on the same universe and span
    controls: dict[str, Any] = {}
    for s in (VolQuintile(top=True), VolQuintile(top=False, name="low_vol_quintile"), Reversal()):
        w = s.targets(panel, steps)
        controls[s.name] = simulate(panel, w, start, name=s.name, capital=args.capital)
    trials = {
        "graph_transformer": ref,
        "graph_transformer+overlay": reported["graph_transformer+overlay"],
        "equal_weight": reported["equal_weight"],
        "buy_and_hold": reported["buy_and_hold"],
        "cross_sectional_momentum": reported["cross_sectional_momentum"],
        **controls,
    }

    out: dict[str, Any] = {"reproduction_max_abs_diff": gap, "summary": {}, "yearly": {}}
    for k, t in trials.items():
        s = t.summary
        out["summary"][k] = {
            "log_growth_annual": s.log_growth_annual,
            "volatility_annual": s.volatility_annual,
            "sharpe_annual": s.sharpe_annual,
            "max_drawdown": s.max_drawdown,
            "turnover_annual": float(t.turnover.sum() / s.periods * 252),
            "cost_annual": float(t.costs.sum() / s.periods * 252),
        }

    # Alpha / beta of daily log returns against equal weight.
    ew = trials["equal_weight"].log_returns
    for k in (
        "graph_transformer",
        "graph_transformer+overlay",
        "high_vol_quintile",
        "reversal_5d",
        "cross_sectional_momentum",
    ):
        y = trials[k].log_returns
        beta, alpha = np.polyfit(ew, y, 1)
        resid = y - (alpha + beta * ew)
        ir = alpha / resid.std(ddof=1) * np.sqrt(252)
        tstat = alpha / (resid.std(ddof=1) / np.sqrt(y.size))
        out["summary"][k] |= {
            "beta_vs_ew": float(beta),
            "alpha_annual_vs_ew": float(alpha * 252),
            "alpha_t": float(tstat),
            "info_ratio_vs_ew": float(ir),
        }

    # Alpha after the tilts the controls capture: market (equal weight),
    # volatility (high minus low quintile), reversal and momentum spreads.
    fac = np.column_stack(
        [
            ew,
            trials["high_vol_quintile"].log_returns - trials["low_vol_quintile"].log_returns,
            trials["reversal_5d"].log_returns - ew,
            trials["cross_sectional_momentum"].log_returns - ew,
        ]
    )
    x = np.column_stack([np.ones(ew.size), fac])
    for k in ("graph_transformer", "graph_transformer+overlay"):
        y = trials[k].log_returns
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ coef
        cov = np.linalg.inv(x.T @ x) * resid.var(ddof=x.shape[1])
        out["summary"][k]["factor_model"] = {
            "alpha_annual": float(coef[0] * 252),
            "alpha_t": float(coef[0] / np.sqrt(cov[0, 0])),
            "loadings": dict(
                zip(
                    ["equal_weight", "vol_spread", "reversal", "momentum"],
                    map(float, coef[1:]),
                    strict=True,
                )
            ),
            "r2": float(1 - resid.var() / y.var()),
        }

    # Year by year.
    years = (
        panel.times[ref.periods].astype("datetime64[ns]").astype("datetime64[Y]").astype(int) + 1970
    )
    for yr in np.unique(years):
        m = years == yr
        out["yearly"][int(yr)] = {k: float(t.log_returns[m].sum()) for k, t in trials.items()}

    # What does it hold? NaN targets hold the previous weight (drift ignored).
    book = np.zeros_like(w_gt)
    cur = np.zeros(w_gt.shape[1])
    for i in range(w_gt.shape[0]):
        cur = np.where(np.isnan(w_gt[i]), cur, w_gt[i])
        book[i] = cur
    ok = panel.investable[steps]
    lr = panel.log_returns()
    vol = trailing_vol(lr, 63)[steps]
    r252, _ = trailing_sum(lr, 252, 200)
    r21, _ = trailing_sum(lr, 21, 15)
    mom = (r252 - r21)[steps]
    r5, _ = trailing_sum(lr, 5, 5)
    dv = (panel.close * panel.volume)[steps]
    inv = book.sum(axis=1)
    char = {}
    for name, x in (
        ("volatility_63d", vol),
        ("momentum_12_1", mom),
        ("return_5d", r5[steps]),
        ("dollar_volume", dv),
    ):
        pr = pct_rank(x, ok)
        wsum = np.where(np.isfinite(pr), book, 0.0)
        avg = (np.nan_to_num(pr) * wsum).sum(1) / np.maximum(wsum.sum(1), 1e-12)
        char[name] = float(np.nanmean(avg))
    out["holdings"] = {
        "invested_mean": float(inv.mean()),
        "invested_p10_p90": [float(np.quantile(inv, 0.1)), float(np.quantile(inv, 0.9))],
        "names_over_0.1pct_mean": float((book > 1e-3).sum(1).mean()),
        "effective_names_mean": float(np.mean(inv**2 / np.maximum((book**2).sum(1), 1e-12))),
        "max_weight_mean": float(book.max(1).mean()),
        "weighted_percentile": char,
    }
    (run / "analysis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["summary"], indent=1))
    print(json.dumps(out["holdings"], indent=1))
    for yr, v in out["yearly"].items():
        print(yr, " ".join(f"{k[:14]}={x:+.3f}" for k, x in v.items()))


if __name__ == "__main__":
    main()

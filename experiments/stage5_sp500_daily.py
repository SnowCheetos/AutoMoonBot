"""Stage 5: the graph transformer against rule baselines on S&P 500 daily bars.

Universe: the *current* S&P 500 constituents with their *current* GICS
sector and sub-industry. Both are survivorship-biased (no delisted or
removed members; today's classification asserted from each company's first
bar), which flatters every long-only strategy here alike; the comparison
between strategies on the same universe is the point, not the levels.

Walk-forward: yearly test windows, expanding training sets, labels purged by
the longest horizon plus an embargo. The model is fitted once per fold and
its targets are emitted with and without the drawdown overlay; every
strategy is scored under three cost assumptions from a single fit, and all
trials enter the deflated Sharpe ratio. Usage::

    python experiments/stage5_sp500_daily.py --store data/d1 --universe sp500.csv --out results/s5

The universe CSV needs the columns ``Symbol``, ``GICS Sector`` and
``GICS Sub-Industry``; the store must hold Yahoo daily bars and corporate
actions for those symbols.
"""

import argparse
import copy
import csv
import dataclasses
import datetime as dt
import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from automoonbot.backtest import MarketPanel, compare, simulate, walk_forward, walk_forward_targets
from automoonbot.backtest.evaluate import check_causal
from automoonbot.baselines import (
    BuyAndHold,
    Cash,
    CrossSectionalMomentum,
    EqualWeight,
    InverseVolatility,
)
from automoonbot.data import ParquetStore
from automoonbot.data.membership import Snapshots, coverage, require_history
from automoonbot.data.providers.yahoo import security_id
from automoonbot.graph import TemporalGraph, to_ns
from automoonbot.models import GraphTransformerStrategy, Overlay, TrainConfig
from automoonbot.models.universe import Membership, build_universe_graph

PPY = 252.0
COSTS: dict[str, dict[str, float]] = {
    "default": {},
    "tight": {"half_spread_bps": 0.5, "commission_bps": 0.2},
    "frictionless": {"half_spread_bps": 0.0, "commission_bps": 0.0, "impact_coef": 0.0},
}
OVERLAYS: dict[str, Overlay | None] = {"": None, "+overlay": Overlay()}


def load_universe(path: str) -> list[tuple[str, str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [
        (security_id(r["Symbol"].replace(".", "-")), r["GICS Sector"], r["GICS Sub-Industry"])
        for r in rows
    ]


def add_universe_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--store", required=True)
    ap.add_argument("--universe", required=True, help="current constituents CSV")
    ap.add_argument("--start", default="1998-01-01T00:00:00Z")
    ap.add_argument(
        "--membership",
        help="dated constituent snapshots (date,tickers CSV): point-in-time universe; "
        "the current list is appended as a final snapshot dated --as-of",
    )
    ap.add_argument("--as-of", default="2026-11-13", help="date of the current list")
    ap.add_argument(
        "--instruments",
        help="JSON {info: {symbol: {type}}} from the price source; non-EQUITY symbols are dropped",
    )
    ap.add_argument(
        "--sic",
        help="SIC codes per symbol (JSON from SEC submissions): classify every name by "
        "SIC industry (4-digit) and major group (2-digit) instead of today's GICS",
    )


def build_universe(
    args: argparse.Namespace,
) -> tuple[MarketPanel, TemporalGraph, dict[str, Any]]:
    """The panel (with point-in-time membership if requested), its
    classification graph, and a description of the universe."""
    uni = load_universe(args.universe)
    info: dict[str, Any] = {"membership": bool(args.membership)}
    if args.sic:
        with open(args.sic) as fh:
            sic = json.load(fh)["sic"]

        def classify(sym: str) -> tuple[str, str] | None:
            code = str((sic.get(sym) or {}).get("sic") or "")
            return (f"sic:{code}", f"sic2:{code[:2]}") if code.isdigit() else None

    else:
        gics = {u[0][7:].replace("-", "."): (u[2], u[1]) for u in uni}

        def classify(sym: str) -> tuple[str, str] | None:
            return gics.get(sym)

    snap = None
    symbols = [u[0][7:].replace("-", ".") for u in uni]
    if args.membership:
        snap = Snapshots.from_csv(args.membership).extended(
            dt.date.fromisoformat(args.as_of), symbols
        )
        since = dt.datetime.fromtimestamp(to_ns(args.start) // 10**9, dt.UTC).date()
        symbols = sorted(snap.symbols(since))
    if args.instruments:
        # Symbols whose current owner is not a common stock (e.g. an ETF that
        # took over a former member's ticker).
        with open(args.instruments) as fh:
            kinds = json.load(fh)["info"]
        info["not_equity"] = sorted(
            s for s in symbols if kinds.get(s, {}).get("type", "EQUITY") != "EQUITY"
        )
        symbols = [s for s in symbols if s not in set(info["not_equity"])]
    unclassified = [s for s in symbols if classify(s) is None]
    symbols = [s for s in symbols if classify(s) is not None]
    store = ParquetStore(args.store)
    ids = [security_id(s.replace(".", "-")) for s in symbols]
    panel = MarketPanel.from_store(store, ids, start=to_ns(args.start))
    has = panel.tradable.any(axis=0)
    if not has.all():
        keep = [s for s, h in zip(panel.securities, has, strict=True) if h]
        info["without_bars"] = sorted(
            s for s, h in zip(panel.securities, has, strict=True) if not h
        )
        panel = MarketPanel.from_store(store, keep, start=to_ns(args.start))
    info["unclassified"] = unclassified
    if snap is not None:
        mask = snap.mask(panel.times, panel.securities, lambda sid: sid[7:].replace("-", "."))
        mask = require_history(mask, panel.tradable)
        panel = panel.with_membership(mask)
        info["coverage"] = coverage(snap, panel.times, mask, panel.tradable)
    first = np.argmax(panel.tradable, axis=0)
    members = []
    for sid in panel.securities:
        ind, sec = classify(sid[7:].replace("-", ".")) or ("", "")
        members.append(Membership(sid, ind, sec))
    graph = build_universe_graph(members, panel.decision_time[first])
    info["securities"] = panel.shape[1]
    return panel, graph, info


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_universe_args(ap)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-train-years", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--capital", type=float, default=1e6)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--audit", type=int, default=2, help="causality probes per fold")
    ap.add_argument("--pilot", action="store_true", help="first fold only, short training")
    ap.add_argument("--train", default="{}", help="TrainConfig overrides (JSON)")
    ap.add_argument("--model", default="{}", help="ModelConfig overrides (JSON)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel, graph, info = build_universe(args)
    (out / "universe.json").write_text(json.dumps(info, indent=1))
    t_bars, n_sec = panel.shape
    print(f"panel {t_bars} bars x {n_sec} securities", flush=True)
    if "coverage" in info:
        print("member-day coverage by year:", {k: round(v, 3) for k, v in info["coverage"].items()})

    folds = list(
        walk_forward(
            t_bars,
            test_size=252,
            min_train=252 * args.min_train_years,
            max_label_horizon=21,
            embargo=5,
        )
    )
    if args.pilot:
        folds = folds[:1]
    train_cfg = TrainConfig(**json.loads(args.train))
    if args.pilot:
        train_cfg = dataclasses.replace(train_cfg, epochs=min(train_cfg.epochs, 4))
    model_kw: dict[str, Any] = json.loads(args.model)

    results: dict[str, tuple[np.ndarray, int, int, dict[str, object]]] = {}
    for s in (Cash(), BuyAndHold(), EqualWeight(), InverseVolatility(), CrossSectionalMomentum()):
        t0 = time.time()
        w, start, n = walk_forward_targets(panel, s, folds, audit=args.audit, seed=args.seed)
        results[s.name] = (w, start, n, dict(getattr(s, "params", {}) or {}))
        print(f"{s.name}: {time.time() - t0:.0f}s", flush=True)

    # The graph transformer: one fit per fold, targets per overlay variant
    # (each variant keeps its own carried book).
    gt = GraphTransformerStrategy(
        graph, model=model_kw, train=train_cfg, seed=args.seed, bf16=args.bf16
    )
    rows: dict[str, list[np.ndarray]] = {k: [] for k in OVERLAYS}
    carries: dict[str, dict[int, Any]] = {k: {} for k in OVERLAYS}
    rng = np.random.default_rng(args.seed)
    t_fit = 0.0
    for i, f in enumerate(folds):
        test = f.test[f.test < t_bars - 1]
        t0 = time.time()
        gt.fit(panel.truncate(int(test[0])), f.train)
        t_fit += time.time() - t0
        for k, ov in OVERLAYS.items():
            gt.overlay, gt._carry = ov, carries[k]
            w = gt.targets(panel, test)
            if args.audit:
                probes = sorted(rng.choice(test.size, size=args.audit, replace=False).tolist())
                check_causal(gt, panel, test, probes=probes, full=w)
            rows[k].append(w)
        last = [h for h in gt.history if h["fold"] == i + 1]
        print(
            f"fold {i + 1}/{len(folds)} test {test[0]}..{test[-1]}: {time.time() - t0:.0f}s, "
            f"epochs {len(last)}, best val {max(h.get('val_objective', np.nan) for h in last):.3f}",
            flush=True,
        )
        torch.save(gt.net.state_dict() if gt.net else {}, out / f"model_fold{i + 1}.pt")
    for k in OVERLAYS:
        start = int(folds[0].test[0])
        results[gt.name + k] = (np.vstack(rows[k]), start, len(folds), gt.params | {"overlay": k})
    (out / "history.json").write_text(json.dumps(gt.history, indent=1))

    report: dict[str, Any] = {
        "panel": {"bars": t_bars, "securities": n_sec, "start": args.start},
        "folds": len(folds),
        "fit_seconds": t_fit,
        "costs": {},
    }
    all_trials = {}
    for cost, kw in COSTS.items():
        trials = [
            simulate(
                panel,
                w,
                start,
                name=name,
                params=params,
                folds=n,
                capital=args.capital,
                periods_per_year=PPY,
                sim_kwargs=kw,
            )
            for name, (w, start, n, params) in results.items()
        ]
        all_trials[cost] = trials
        report["costs"][cost] = compare(trials)
        print(f"\n== costs: {cost}")
        for r in report["costs"][cost]:
            print(
                f"{r['name']:<32} logG {r['log_growth']:+.3f}  ann {r['log_growth_annual']:+.3f}  "
                f"vol {r['volatility_annual']:.3f}  maxDD {r['max_drawdown']:.3f}  "
                f"DSR {r['deflated_sharpe']:.2f}  turn {r['turnover_annual']:.1f}"
            )
    (out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    with open(out / "trials.pkl", "wb") as fh:
        pickle.dump(copy.deepcopy(all_trials), fh)


if __name__ == "__main__":
    main()

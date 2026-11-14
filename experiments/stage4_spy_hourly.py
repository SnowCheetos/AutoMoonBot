"""Stage 4: re-measure PGPerceptron out of sample on SPY hourly bars.

Reproduces the original setting (SPY, 1h bars, retrain every 50 bars on the
last 1000) with the walk-forward harness, and scores every strategy under
three cost assumptions from a single fit. Usage::

    python experiments/stage4_spy_hourly.py --store data/h1 --out results/seed0 --seed 0

The store must hold Yahoo hourly bars and corporate actions for SPY, e.g.::

    from automoonbot.data import ParquetStore
    from automoonbot.data.providers.yahoo import YahooProvider
    y = YahooProvider("1h")
    for ds in ("bars", "corporate_actions"):
        y.ingest(ParquetStore("data/h1"), ds, ["SPY"])
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from automoonbot.backtest import MarketPanel, compare, simulate, walk_forward, walk_forward_targets
from automoonbot.baselines import BuyAndHold, Cash, TrendFollowing
from automoonbot.baselines.pgperceptron import variants
from automoonbot.data import ParquetStore

PPY = 252 * 7  # regular-session hourly bars per year
COSTS: dict[str, dict[str, float]] = {
    "default": {},
    "spy_tight": {"half_spread_bps": 0.5, "commission_bps": 0.2},
    "frictionless": {"half_spread_bps": 0.0, "commission_bps": 0.0, "impact_coef": 0.0},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    panel = MarketPanel.from_store(ParquetStore(args.store), ["ticker:SPY"])
    folds = list(
        walk_forward(
            panel.shape[0], test_size=50, min_train=1000, max_label_horizon=1, train_window=1000
        )
    )
    strategies = [
        Cash(),
        BuyAndHold(),
        TrendFollowing(lookback=7 * 126, vol_window=7 * 21, periods_per_year=PPY, rebalance=7),
        *variants(seed=args.seed),
    ]
    results: dict[str, list] = {k: [] for k in COSTS}
    start = 0
    for s in strategies:
        t0 = time.time()
        targets, start, n = walk_forward_targets(panel, s, folds, audit=2)
        for k, kw in COSTS.items():
            results[k].append(
                simulate(
                    panel,
                    targets,
                    start,
                    name=s.name,
                    params=getattr(s, "params", None),
                    folds=n,
                    capital=1e6,
                    periods_per_year=PPY,
                    sim_kwargs=kw,
                )
            )
        print(f"{s.name}: {time.time() - t0:.0f}s", flush=True)
        if hasattr(s, "history"):
            (out / f"{s.name}.history.json").write_text(json.dumps(s.history))
    # Exposure-matched reference: half of SPY bought once and held.
    n_steps = results["default"][0].periods.size
    half = np.full((n_steps, 1), np.nan)
    half[0] = 0.5
    for k, kw in COSTS.items():
        results[k].append(
            simulate(
                panel,
                half,
                start,
                name="buy_and_hold_50",
                capital=1e6,
                periods_per_year=PPY,
                sim_kwargs=kw,
            )
        )
    tables = {k: compare(v) for k, v in results.items()}
    (out / "summary.json").write_text(json.dumps(tables, default=float, indent=1))
    with open(out / "trials.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()

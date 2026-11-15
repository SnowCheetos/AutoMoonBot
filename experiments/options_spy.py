"""Covered calls and put writes on SPY against buy-and-hold, on the two
years of option history the free Massive plan provides.

Two steps::

    # SPY bars and actions (Yahoo), then SPY monthly chains (Massive). About
    # 1,500 requests at 5 a minute (~5 hours); a rerun resumes from the cache.
    MASSIVE_API_KEY=... python experiments/options_spy.py ingest \\
        --store data/options --cache data/http

    python experiments/options_spy.py run --store data/options --out results/options-spy

The chain is point-in-time (a contract enters on the first close within 10%
of its strike), contracts are monthly with $5 strikes, and every strategy
runs on one out-of-sample path with causality probes, under portfolio
margin and two cost assumptions.
"""

import argparse
import datetime as dt
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.compute as pc

from automoonbot.backtest import Fold, MarketPanel, Trial, compare, simulate, walk_forward_targets
from automoonbot.backtest.panel import session_dates
from automoonbot.baselines import OptionWrite
from automoonbot.data import ParquetStore
from automoonbot.data.providers.massive import ChainSelection, MassiveProvider
from automoonbot.data.providers.yahoo import YahooProvider, security_id

SELECTION = ChainSelection(expirations="monthly", moneyness=0.10, max_days=70, strike_step=5.0)
#: The free plan's option history.
HISTORY = dt.timedelta(days=730)
SIM: dict[str, object] = {
    "long_only": False,
    "max_gross": 2.5,
    "max_weight": 1.5,
    "margin": "portfolio",
}
COSTS: dict[str, dict[str, float]] = {
    # Stock costs as in the stage 4 SPY study; option costs at their
    # defaults ($0.65 a contract, half-spread max(1%, $0.005)).
    "default": {},
    "spy_tight": {"half_spread_bps": 0.5, "commission_bps": 0.2},
}


def session_closes(store: ParquetStore, security: str) -> dict[dt.date, float]:
    """As-traded closes of ``security`` by New York session date."""
    bars = store.read("bars", filter=pc.field("security_id") == security, latest=True)
    if bars.num_rows == 0:
        raise ValueError(f"no bars for {security}")
    days = session_dates(bars.column("end").cast("int64").to_numpy())
    close = bars.column("close").to_numpy()
    return {d.astype(dt.date): float(c) for d, c in zip(days, close, strict=True)}


def ingest(
    store: ParquetStore,
    symbol: str,
    *,
    cache: str | None,
    today: dt.date,
    selection: ChainSelection = SELECTION,
    requests_per_minute: float = 5.0,
) -> dict[str, Any]:
    yahoo = YahooProvider("1d", cache_dir=cache)
    for ds in ("bars", "corporate_actions"):
        yahoo.ingest(store, ds, [symbol])
    closes = session_closes(store, security_id(symbol))
    # Expirations whose whole selection window is inside the history.
    start = today - HISTORY + dt.timedelta(days=selection.max_days)
    end = today - dt.timedelta(days=1)
    massive = MassiveProvider(requests_per_minute=requests_per_minute, cache_dir=cache)

    def progress(n: int, total: int, sid: str) -> None:
        if n % 25 == 0:
            print(f"  {n}/{total} {sid}", flush=True)

    res = massive.ingest_chain(
        store, symbol, closes, start=start, end=end, selection=selection, progress=progress
    )
    return {
        "expirations": [start.isoformat(), end.isoformat()],
        "listed": res.listed,
        "selected": res.selected,
        "bars_new": res.bars.new,
        "contracts_new": res.contracts.new,
    }


def build_panel(store: ParquetStore, symbol: str) -> MarketPanel:
    """The underlying and every stored contract on it, one row per session,
    from the first option bar (minus a volatility warm-up) on."""
    sid = security_id(symbol)
    terms = store.read("option_contracts", filter=pc.field("underlying_id") == sid, latest=True)
    if terms.num_rows == 0:
        raise ValueError(f"no option contracts on {sid}")
    options = sorted(terms.column("security_id").to_pylist())
    first = store.read(
        "bars", filter=pc.field("security_id").isin(options), columns=["end"]
    ).column("end")
    warmup = 60 * 86_400 * 10**9  # trailing vol and volume need history
    start = int(pc.min(first).cast("int64").as_py()) - warmup
    panel = MarketPanel.from_store(store, [sid, *options], start=start, align="session")
    return panel.with_contracts(store)


def strategies() -> list[OptionWrite]:
    return [
        OptionWrite("call", otm=0.0),
        OptionWrite("call", otm=0.02),
        OptionWrite("put", otm=0.0),
        OptionWrite("put", otm=0.02),
    ]


def run(
    panel: MarketPanel,
    *,
    capital: float = 1e7,
    costs: Mapping[str, Mapping[str, float]] = COSTS,
    audit: int = 20,
    writers: Sequence[OptionWrite] | None = None,
) -> dict[str, Any]:
    """Every strategy on one path starting at the first session with an
    option bar, scored under each cost assumption."""
    t_bars = panel.shape[0]
    opt_cols = np.array([i for i, s in enumerate(panel.securities) if s.startswith("option:")])
    first = int(np.argmax(panel.tradable[:, opt_cols].any(axis=1)))
    if first >= t_bars - 2:
        raise ValueError("not enough sessions after the first option bar")
    steps = np.arange(first, t_bars - 1, dtype=np.int64)
    fold = [Fold(np.arange(0, dtype=np.int64), steps)]
    targets: dict[str, np.ndarray] = {}
    hold = np.full((steps.size, panel.shape[1]), np.nan)
    hold[0] = 0.0
    hold[0, 0] = 1.0
    targets["buy_and_hold"] = hold
    for s in writers if writers is not None else strategies():
        targets[s.name] = walk_forward_targets(panel, s, fold, audit=audit)[0]
    results: dict[str, Any] = {
        "sessions": int(steps.size),
        "first_session": str(session_dates(panel.times[first : first + 1])[0]),
        "contracts": int(opt_cols.size),
        "stale_share": stale_share(panel, targets),
        "costs": {},
    }
    for label, extra in costs.items():
        trials: list[Trial] = [
            simulate(
                panel,
                w,
                first,
                name=name,
                capital=capital,
                sim_kwargs={**SIM, **extra},
            )
            for name, w in targets.items()
        ]
        results["costs"][label] = compare(trials)
    return results


def stale_share(panel: MarketPanel, targets: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Share of the written contracts' marking sessions with no trade (a
    carried-forward close): how much of each strategy's option P&L rests on
    stale prices."""
    out: dict[str, float] = {}
    for name, w in targets.items():
        written = np.flatnonzero(np.nan_to_num(w[:, 1:], nan=0.0).min(axis=0) < 0) + 1
        if written.size == 0:
            continue
        live = panel.listed[:, written]
        out[name] = float((live & ~panel.tradable[:, written]).sum() / max(live.sum(), 1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ingest")
    a.add_argument("--store", required=True)
    a.add_argument("--cache", default=None)
    a.add_argument("--symbol", default="SPY")
    a.add_argument("--rpm", type=float, default=5.0)
    r = sub.add_parser("run")
    r.add_argument("--store", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--symbol", default="SPY")
    args = ap.parse_args()
    store = ParquetStore(args.store)
    if args.cmd == "ingest":
        today = dt.datetime.now(dt.UTC).date()
        info = ingest(
            store, args.symbol, cache=args.cache, today=today, requests_per_minute=args.rpm
        )
        print(json.dumps(info, indent=2))
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = run(build_panel(store, args.symbol))
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    for label, rows in results["costs"].items():
        print(f"\n{label}")
        for row in rows:
            print(
                f"  {row['name']:<22} cagr {row['cagr']:+.3f}  vol {row['volatility_annual']:.3f}"
                f"  sharpe {row['sharpe_annual']:+.2f}  maxdd {row['max_drawdown']:.3f}"
                f"  cvar95 {row['cvar_95']:+.4f}"
            )


if __name__ == "__main__":
    main()

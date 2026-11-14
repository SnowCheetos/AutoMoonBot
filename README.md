![](media/logo_100.png)

-----

This repository contains code and resources for automated algorithmic trading implementations. All content provided in this repository is for educational and informational purposes only. It is **NOT** intended to be a financial product or investment advice. By viewing, citing, using, or contributing to the contents and resources provided in this repository, you automatically acknowledge and agree to all terms and conditions found in [`DISCLAIMERS`](docs/DISCLAIMER.md), both present and future, unless explicitly stated otherwise.

# Table of Contents

**Please note that this project is still super-pre-alpha, so hold your bags for now...**

- [`Introduction`](#introduction)

- [`Architecture`](#architecture)

- [`Usage`](#usage)

- [`Roadmap`](#roadmap)

- [`Legacy`](#legacy)

- [`Contribution`](#contribution)

- [`Citation`](#citation)

# Introduction

## Inspiration

[The Medallion Fund](https://www.cornell-capital.com/blog/2020/02/medallion-fund-the-ultimate-counterexample.html) is one of the core inspirations for this project. It showed that quantitative trading, when executed perfectly, has enormous potential for success and profit. The fund averaged an astonishing annualized return of **63.3%** for its duration, and it didn't have a single negative year, even during the **2000 dot-com bubble** and **2008 financial crisis**. This level of success caused many to question the highly controversial [Efficient Market Hypothesis](https://www.investopedia.com/terms/e/efficientmarkethypothesis.asp), whose definition will squash the hopes and dreams of any u̶n̶e̶m̶p̶l̶o̶y̶e̶d̶ ̶e̶c̶o̶n̶ ̶m̶a̶j̶o̶r̶ traders who aims to beat the market with technical analysis.

Fortunately, t̶h̶e̶i̶r̶ ̶p̶a̶r̶e̶n̶t̶s̶ ̶s̶t̶i̶l̶l̶ ̶l̶o̶v̶e̶ ̶t̶h̶e̶m̶ the efficient market hypothesis has been proven wrong on many occasions (*although I question that every time I make a trade*). Still, a majority of the funds do not manage to beat the market consistently, not to mention retail investors. Point being, it's possible to beat the market, but possible doesn't remotely equate probable in this case. Now, I'm no [Jim Simons](https://en.wikipedia.org/wiki/Jim_Simons) d̶u̶h̶ (*peace be with him* 🙏🏼), and the Medallion Fund keeps most details of its methodologies confidential, but Jim Simons and others have provided some valuable information to the public, maybe just enough to make m̶e̶ ̶r̶i̶c̶h̶ a difference. I have summarized them into four points (*note that a lot of them are my personal interpretations*)

### 1. It's Just Math
-----
Algorithmic trading is at the core of the Medallion Fund, with Jim Simons himself being a renowned mathematician in his earlier life. Their approach was almost purely from the numerical perspective, rather than finance. S̶o̶r̶r̶y̶ ̶p̶a̶l̶,̶ ̶t̶h̶a̶t̶ ̶e̶c̶o̶n̶ ̶d̶e̶g̶r̶e̶e̶ ̶t̶h̶a̶t̶ ̶y̶o̶u̶ ̶"̶w̶o̶r̶k̶e̶d̶ ̶h̶a̶r̶d̶"̶ ̶f̶o̶r̶ ̶i̶s̶ ̶n̶o̶t̶ ̶g̶o̶n̶n̶a̶ ̶p̶a̶y̶ ̶o̶f̶f̶ ̶t̶h̶e̶ ̶w̶a̶y̶ ̶y̶o̶u̶ ̶t̶h̶i̶n̶k̶

### 2. Everything, Anything
-----
Jim Simons have stated explicitly in public that the fund makes extensive use of publically available data to train their model. Anything from news to interest rates were taken into consideration, and much of it **before** the wide adaption of the internet, so they spent a lot of time copying data by hand. S̶o̶ ̶i̶t̶'̶s̶ ̶t̶i̶m̶e̶ ̶f̶o̶r̶ ̶y̶o̶u̶ ̶t̶o̶ ̶s̶t̶o̶p̶ ̶w̶h̶i̶n̶i̶n̶g̶ ̶a̶b̶o̶u̶t̶ ̶b̶e̶i̶n̶g̶ ̶b̶r̶o̶k̶e̶

### 3. Win Small or Lose Big
-----
For the duration of the fund's existence, their model had an average win rate of $`50.75\%`$. That's right, they operated on a tiny $`0.75\%`$ edge over the market, which was enough to turn $`\$100`$ in $`\text{1982}`$ to $`\$88,190,084,543`$ in $`\text{2024}`$. Instead of trying to count cards at Blackjack, they just opened a casino. Staying consistent over millions of trades turned out to be much more advantageous than trying to win big on a few trades.

### 4. Focus on The Right Thing
-----
Many people are willing to take on exceptional risks in pursue of, at least theoretically, higher possible returns, l̶i̶k̶e̶ ̶t̶h̶a̶t̶ ̶o̶n̶e̶ ̶w̶e̶e̶k̶l̶y̶ ̶F̶D̶ ̶y̶o̶u̶ ̶l̶o̶s̶t̶ ̶y̶o̶u̶r̶ ̶l̶u̶n̶c̶h̶ ̶m̶o̶n̶e̶y̶ ̶o̶n̶ ̶l̶a̶s̶t̶ ̶F̶r̶i̶d̶a̶y̶, and in some cases, risk taking is even being glorified (*not pointing any fingers* 👉🏻 [`👀`](https://www.reddit.com/r/wallstreetbets/)). Too often do m̶y̶s̶e̶l̶f̶ traders focus solely on potential (*not even real*) profits while completely ignoring the associated risk. The Medallion Fund, on the other hand, placed a tremendous emphasis on risk reduction. The fund consistently hedged the portfolio near perfectly, which enabled them to use margins that would be otherwise considered risky. 

## Approach

The Medallion Fund publically stated numerous times that they collect nearly all publically available data, anything that could remotely affect the price actions of an asset was included. Having the data is the easy part; making effective use of data that comes in wildly different forms (news text, filings, prices, relationships between companies) is the hard part.

This project represents that world as a **temporal heterogeneous graph**: companies, securities, publishers, articles and filings are nodes; real relationships (issuer, supplier/customer, mentions, ...) are typed edges. A graph model reads point-in-time snapshots of that graph and an actor-critic policy makes every trading decision. Large language models are used **offline only**, to turn text into node features and typed relationship edges; they are never in the trading loop.

Two rules come before any model:

1. **No look-ahead, by construction.** Every fact carries a *valid time* (when it is true in the world) and a *known time* (when it could first have been known). A snapshot "as of" `T` can only see facts known by `T`. Restatements, delistings and ended relationships become visible exactly when they were knowable, and the universe is reconstructed point in time (no survivorship bias).
2. **Risk first.** The objective is log-growth / return per unit of CVaR under a drawdown cap, measured strictly out of sample, against honest baselines.

# Architecture

```
moonrs/                 Rust core (bitemporal graph store, snapshots, persistence)
  src/time.rs           Timestamps (UTC ns) and half-open validity intervals
  src/schema.rs         Runtime schema: node kinds, feature channels, relations
  src/versions.rs       Bitemporal interval facts (existence, edges)
  src/series.rs         Bitemporal observation series (revisions, history)
  src/store.rs          Atomic, idempotent batch ingestion
  src/snapshot.rs       Point-in-time materialization (+ homogeneous token view)
  src/persist.rs        Checksummed, versioned, atomically written store files
  tests/leakage.rs      Property tests against an independent reference model
python/automoonbot/     Python package (wrapper, PyG / graph-transformer inputs)
  data/schemas.py       Canonical point-in-time datasets (bars, filings, facts, articles, relations, ...)
  data/store.py         Append-only, conflict-checked, as-of-readable Parquet store
  data/providers/       Provider interface + SEC EDGAR (acceptance-time knowledge)
  data/fundamentals.py  Bitemporal quarterly fundamentals from as-reported XBRL
  data/ingest.py        Loaders from the store into the temporal graph
  backtest/             Market panels, labels, walk-forward, metrics (simulator in moonrs/src/backtest.rs)
tests/                  Python tests
legacy/                 Earlier generations of the project
```

**Snapshots.** `snapshot(known_at, valid_at=None, history=..., homogeneous=...)` returns, per node kind, rows sorted by key with `[N, K, D]` feature tensors (chronological, right-aligned history), presence masks, and the valid/known time of every slot; per relation, an `edge_index`, attributes and temporal fields. With `homogeneous="reverse"` it also returns a flat token graph (global token ids, node and edge type ids, CSR adjacency) for graph transformers. Output is deterministic and independent of ingestion order.

**Ingestion** is strict: shapes, finiteness and declared bounds are validated; two different facts with the same key and knowledge time are rejected (corrections must carry a later knowledge time); every batch is atomic; replaying identical rows is a no-op.

**Performance** (4-core cloud VM, `cargo run --release --example bench`): 3,000 securities with 10 years of daily bars (7.6M rows) ingest at ~6.5M rows/s; a full snapshot with 30k supply-chain edges and the trailing 3-day article window takes ~3 ms, or ~9 ms with 64 days of bar history per security.

# Usage

Requires Rust ≥ 1.97 (edition 2024) and Python ≥ 3.14.

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop --release --extras dev      # add ,torch for PyG conversion
pytest                                       # Python tests
(cd moonrs && cargo test --release)          # Rust unit + property tests
```

```python
import numpy as np
from automoonbot import Channel, TemporalGraph

g = TemporalGraph()
g.define_entity("company", [Channel("bar", 5)])
g.define_event("article", window=np.timedelta64(3, "D"), channels=[Channel("embedding", 768)])
g.define_relation("supplies", "company", "company", attr_dim=1, bounds=(0.0, 1.0))
g.define_relation("mentions", "article", "company", attr_dim=2)

g.upsert_nodes("company", ["TSMC", "AAPL"], known_time="2020-01-01T00:00:00Z")
# A daily bar is valid at the close and known when the vendor publishes it.
g.observe("company", "bar", "AAPL",
          valid_time="2024-03-01T21:00:00Z", known_time="2024-03-01T21:15:00Z",
          values=[180.0, 182.0, 179.5, 181.2, 5.1e7])
# A relationship disclosed in a filing accepted on 2024-02-02.
g.assert_edges("supplies", "TSMC", "AAPL", known_time="2024-02-02T21:05:00Z",
               valid_from="2023-09-30T00:00:00Z", attrs=[[0.25]])

snap = g.snapshot("2024-03-04T14:30:00Z", history={"company.bar": 64}, homogeneous="reverse")

from automoonbot.pyg import to_hetero_data, to_token_graph   # needs torch + torch_geometric
data = to_hetero_data(snap)       # heterogeneous GNN input
tokens = to_token_graph(snap)     # graph-transformer input
```

## Data

Every source is normalized into canonical Arrow schemas whose rows carry a `known_time` and a `source`, and lands in an append-only Parquet store with the same rules as the graph (idempotent replays, conflicting versions rejected, as-of reads). Providers declare a knowledge policy (`exact`, `conservative`, or `estimated`, which is never used for evaluation). Prices are stored unadjusted; split/dividend adjustment is applied point in time.

The first provider is SEC EDGAR: filings with acceptance timestamps, XBRL company facts, and issuer identity. Quarterly fundamentals are derived per fiscal quarter (including Q4 = FY − 9M and year-to-date cash-flow differencing), and every restatement becomes a new observation at the moment it was filed.

```python
from automoonbot.data import ParquetStore
from automoonbot.data.providers.edgar import EdgarProvider
from automoonbot.data.ingest import define_core_schema, load_entities, load_filings, load_fundamentals

store = ParquetStore("data/")
edgar = EdgarProvider(user_agent="Your Name you@example.com", cache_dir="data/.http")  # SEC policy
for dataset in ("entities", "filings", "facts"):
    edgar.ingest(store, dataset, ["320193", "1045810"])

g = TemporalGraph()
define_core_schema(g)
load_entities(store, g); load_filings(store, g); load_fundamentals(store, g)
```

## Backtesting

The execution simulator lives in Rust and fixes the timing so it cannot be gotten wrong by a caller: a decision made after bar `t` closes executes at the **open of bar `t + 1`**, sized and priced only with quantities known before that open (trailing ADV and volatility). It models commission, half-spread, square-root impact, a participation cap, borrow fees, cash/margin financing, splits, dividends, halts (frozen positions) and delistings (forced exit with a haircut), and enforces long-only / gross / per-name constraints.

```python
from automoonbot.backtest import MarketPanel, Simulator, forward_returns, summary, walk_forward

panel = MarketPanel.from_store(store, ["AAPL", "MSFT", "NVDA"])     # unadjusted bars + corporate actions
sim = Simulator(panel.market(), capital=1.0, half_spread_bps=2.0)
report = sim.run(weights)                                           # [steps, S] target weights
print(summary(report["log_return"]))                                # log-growth, CVaR95, max DD, PSR, ...

labels = forward_returns(panel, horizons=[1, 5, 21])                # multi-horizon targets (labels only)
for fold in walk_forward(panel.shape[0], test_size=63, min_train=504, max_label_horizon=21, embargo=5):
    ...
```

Every strategy, rule or learned, implements the same `Strategy` protocol and is scored by the same walk-forward harness: each fold is fitted on a panel truncated before its test window, the test windows form one continuous simulated path, and `check_causal` re-computes targets on truncated panels to prove no future bar was used.

```python
from automoonbot.baselines import report

trials, table = report.run(panel, periods_per_year=252)   # baselines + PGPerceptron variants
```

Metrics lead with log-growth, CVaR and drawdown; Sharpe is reported with its probabilistic and deflated versions, so a result found by searching many configurations is judged against the best of that many unskilled ones.

# Roadmap

See [`ROADMAP`](docs/ROADMAP.md).

# Legacy

The current version of this project stands on the shoulders of... its past versions? Oh and giants too. Jokes aside, it went through a few significant iterations in the past, previous approaches used were very different. If you would like to view the legacy code and implementations for more insights, go to the `legacy` directory, or use the links below.

- [`Automated Trading with Policy Gradient`](https://github.com/MedallionDream/PolicyGradient)
- [`Automated Trading with Classical Ensemble`](https://github.com/MedallionDream/ClassicalEnsemble)

# Contribution

Like software and finance? Have ideas on how to improve this project? Consider contributing! Refer to [`CONTRIBUTING`](docs/CONTRIBUTING.md) for more information.

# Citation

If enjoyed the contents in this repository, found it helpful, or at least entertaining, please consider starring ⭐ s̶o̶ ̶I̶ ̶c̶a̶n̶ ̶f̶e̶e̶l̶ ̶b̶e̶t̶t̶e̶r̶ ̶a̶b̶o̶u̶t̶ ̶m̶y̶s̶e̶l̶f̶ to show your support. If you plan on using the contents in another project, please cite this repository. You can use the following `BibTeX` entry:

```bibtex
@misc{SnowCheetos2024,
    author = {Zhihao (Cass) Sheng},
    title = {Algorithmic trading using heterogeneous graph neural network and reinforcement learning {AutoMoonBot}},
    year = {2024},
    publisher = {GitHub},
    journal = {GitHub Repository},
    howpublished = {\url{https://github.com/SnowCheetos/AutoMoonBot}},
}
```

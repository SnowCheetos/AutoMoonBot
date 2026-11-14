# Roadmap

The project is being rebuilt in stages, each landing as its own pull request.

1. **Temporal core** *(this stage)*. `moonrs` rewritten as a bitemporal,
   point-in-time heterogeneous graph store: valid/known time on every fact,
   interval-versioned nodes and edges, revision-aware series, deterministic
   snapshots with a graph-transformer token view, atomic ingestion, checksummed
   persistence, property tests against a reference model.
2. **Data layer.** Provider abstraction and canonical schemas (bars, corporate
   actions, filings, articles, relationships) with a Parquet store; SEC EDGAR
   adapter (acceptance timestamps give true knowledge times). The Alpha Vantage
   client is retired.
3. **Backtester.** Event-driven, next-bar execution with commissions, spread,
   slippage and borrow; walk-forward evaluation with purging and embargo;
   log-growth, CVaR, max drawdown, deflated Sharpe. A multi-horizon labeler kept
   strictly separate from features.
4. **Baselines.** Cash, buy-and-hold, equal-weight, inverse-volatility,
   cross-sectional momentum, trend following, and PGPerceptron re-implemented
   faithfully (as published, and with its four porting-time bugs repaired,
   with in-sample or validation model selection), all on one walk-forward
   out-of-sample path with causality audits and deflated Sharpe. *Harness and
   strategies done. Result (SPY hourly, three seeds): no PGPerceptron variant
   beats an exposure-matched passive position; see
   `docs/results/stage4-spy-hourly.md`.*
5. **Model.** A graph transformer over company/industry/sector tokens
   (PatchTST-style temporal encoder, GPS-style layers: typed message passing
   plus global attention) with MTP-style multi-horizon Student-t heads and a
   policy head trained directly on cost-adjusted log-growth with a CVaR
   penalty (differentiable, so no critic), plus an optional drawdown overlay.
   *First result on S&P 500 daily (current constituents): beats every
   baseline and factor control (alpha about 6-8% a year, t about 3), but the
   universe is survivorship-biased and the model tilts to high-volatility
   names; see `docs/results/stage5-sp500-daily.md`. Next: re-run on a
   point-in-time universe with delisted names; heterogeneous-GNN and no-graph
   ablations; seeds.*
6. **Offline LLM enrichment.** Article embeddings and structured event fields as
   node features; typed relationship edges extracted from filings and news, each
   stamped with the time the extraction could have been made. Evaluation only on
   periods after the models' training cutoffs.
7. **Propagation validation.** Narrow, falsifiable tests first, e.g.
   supplier → customer spillover after earnings surprises, before any portfolio
   claims.

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
   strictly separate from features. *Extended with options, futures and
   margin (Reg T and portfolio margin, forced liquidation) and a
   point-in-time option-chain provider; see `docs/derivatives.md`. Open: two
   years of SPY chains on the free data plan, then covered-call and
   put-write baselines against buy-and-hold. Synthetic Black-Scholes chains
   at VIX-implied volatility cover 1996–2026 meanwhile; see
   `docs/results/options-synthetic.md`.*
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
   *On the current S&P 500 it beats every baseline and factor control
   (alpha about 6-8% a year, t about 3), but on a point-in-time S&P 500
   (historical membership, former members' prices, SIC classification) the
   edge is gone: +10.5% a year against +10.7% for equal weight, factor alpha
   -1.0% (t -0.5). The survivor-only result was survivorship bias; see
   `docs/results/stage5-pit.md`. Open: delisting returns and the remaining
   coverage holes (55-70% of member-days before 2014), a survivor-only run
   with SIC to isolate the classification change, heterogeneous-GNN and
   no-graph ablations, seeds.*
6. **Offline LLM enrichment.** Article embeddings and structured event fields as
   node features; typed relationship edges extracted from filings and news, each
   stamped with the time the extraction could have been made. Evaluation only on
   periods after the models' training cutoffs.
7. **Propagation validation.** Narrow, falsifiable tests first, e.g.
   supplier → customer spillover after earnings surprises, before any portfolio
   claims.

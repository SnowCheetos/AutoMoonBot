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
   strategies done; the real-data measurement needs a price source.*
5. **Model.** A graph transformer over snapshot tokens (typed sparse attention,
   temporal and structural encodings) with MTP-style multi-horizon prediction
   heads, feeding an actor-critic policy trained on a risk-first objective
   (log-growth with a CVaR penalty and a drawdown cap). A heterogeneous GNN is
   kept as an ablation baseline. Training and inference throughput are
   first-class requirements.
6. **Offline LLM enrichment.** Article embeddings and structured event fields as
   node features; typed relationship edges extracted from filings and news, each
   stamped with the time the extraction could have been made. Evaluation only on
   periods after the models' training cutoffs.
7. **Propagation validation.** Narrow, falsifiable tests first, e.g.
   supplier → customer spillover after earnings surprises, before any portfolio
   claims.

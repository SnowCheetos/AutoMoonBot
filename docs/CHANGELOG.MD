
## 0.2.0 (unreleased)

- Rewrote `moonrs` as a bitemporal point-in-time graph store (see `docs/ROADMAP.md`, stage 1).
- Removed the 2024 prototype (`automoonbot/moonrs`, `automoonbot/moonpy`), the Alpha Vantage client and Redis streamer; they remain in git history.
- New Python package layout built with maturin (`python/automoonbot`).
- Data layer: canonical point-in-time schemas, Parquet store, provider interface, SEC EDGAR provider, bitemporal quarterly fundamentals, graph loaders.
- Backtesting: Rust execution simulator (next-open fills, costs, corporate actions, halts, delistings), market panels, multi-horizon total-return labels, purged/embargoed walk-forward, risk-first metrics with probabilistic and deflated Sharpe.
- Simulator: `NaN` target weights hold the current shares (no trade), so discrete-trade strategies let weights drift.
- `MarketPanel`: high/low, `truncate`, causal `log_returns` and `total_return_index`.
- Evaluation harness (`backtest.evaluate`): `Strategy` protocol, fitting on panels truncated before each test window, one continuous out-of-sample path, `check_causal` truncation audits, `compare` with deflated Sharpe across trials.
- Baselines: cash, buy-and-hold, equal-weight, inverse-volatility, cross-sectional momentum, trend following; `baselines.report.run` for the standard comparison.
- PGPerceptron port (`baselines.pgperceptron`): vectorized descriptors verified against the original formulas, literal trade-manager port, same policy network, REINFORCE with batched recomputation of log-probabilities, the server's retrain/deploy rule; `legacy` and `repaired` variants, `in_sample` and `validation` selection.
- Yahoo Finance provider; full history is requested as an explicit period and bar granularity is verified (`range=max` silently returns weekly or monthly bars).
- Filings are keyed by `(entity_id, accession)`: co-registrant filings share an accession.
- Models (`automoonbot.models`): causal cross-sectional price features, company/industry/sector universe graph read point-in-time, graph transformer (temporal patch encoder, GPS-style layers, chained Student-t horizon heads, capped long-only policy), risk-first differentiable training with sub-universe sampling and warm-started walk-forward folds, `GraphTransformerStrategy` with bit-exact causal inference, no-trade band and drawdown overlay.
- Point-in-time universe membership (`data.membership`): dated constituent snapshots, per-bar membership masks, coverage by year, and `require_history` to drop runs whose price series belongs to a later owner of a reused ticker. `MarketPanel.member` and `investable`; features, baselines and the graph transformer only hold members, and keep non-member prices as history.
- Derivatives and margin (`docs/derivatives.md`): options (American/European, physical or cash settlement, exercise and assignment at expiry, rational early exercise, OCC split adjustments), futures (daily variation margin, close or next-open final settlement), per-contract costs and lots, contract-count targets (`units=True`), and margin models (`none`, Reg T, TIMS-style portfolio margin with Black-Scholes/Black-76 repricing at implied vol) with initial-margin cutbacks and maintenance liquidations. Equity-only backtests are unchanged.
- Massive (Polygon) options provider: monthly chains selected point-in-time (a contract enters on the first close near its strike), unadjusted daily bars, `option_contracts` dataset; the API key travels only in the `Authorization` header. `MarketPanel.from_store(align="session")` joins 16:15 option bars with 16:00 stock bars, and `with_contracts` reads option terms from the store. `from_store` now rejects two bars of one security in one row instead of silently keeping one.
- Experiments and results: `experiments/stage4_spy_hourly.py`, `experiments/stage5_sp500_daily.py`, `experiments/stage5_analysis.py`, `docs/results/` (Stage 5 on a point-in-time S&P 500: no edge over equal weight).

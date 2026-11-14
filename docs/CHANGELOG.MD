
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

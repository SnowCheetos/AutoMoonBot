
## 0.2.0 (unreleased)

- Rewrote `moonrs` as a bitemporal point-in-time graph store (see `docs/ROADMAP.md`, stage 1).
- Removed the 2024 prototype (`automoonbot/moonrs`, `automoonbot/moonpy`), the Alpha Vantage client and Redis streamer; they remain in git history.
- New Python package layout built with maturin (`python/automoonbot`).
- Data layer: canonical point-in-time schemas, Parquet store, provider interface, SEC EDGAR provider, bitemporal quarterly fundamentals, graph loaders.
- Backtesting: Rust execution simulator (next-open fills, costs, corporate actions, halts, delistings), market panels, multi-horizon total-return labels, purged/embargoed walk-forward, risk-first metrics with probabilistic and deflated Sharpe.

"""Point-in-time backtesting: market panels, the Rust execution simulator,
multi-horizon labels, walk-forward splits, and risk-first metrics."""

from automoonbot._moonrs import Market, Simulator
from automoonbot.backtest.labels import Labels, forward_returns
from automoonbot.backtest.metrics import Summary, summary
from automoonbot.backtest.panel import MarketPanel
from automoonbot.backtest.walkforward import Fold, walk_forward

__all__ = [
    "Fold",
    "Labels",
    "Market",
    "MarketPanel",
    "Simulator",
    "Summary",
    "forward_returns",
    "summary",
    "walk_forward",
]

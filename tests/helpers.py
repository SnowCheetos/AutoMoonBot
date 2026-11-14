"""Shared synthetic market data for tests."""

import numpy as np
from automoonbot.backtest import MarketPanel

HOUR = 3600 * 10**9


def synthetic(t: int, s: int, seed: int = 0, halts: bool = False) -> MarketPanel:
    rng = np.random.default_rng(seed)
    drift = rng.normal(0, 0.0005, s)
    c = 50 * np.exp(np.cumsum(rng.normal(drift, 0.01, (t, s)), axis=0))
    o = np.vstack([c[:1], c[:-1]]) * np.exp(rng.normal(0, 0.002, (t, s)))
    h = np.maximum(o, c) * (1 + rng.uniform(0, 0.004, (t, s)))
    lo = np.minimum(o, c) * (1 - rng.uniform(0, 0.004, (t, s)))
    split = np.ones((t, s))
    split[t // 2, 0] = 2.0  # 2:1 split: post-split prices halve
    c[t // 2 :, 0] /= 2
    o[t // 2 :, 0] /= 2
    h[t // 2 :, 0] /= 2
    lo[t // 2 :, 0] /= 2
    if halts:
        for arr in (o, c, h, lo):
            arr[t // 3 : t // 3 + 3, -1] = np.nan
    times = np.datetime64("2024-01-02T15:00", "ns").astype(np.int64) + np.arange(t) * HOUR
    return MarketPanel.from_arrays(
        [f"s{i}" for i in range(s)],
        times,
        o,
        c,
        np.full((t, s), 1e8),
        split=split,
        high=h,
        low=lo,
    )

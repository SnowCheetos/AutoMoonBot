"""Option-writing baselines: covered calls and cash-secured put writes.

The classic benchmarks for selling volatility (in the spirit of the Cboe
BXM and PUT indices, not replicas of them). Each writes one listed contract
on a fixed fraction of equity, holds it to expiration, and writes the next
one at the first decision after it settles:

* ``right="call"``: hold the underlying at weight 1 and write calls on
  ``coverage`` of it. An assigned call takes the shares away; they are
  bought back with the next write.
* ``right="put"``: hold cash and write puts on ``coverage`` of equity. An
  assigned put delivers shares; they are sold with the next write.

Weights of options are notional (see :mod:`automoonbot.backtest`), so a
call weight of ``-coverage`` covers exactly that share of the stock, sized
from the same opening price.

Contract choice uses only what is known at the decision: contracts with a
bar at ``t`` (which the point-in-time chain only has once they are near the
money), their listed terms, and the underlying's close. Among those with
``min_bars`` to ``max_bars`` bars left, the nearest expiration wins, then
the strike nearest to ``close * (1 + otm)`` for calls (``1 - otm`` for
puts), then the further out-of-the-money strike.

Known gap: the simulator exercises American calls early before an
ex-dividend date when that is rational, which the strategy cannot see; the
book then sits in cash until the contract's expiration.
"""

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from automoonbot.backtest.panel import MarketPanel

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]


@dataclass
class OptionWrite:
    right: str = "call"
    #: Strike target as a fraction out of the money (0: at the money).
    otm: float = 0.0
    #: Notional written per unit of equity.
    coverage: float = 1.0
    min_bars: int = 15
    max_bars: int = 50
    #: Security id of the underlying; default: the only one the panel's
    #: options are written on.
    underlying: str | None = None
    name: str = field(default="")

    def __post_init__(self) -> None:
        if self.right not in ("call", "put"):
            raise ValueError("right must be 'call' or 'put'")
        if not (0 <= self.otm < 1 and 0 < self.coverage <= 1):
            raise ValueError("need 0 <= otm < 1 and 0 < coverage <= 1")
        if not 0 <= self.min_bars <= self.max_bars:
            raise ValueError("need 0 <= min_bars <= max_bars")
        if not self.name:
            kind = "covered_call" if self.right == "call" else "put_write"
            self.name = f"{kind}_otm{self.otm:g}"

    def fit(self, panel: MarketPanel, train: I64) -> None:
        """Nothing to fit."""

    def _chain(self, panel: MarketPanel) -> tuple[int, I64, I64, F64]:
        inst = panel.instruments
        if inst is None:
            raise ValueError(f"{self.name}: the panel has no instruments (use with_contracts)")
        opts = [(j, x) for j, x in enumerate(inst) if x.get("kind") == "option"]
        roots = sorted({int(x["underlying"]) for _, x in opts})
        if self.underlying is not None:
            u = panel.securities.index(self.underlying)
        elif len(roots) == 1:
            u = roots[0]
        else:
            raise ValueError(f"{self.name}: options on {len(roots)} underlyings; name one")
        mine = [(j, x) for j, x in opts if x["underlying"] == u and x["right"] == self.right]
        if not mine:
            raise ValueError(f"{self.name}: no {self.right}s on {panel.securities[u]}")
        cols = np.array([j for j, _ in mine], dtype=np.int64)
        expiry = np.array([int(x["expiry"]) for _, x in mine], dtype=np.int64)
        strike = np.array([float(x["strike"]) for _, x in mine])
        return u, cols, expiry, strike

    def _pick(
        self, panel: MarketPanel, t: int, u: int, cols: I64, expiry: I64, strike: F64
    ) -> int | None:
        spot = panel.close[t, u]
        if not (panel.tradable[t, u] and np.isfinite(spot) and spot > 0):
            return None
        left = expiry - t
        ok = panel.tradable[t, cols] & (left >= self.min_bars) & (left <= self.max_bars)
        if not ok.any():
            return None
        idx = np.flatnonzero(ok)
        idx = idx[expiry[idx] == expiry[idx].min()]
        sign = 1.0 if self.right == "call" else -1.0
        target = spot * (1.0 + sign * self.otm)
        # Nearest strike; ties go further out of the money.
        order = np.lexsort((-sign * strike[idx], np.abs(strike[idx] - target)))
        return int(idx[order[0]])

    def targets(self, panel: MarketPanel, steps: I64) -> F64:
        u, cols, expiry, strike = self._chain(panel)
        out = np.full((steps.size, panel.shape[1]), np.nan)
        stock = 1.0 if self.right == "call" else 0.0
        held_until = -1  # expiry bar of the contract written, -1: none
        waiting_row_sent = False
        for k, t in enumerate(steps.tolist()):
            if t < held_until:
                continue  # hold the contract (and the stock) to expiration
            pick = self._pick(panel, t, u, cols, expiry, strike)
            row = out[k]
            if pick is None:
                # Nothing to write yet: hold the base position, once.
                if not waiting_row_sent:
                    row[u] = stock
                    row[cols] = 0.0
                    waiting_row_sent = True
                held_until = -1
                continue
            row[u] = stock
            row[cols] = 0.0  # anything left over, e.g. from an earlier fold
            row[cols[pick]] = -self.coverage
            held_until = int(expiry[pick])
            waiting_row_sent = False
        return out

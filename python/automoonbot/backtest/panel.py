"""Market panels: aligned ``[T, S]`` arrays for the execution simulator.

The panel is the simulated *world*, so it uses the latest corrected bars and
actual corporate actions. What a strategy may *know* at each step is a
separate question answered by the temporal graph at
:attr:`MarketPanel.decision_time`.

Every quantity the simulator uses to size or price the trade executed at the
open of bar ``t`` (``adv_shares[t]``, ``sigma[t]``) is computed from bars
``<= t - 1`` only.
"""

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, fields, replace
from typing import Any

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc

from automoonbot import _moonrs
from automoonbot.data.store import ParquetStore

F64 = npt.NDArray[np.float64]


def _trailing(x: F64, window: int, fn: str) -> F64:
    """Trailing mean or sample std over the previous ``window`` rows,
    *excluding* the current row and ignoring NaN; NaN until
    ``max(2, window // 2)`` observations exist. Vectorized via prefix sums
    (O(T * S))."""
    if window < 2:
        raise ValueError("window must be >= 2")
    t, s = x.shape
    ok = ~np.isnan(x)
    v = np.where(ok, x, 0.0)
    zeros = np.zeros((1, s))
    cs = np.vstack([zeros, np.cumsum(v, axis=0)])
    cq = np.vstack([zeros, np.cumsum(v * v, axis=0)])
    cn = np.vstack([zeros, np.cumsum(ok, axis=0, dtype=np.float64)])
    # Row k uses rows [k - window, k): prefix indices k and max(0, k - window).
    hi = np.arange(t)
    lo = np.maximum(0, hi - window)
    n = cn[hi] - cn[lo]
    total = cs[hi] - cs[lo]
    out = np.full((t, s), np.nan)
    good = n >= max(2, window // 2)
    if fn == "mean":
        out[good] = total[good] / n[good]
    elif fn == "std":
        sq = cq[hi] - cq[lo]
        var = (sq - total * total / np.where(n > 0, n, 1)) / np.where(n > 1, n - 1, 1)
        out[good] = np.sqrt(np.maximum(var[good], 0.0))
    else:
        raise ValueError(fn)
    return out


@dataclass
class MarketPanel:
    securities: list[str]
    #: ``[T]`` bar end times (UTC ns).
    times: npt.NDArray[np.int64]
    #: ``[T]`` earliest time at which every bar ending at ``times[t]`` was
    #: known: the moment a decision for step ``t`` can be made.
    decision_time: npt.NDArray[np.int64]
    open: F64
    #: Last valid close carried forward over halts; NaN when not listed.
    close: F64
    volume: F64
    tradable: npt.NDArray[np.bool_]
    #: Between a security's first and last bar (inclusive).
    listed: npt.NDArray[np.bool_]
    split: F64
    dividend: F64
    adv_shares: F64
    sigma: F64
    #: Bar high/low (NaN where not tradable); ``max/min(open, close)`` when
    #: the source has none.
    high: F64
    low: F64
    #: ``[T, S]`` point-in-time universe membership (e.g. index constituents
    #: as known at each decision), or ``None`` for "every security". Prices
    #: of non-members stay available (history for features); strategies only
    #: invest in :attr:`investable` securities.
    member: npt.NDArray[np.bool_] | None = None

    @property
    def investable(self) -> npt.NDArray[np.bool_]:
        """``[T, S]`` a bar exists at ``t`` and the security is in the
        universe at ``t``."""
        if self.member is None:
            return self.tradable
        out: npt.NDArray[np.bool_] = self.tradable & self.member
        return out

    def with_membership(self, member: npt.ArrayLike) -> MarketPanel:
        """A copy restricted to the universe ``member[T, S]``."""
        m = np.asarray(member, dtype=np.bool_)
        if m.shape != self.shape:
            raise ValueError(f"membership must be {self.shape}, got {m.shape}")
        return replace(self, member=m)

    @property
    def shape(self) -> tuple[int, int]:
        t, s = self.open.shape
        return int(t), int(s)

    @classmethod
    def from_arrays(
        cls,
        securities: Sequence[str],
        times: npt.ArrayLike,
        open: npt.ArrayLike,
        close: npt.ArrayLike,
        volume: npt.ArrayLike,
        *,
        decision_time: npt.ArrayLike | None = None,
        split: npt.ArrayLike | None = None,
        dividend: npt.ArrayLike | None = None,
        high: npt.ArrayLike | None = None,
        low: npt.ArrayLike | None = None,
        window: int = 20,
    ) -> MarketPanel:
        """Build from dense arrays; NaN prices mark untradable bars."""
        o, c, v = (np.asarray(a, dtype=np.float64) for a in (open, close, volume))
        with np.errstate(invalid="ignore"):
            hi = np.fmax(o, c) if high is None else np.asarray(high, dtype=np.float64)
            lo = np.fmin(o, c) if low is None else np.asarray(low, dtype=np.float64)
        if hi.shape != o.shape or lo.shape != o.shape:
            raise ValueError("high/low must match open")
        t = np.asarray(times, dtype=np.int64)
        if o.shape != c.shape or o.shape != v.shape or o.shape[0] != t.size:
            raise ValueError("open/close/volume must be [T, S] matching times")
        if np.any(np.diff(t) <= 0):
            raise ValueError("times must be strictly increasing")
        tradable = np.isfinite(o) & np.isfinite(c) & (o > 0) & (c > 0) & np.isfinite(v)
        sp = np.ones_like(o) if split is None else np.asarray(split, dtype=np.float64)
        dv = np.zeros_like(o) if dividend is None else np.asarray(dividend, dtype=np.float64)
        # Close-to-close log returns, split-adjusted at the boundary, with
        # dividends added back (total return), for trailing volatility.
        with np.errstate(invalid="ignore", divide="ignore"):
            prev = np.vstack([np.full((1, o.shape[1]), np.nan), c[:-1]])
            r = np.log((c * sp + dv) / prev)
        r[~tradable] = np.nan
        vol_shares = np.where(tradable, v, np.nan)
        adv = _trailing(vol_shares, window, "mean")
        sig = _trailing(r, window, "std")
        dt_arr = t if decision_time is None else np.asarray(decision_time, dtype=np.int64)
        rows = np.arange(t.size)[:, None]
        has = np.where(tradable, rows, -1)
        first = np.where(tradable.any(axis=0), np.argmax(tradable, axis=0), t.size)
        last = np.max(has, axis=0)
        listed = (rows >= first[None, :]) & (rows <= last[None, :])
        # Forward-fill closes over halts (index of the last tradable row).
        last_idx = np.maximum.accumulate(has, axis=0)
        filled = np.take_along_axis(np.where(tradable, c, np.nan), np.maximum(last_idx, 0), axis=0)
        close_ff = np.where(listed, filled, np.nan)
        return cls(
            securities=list(securities),
            times=t,
            decision_time=dt_arr,
            open=np.where(tradable, o, np.nan),
            close=close_ff,
            volume=np.where(tradable, v, 0.0),
            tradable=tradable,
            listed=listed,
            split=sp,
            dividend=dv,
            adv_shares=np.nan_to_num(adv, nan=0.0),
            sigma=np.nan_to_num(sig, nan=0.0),
            high=np.where(tradable, hi, np.nan),
            low=np.where(tradable, lo, np.nan),
        )

    @classmethod
    def from_store(
        cls,
        store: ParquetStore,
        securities: Sequence[str],
        *,
        start: int | None = None,
        end: int | None = None,
        as_of: int | None = None,
        window: int = 20,
    ) -> MarketPanel:
        """Assemble from the store's unadjusted ``bars`` and
        ``corporate_actions`` (latest versions known by ``as_of``)."""
        secs = list(dict.fromkeys(securities))
        flt = pc.field("security_id").isin(secs)
        bars = store.read("bars", as_of, filter=flt, latest=True)
        if start is not None:
            bars = bars.filter(
                pc.field("end") >= pa.scalar(start).cast(bars.schema.field("end").type)
            )
        if end is not None:
            bars = bars.filter(
                pc.field("end") <= pa.scalar(end).cast(bars.schema.field("end").type)
            )
        if bars.num_rows == 0:
            raise ValueError("no bars for the requested securities and range")
        ends = bars.column("end").cast(pa.int64()).to_numpy()
        times = np.unique(ends)
        ti = np.searchsorted(times, ends)
        col = {s: i for i, s in enumerate(secs)}
        si = np.array([col[s] for s in bars.column("security_id").to_pylist()])
        shape = (times.size, len(secs))

        def dense(name: str) -> F64:
            a = np.full(shape, np.nan)
            a[ti, si] = bars.column(name).to_numpy()
            return a

        known = bars.column("known_time").cast(pa.int64()).to_numpy()
        decision = np.full(times.size, np.iinfo(np.int64).min)
        np.maximum.at(decision, ti, known)

        split = np.ones(shape)
        dividend = np.zeros(shape)
        actions = store.read("corporate_actions", as_of, filter=flt, latest=True)
        if actions.num_rows:
            epoch = dt.date(1970, 1, 1)
            # The ex-date's session is the first bar ending on/after the ex-date.
            ex_ns = np.array(
                [(d - epoch).days * 86_400 * 10**9 for d in actions.column("ex_date").to_pylist()],
                dtype=np.int64,
            )
            idx = np.searchsorted(times, ex_ns, side="left")
            for k, (sec, act, val) in enumerate(
                zip(
                    actions.column("security_id").to_pylist(),
                    actions.column("action").to_pylist(),
                    actions.column("value").to_pylist(),
                    strict=True,
                )
            ):
                if idx[k] >= times.size:
                    continue
                if act == "split":
                    split[idx[k], col[sec]] *= val
                elif act == "cash_dividend":
                    dividend[idx[k], col[sec]] += val
                else:
                    raise ValueError(f"unknown corporate action {act!r}")
        return cls.from_arrays(
            secs,
            times,
            dense("open"),
            dense("close"),
            dense("volume"),
            decision_time=decision,
            split=split,
            dividend=dividend,
            high=dense("high"),
            low=dense("low"),
            window=window,
        )

    def truncate(self, n: int) -> MarketPanel:
        """The first ``n`` bars: everything a strategy may know when deciding
        at step ``n - 1``. (``listed`` is the simulator's world view and may
        differ from a recomputation on the truncated data; strategies use
        ``tradable`` instead.)"""
        if not 1 <= n <= self.shape[0]:
            raise ValueError(f"truncate length {n} outside [1, {self.shape[0]}]")
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = v[:n] if isinstance(v, np.ndarray) else v
        return MarketPanel(**out)

    def log_returns(self) -> F64:
        """``[T, S]`` close-to-close total log returns (splits and dividends
        included); NaN where bar ``t`` or a previous close is missing. Row
        ``t`` uses bars ``<= t`` only."""
        prev = np.vstack([np.full((1, self.shape[1]), np.nan), self.close[:-1]])
        with np.errstate(invalid="ignore", divide="ignore"):
            r: F64 = np.log((self.close * self.split + self.dividend) / prev)
        r[~self.tradable] = np.nan
        return r

    def total_return_index(self) -> F64:
        """``[T, S]`` causal total-return price index, equal to ``close`` at
        each security's first tradable bar and carried forward over halts.
        Ratios of it equal ratios of prices back-adjusted *as of* any later
        bar, so scale-free features computed on it never see future
        corporate actions."""
        r = self.log_returns()
        first = np.argmax(self.tradable, axis=0)
        cols = np.arange(self.shape[1])
        base = self.close[first, cols]
        cum = np.cumsum(np.nan_to_num(r, nan=0.0), axis=0)
        cum = cum - cum[first, cols]
        rows = np.arange(self.shape[0])[:, None]
        started = rows >= first[None, :]
        return np.where(started & self.tradable.any(axis=0), base * np.exp(cum), np.nan)

    def market(self) -> _moonrs.Market:
        """The Rust market object (shared by any number of simulators)."""
        c = np.ascontiguousarray
        return _moonrs.Market(
            c(np.nan_to_num(self.open, nan=0.0)),
            c(np.nan_to_num(self.close, nan=0.0)),
            c(self.adv_shares),
            c(self.sigma),
            c(self.tradable),
            c(self.listed),
            c(self.split),
            c(self.dividend),
        )

"""Synthetic option chains: listed-style contracts priced with Black-Scholes
at the volatility the market was charging, for history no option data
covers.

Pricing at the underlying's own trailing volatility would erase the
volatility risk premium (implied volatility has mostly been above the
volatility that followed), which is most of what option sellers earn, and
would price out-of-the-money puts as cheap as calls. So the volatility here
comes from a market-implied index (VIX for the S&P 500, quoted daily since
1990) with a skew on top:

``sigma(K) = atm * clip(1 - slope * z + curvature * z**2, floor, cap)``,
``atm = atm_ratio * index / 100``, ``z = ln(K / F) / (atm * sqrt(tau))``.

``z`` is the log-moneyness in standard deviations, so the same shape holds
across maturities and volatility levels. The defaults are typical of
one-month S&P 500 smiles (VIX above at-the-money volatility because it
integrates the put wing; puts 25 delta out of the money about 30% richer
than at the money, calls about 13% cheaper). They are starting points to
calibrate against real chains (Massive), not measurements.

Point in time
-------------
* A cycle's contracts are listed at the close of the previous monthly
  expiration (the first cycle: ``first_life`` sessions before its
  expiration), with strikes from that close, rounded to ``strike_step``.
* A close at row ``t`` uses the underlying's close, the index close, and
  the rate and dividend yield known at that close; an open uses the
  underlying's and the index's opens with the previous close's rate and
  yield. Nothing after the price's own time enters.
* Option bars close at 16:15 New York and the index settles then too, so
  each row's time is at least 16:15 and its decision time at least 16:15
  plus ``delay``.

Prices are the generalized Black-Scholes value (carry ``r - q``) floored at
intrinsic (American) and at one cent (the minimum tick), computed by the
same code the simulator uses for portfolio margin. Synthetic contracts are
fully liquid; costs come from the simulator's option spread model.
"""

import datetime as dt
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import pyarrow.compute as pc

from automoonbot import _moonrs
from automoonbot.backtest.panel import MarketPanel, session_dates
from automoonbot.data.providers.massive import monthly_candidates
from automoonbot.data.store import ParquetStore

F64 = npt.NDArray[np.float64]
I64 = npt.NDArray[np.int64]
_NY = ZoneInfo("America/New_York")
_SEC = 1_000_000_000
TICK = 0.01


@dataclass(frozen=True)
class Smile:
    atm_ratio: float = 0.9
    slope: float = 0.32
    curvature: float = 0.19
    floor: float = 0.5
    cap: float = 3.0

    def __post_init__(self) -> None:
        if not (self.atm_ratio > 0 and 0 < self.floor <= 1 <= self.cap):
            raise ValueError("need atm_ratio > 0 and 0 < floor <= 1 <= cap")
        if not (np.isfinite(self.slope) and np.isfinite(self.curvature) and self.curvature >= 0):
            raise ValueError("slope must be finite and curvature non-negative")

    def vol(self, index: F64, log_moneyness: F64, tau: F64) -> F64:
        """Volatility for an index level (in percent), ``ln(K / F)`` and
        years to expiry, elementwise."""
        atm = self.atm_ratio * np.asarray(index, dtype=np.float64) / 100.0
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(tau > 0, log_moneyness / (atm * np.sqrt(np.maximum(tau, 0.0))), 0.0)
        shape = np.clip(1.0 - self.slope * z + self.curvature * z * z, self.floor, self.cap)
        out: F64 = atm * shape
        return out


def tbill_rate(discount_pct: F64, days: int = 91) -> F64:
    """Continuously compounded rate from a T-bill discount yield in percent
    (e.g. Yahoo's ``^IRX``, 13 weeks)."""
    d = np.asarray(discount_pct, dtype=np.float64) / 100.0
    price = 1.0 - d * days / 360.0
    with np.errstate(divide="ignore", invalid="ignore"):
        out: F64 = -np.log(price) / (days / 365.0)
    return out


def trailing_dividend_yield(panel: MarketPanel, column: int, window: int = 252) -> F64:
    """Dividends with ex-dates in the last ``window`` bars over the close:
    known at each close."""
    d = np.nan_to_num(panel.dividend[:, column])
    cs = np.concatenate([[0.0], np.cumsum(d)])
    hi = np.arange(1, d.size + 1)
    total = cs[hi] - cs[np.maximum(0, hi - window)]
    with np.errstate(divide="ignore", invalid="ignore"):
        out: F64 = np.where(panel.close[:, column] > 0, total / panel.close[:, column], 0.0)
    return np.nan_to_num(out)


def aligned_bars(
    store: ParquetStore, security: str, panel: MarketPanel, *, as_of: int | None = None
) -> tuple[F64, F64]:
    """``security``'s open and close on each of ``panel``'s sessions,
    carrying the last close forward over sessions it has no bar for (never
    backward). NaN before its first bar."""
    bars = store.read("bars", as_of, filter=pc.field("security_id") == security, latest=True)
    if bars.num_rows == 0:
        raise ValueError(f"no bars for {security}")
    days = session_dates(bars.column("end").cast("int64").to_numpy())
    order = np.argsort(days, kind="stable")
    days = days[order]
    if np.any(days[1:] == days[:-1]):
        raise ValueError(f"{security}: two bars in one session")
    o = bars.column("open").to_numpy()[order]
    c = bars.column("close").to_numpy()[order]
    want = session_dates(panel.times)
    k = np.searchsorted(days, want, side="right") - 1
    exact = (k >= 0) & (days[np.maximum(k, 0)] == want)
    close = np.where(k >= 0, c[np.maximum(k, 0)], np.nan)
    # An open is only what that session opened at; else the last close.
    opn = np.where(exact, o[np.maximum(k, 0)], close)
    opn = np.where(np.isfinite(opn) & (opn > 0), opn, close)
    return opn.astype(np.float64), close.astype(np.float64)


@dataclass(frozen=True)
class Contract:
    security_id: str
    right: str
    strike: float
    listed: int
    expiry: int


@dataclass(frozen=True)
class SyntheticChain:
    """Which contracts to list and how to price them.

    * ``levels``: strikes as fractions out of the money at listing (calls
      above the close, puts below; 0 is at the money).
    * ``rights``: ``("call",)``, ``("put",)`` or both.
    """

    levels: tuple[float, ...] = (0.0,)
    rights: tuple[str, ...] = ("call", "put")
    strike_step: float = 1.0
    first_life: int = 21
    smile: Smile = field(default_factory=Smile)
    multiplier: float = 100.0
    periods_per_year: float = 252.0
    volume: float = 1e9
    option_close: dt.time = dt.time(16, 15)
    delay: dt.timedelta = dt.timedelta(minutes=15)

    def __post_init__(self) -> None:
        if not self.levels or any(not 0 <= x < 1 for x in self.levels):
            raise ValueError("levels must be in [0, 1)")
        if not self.rights or set(self.rights) - {"call", "put"}:
            raise ValueError("rights must be a non-empty subset of ('call', 'put')")
        if not (self.strike_step > 0 and self.first_life >= 1 and self.periods_per_year > 0):
            raise ValueError("need strike_step > 0, first_life >= 1, periods_per_year > 0")

    def expiries(self, panel: MarketPanel, underlying: int) -> I64:
        """Rows of the standard monthly expirations (third Friday, or the
        Thursday before it when the Friday is not a session) on which the
        underlying trades."""
        days = session_dates(panel.times)
        row = {d.astype(dt.date): i for i, d in enumerate(days)}
        first, last = days[0].astype(dt.date), days[-1].astype(dt.date)
        out: list[int] = []
        y, m = first.year, first.month
        while (y, m) <= (last.year, last.month):
            for d in monthly_candidates(y, m):
                i = row.get(d)
                if i is not None:
                    if panel.tradable[i, underlying]:
                        out.append(i)
                    break
            y, m = (y + 1, 1) if m == 12 else (y, m + 1)
        return np.array(out, dtype=np.int64)

    def contracts(self, panel: MarketPanel, underlying: int) -> list[Contract]:
        root = panel.securities[underlying].split(":", 1)[-1]
        days = session_dates(panel.times)
        exp = self.expiries(panel, underlying)
        out: list[Contract] = []
        seen: set[tuple[str, float, int]] = set()
        for n, e in enumerate(exp.tolist()):
            listed = int(exp[n - 1]) if n else e - self.first_life
            if listed < 0 or listed >= e or not panel.tradable[listed, underlying]:
                continue
            spot = float(panel.close[listed, underlying])
            for right in self.rights:
                sign = 1.0 if right == "call" else -1.0
                for level in self.levels:
                    k = round(spot * (1.0 + sign * level) / self.strike_step) * self.strike_step
                    k = round(k, 6)
                    if k <= 0 or (right, k, e) in seen:
                        continue
                    seen.add((right, k, e))
                    ymd = days[e].astype(dt.date).strftime("%y%m%d")
                    sid = f"synthetic:{root}{ymd}{right[0].upper()}{round(k * 1000):08d}"
                    out.append(Contract(sid, right, k, listed, e))
        return out

    def build(
        self,
        panel: MarketPanel,
        underlying: str,
        index_open: F64,
        index_close: F64,
        rate: F64,
        dividend_yield: F64,
    ) -> MarketPanel:
        """``panel`` with the chain's contracts appended as option columns.

        ``index_*`` are the implied-volatility index (percent) at each bar's
        open and close; ``rate`` and ``dividend_yield`` are continuously
        compounded, known at each close. All are ``[T]`` arrays aligned to
        the panel's rows.
        """
        t_bars = panel.shape[0]
        u = panel.securities.index(underlying)
        arrays = [np.asarray(a, dtype=np.float64) for a in (index_open, index_close, rate)]
        q = np.asarray(dividend_yield, dtype=np.float64)
        if any(a.shape != (t_bars,) for a in [*arrays, q]):
            raise ValueError(f"inputs must be [{t_bars}] arrays aligned to the panel")
        vix_o, vix_c, r = arrays
        chain = self.contracts(panel, u)
        if not chain:
            raise ValueError("no monthly expirations with a listing session in the panel")
        n = len(chain)
        opn = np.full((t_bars, n), np.nan)
        cls = np.full((t_bars, n), np.nan)
        # One vectorized call per price field over every contract-row.
        rows = np.concatenate([np.arange(c.listed, c.expiry + 1) for c in chain])
        col = np.concatenate(
            [np.full(c.expiry - c.listed + 1, j) for j, c in enumerate(chain)]
        ).astype(np.int64)
        strike = np.array([c.strike for c in chain])[col]
        call = np.array([c.right == "call" for c in chain])[col]
        left = np.array([c.expiry for c in chain])[col] - rows
        # An open is priced with the last close's rate and dividend yield.
        prev = np.maximum(rows - 1, 0)
        for spot_px, index, tau_bars, at, out in (
            (panel.close[:, u], vix_c, left, rows, cls),
            (panel.open[:, u], vix_o, left + 1, prev, opn),
        ):
            spot = spot_px[rows]
            tau = tau_bars / self.periods_per_year
            carry = r[at] - q[at]
            with np.errstate(divide="ignore", invalid="ignore"):
                lm = np.log(strike / spot) - carry * tau
            vol = self.smile.vol(index[rows], lm, tau)
            ok = np.isfinite(spot) & (spot > 0) & np.isfinite(vol) & (vol > 0) & np.isfinite(carry)
            px = np.full(rows.size, np.nan)
            px[ok] = _moonrs.option_value(
                np.ascontiguousarray(call[ok]),
                np.ascontiguousarray(spot[ok]),
                np.ascontiguousarray(strike[ok]),
                np.ascontiguousarray(tau[ok]),
                np.ascontiguousarray(r[at][ok]),
                np.ascontiguousarray(carry[ok]),
                np.ascontiguousarray(vol[ok]),
                american=True,
            )
            out[rows, col] = np.maximum(px, TICK)
        return self._assemble(panel, u, chain, opn, cls)

    def _assemble(
        self, panel: MarketPanel, u: int, chain: Sequence[Contract], opn: F64, cls: F64
    ) -> MarketPanel:
        t_bars, s = panel.shape
        n = len(chain)
        live = np.isfinite(cls)
        vol = np.where(live, self.volume, 0.0)
        times, decision = self._session_times(panel)
        p = MarketPanel.from_arrays(
            [*panel.securities, *(c.security_id for c in chain)],
            times,
            np.hstack([panel.open, opn]),
            np.hstack([panel.close, cls]),
            np.hstack([panel.volume, vol]),
            decision_time=decision,
            split=np.hstack([panel.split, np.ones((t_bars, n))]),
            dividend=np.hstack([panel.dividend, np.zeros((t_bars, n))]),
            high=np.hstack([panel.high, np.fmax(opn, cls)]),
            low=np.hstack([panel.low, np.fmin(opn, cls)]),
        )
        # Keep the original columns' lagged inputs; synthetic contracts are
        # fully liquid from their listing.
        adv = np.hstack([panel.adv_shares, np.where(live, self.volume, 0.0)])
        sigma = np.hstack([panel.sigma, np.zeros((t_bars, n))])
        member = None if panel.member is None else np.hstack([panel.member, live])
        p = replace(
            p,
            adv_shares=adv,
            sigma=sigma,
            member=member,
            listed=np.hstack([panel.listed, p.listed[:, s:]]),
            tradable=np.hstack([panel.tradable, p.tradable[:, s:]]),
            open=np.hstack([panel.open, p.open[:, s:]]),
            close=np.hstack([panel.close, p.close[:, s:]]),
        )
        base: list[dict[str, Any]] = (
            [dict(x) for x in panel.instruments]
            if panel.instruments
            else [{"kind": "equity"} for _ in range(s)]
        )
        opts = [
            {
                "kind": "option",
                "underlying": u,
                "right": c.right,
                "strike": c.strike,
                "expiry": c.expiry,
                "style": "american",
                "settlement": "physical",
                "deliverable": self.multiplier,
                "multiplier": self.multiplier,
            }
            for c in chain
        ]
        out = p.with_instruments([*base, *opts])
        if panel.half_spread is not None:
            out = out.with_half_spread(np.hstack([panel.half_spread, np.full((t_bars, n), np.nan)]))
        if panel.rate is not None:
            out = out.with_rate(panel.rate)
        return out

    def _session_times(self, panel: MarketPanel) -> tuple[I64, I64]:
        days = session_dates(panel.times)
        close_ns = np.array(
            [
                int(dt.datetime.combine(d.astype(dt.date), self.option_close, _NY).timestamp())
                * _SEC
                for d in days
            ],
            dtype=np.int64,
        )
        times = np.maximum(panel.times, close_ns)
        delay = int(self.delay.total_seconds() * _SEC)
        decision = np.maximum(panel.decision_time, close_ns + delay)
        if np.any(np.diff(times) <= 0):
            raise ValueError("the panel must have one row per session")
        return times, decision

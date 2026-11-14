"""Point-in-time index membership from dated constituent snapshots.

A snapshot list is a sequence of ``(effective date, symbols)`` pairs: the
constituents in force from that date until the next snapshot. Membership on
a bar is the latest snapshot whose date is on or before the bar's trading
date (in the exchange's time zone), so a change becomes visible on its
effective date and never earlier. Index changes are announced days before
they take effect, so this only ever lags what was knowable.

Symbols are those in use *at the time*; mapping them to the identifiers of
a price source (which typically keys on today's symbol) is the caller's
job, and every symbol that cannot be mapped is a hole in the universe that
:func:`coverage` reports.
"""

import csv
import datetime as dt
import os
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

_NY = ZoneInfo("America/New_York")


def _trading_days(times_ns: npt.NDArray[np.int64], tz: ZoneInfo) -> list[dt.date]:
    return [dt.datetime.fromtimestamp(int(t) // 10**9, tz).date() for t in times_ns]


@dataclass(frozen=True)
class Snapshots:
    dates: list[dt.date]
    members: list[frozenset[str]]

    def __post_init__(self) -> None:
        if len(self.dates) != len(self.members) or not self.dates:
            raise ValueError("need one member set per date, and at least one")
        if any(b <= a for a, b in zip(self.dates, self.dates[1:], strict=False)):
            raise ValueError("snapshot dates must be strictly increasing")

    @staticmethod
    def from_csv(path: str | os.PathLike[str]) -> Snapshots:
        """Read ``date,tickers`` rows (ISO date, comma-separated symbols)."""
        with open(path, newline="") as f:
            rows = [(dt.date.fromisoformat(r["date"]), r["tickers"]) for r in csv.DictReader(f)]
        rows.sort()
        return Snapshots(
            [d for d, _ in rows],
            [frozenset(s.strip() for s in t.split(",") if s.strip()) for _, t in rows],
        )

    def extended(self, date: dt.date, members: Iterable[str]) -> Snapshots:
        """With a later snapshot appended (e.g. today's list)."""
        if date <= self.dates[-1]:
            raise ValueError("extension must be after the last snapshot")
        return Snapshots([*self.dates, date], [*self.members, frozenset(members)])

    def symbols(self, since: dt.date | None = None) -> set[str]:
        """Every symbol that is a member on or after ``since``."""
        start = 0
        if since is not None:
            start = max(0, int(self._index([since])[0]))
        return set().union(*self.members[start:])

    def _index(self, days: Sequence[dt.date]) -> npt.NDArray[np.int64]:
        """Index of the snapshot in force on each day (-1: before the first)."""
        ords = np.array([d.toordinal() for d in self.dates], dtype=np.int64)
        q = np.array([d.toordinal() for d in days], dtype=np.int64)
        out: npt.NDArray[np.int64] = np.searchsorted(ords, q, side="right") - 1
        return out

    def mask(
        self,
        bar_times_ns: npt.ArrayLike,
        securities: Sequence[str],
        symbol_of: Callable[[str], str | None],
        tz: ZoneInfo = _NY,
    ) -> npt.NDArray[np.bool_]:
        """``[T, S]`` membership of ``securities`` (mapped to the snapshot's
        symbols by ``symbol_of``; ``None``: never a member) on the trading
        dates of ``bar_times_ns``. Bars before the first snapshot are not
        members."""
        times = np.asarray(bar_times_ns, dtype=np.int64)
        idx = self._index(_trading_days(times, tz))
        syms = [symbol_of(s) for s in securities]
        # One boolean row per snapshot, then gather by bar.
        table = np.zeros((len(self.dates), len(securities)), dtype=np.bool_)
        for k, m in enumerate(self.members):
            table[k] = [s is not None and s in m for s in syms]
        out = np.zeros((times.size, len(securities)), dtype=np.bool_)
        ok = idx >= 0
        out[ok] = table[idx[ok]]
        return out


def coverage(
    snapshots: Snapshots,
    bar_times_ns: npt.ArrayLike,
    member: npt.NDArray[np.bool_],
    tradable: npt.NDArray[np.bool_],
    tz: ZoneInfo = _NY,
) -> dict[int, float]:
    """Per calendar year: the fraction of index member-days (constituents
    times bars) for which the panel has a tradable member bar."""
    times = np.asarray(bar_times_ns, dtype=np.int64)
    days = _trading_days(times, tz)
    idx = snapshots._index(days)
    size = np.array([len(m) for m in snapshots.members])
    total = np.where(idx >= 0, size[np.maximum(idx, 0)], 0)
    have = (member & tradable).sum(axis=1)
    years = np.array([d.year for d in days])
    out = {}
    for y in np.unique(years):
        m = years == y
        if total[m].sum():
            out[int(y)] = float(have[m].sum() / total[m].sum())
    return out


def require_history(
    member: npt.NDArray[np.bool_], tradable: npt.NDArray[np.bool_], slack: int = 21
) -> npt.NDArray[np.bool_]:
    """Drop membership runs whose price series starts more than ``slack``
    bars after the run does.

    Price sources key on today's symbols, and symbols get reused: a series
    found under a former member's symbol often belongs to a later company
    (or an ETF) that took the symbol over, and starts long after the
    original company's membership did. Such runs are removed (left as
    holes, which :func:`coverage` counts) rather than filled with another
    company's prices. A spin-off added on its first trading day passes."""
    m = np.asarray(member, dtype=np.bool_)
    t = np.asarray(tradable, dtype=np.bool_)
    if m.shape != t.shape:
        raise ValueError("member and tradable must have the same shape")
    n = m.shape[0]
    first = np.where(t.any(axis=0), np.argmax(t, axis=0), n)
    out = m.copy()
    prev = np.vstack([np.zeros((1, m.shape[1]), np.bool_), m[:-1]])
    starts_r, starts_c = np.nonzero(m & ~prev)
    for r, c in zip(starts_r.tolist(), starts_c.tolist(), strict=True):
        if first[c] > r + slack:
            end = r
            while end < n and m[end, c]:
                end += 1
            out[r:end, c] = False
    return out

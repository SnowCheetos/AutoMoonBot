"""Yahoo Finance chart API provider (OHLCV bars, splits, dividends).

Prototype-grade price history: Yahoo publishes no knowledge times and may
revise history, so the policy is :attr:`KnowledgePolicy.ESTIMATED` and
``known_time`` is the bar's end plus ``delay``. Evaluation code that requires
point-in-time-safe sources can refuse it; it exists so baselines can be
measured on real prices before a licensed feed is wired in.

Normalization
-------------
* Yahoo's OHLC and volume are *split-adjusted* after the fact (dividends are
  not applied to OHLC). Bars are converted back to as-traded prices using the
  split events, matching the ``bars`` schema (unadjusted) and the simulator,
  which applies corporate actions itself.
* Timestamps are bar starts. A bar ends at ``start + interval`` or the end of
  its regular session (e.g. the 15:30 hourly bar ends at 16:00), taken from
  the response's ``tradingPeriods``; daily bars end at 16:00 New York (later
  than an early close, so only ever conservative).
* ``range=max`` silently returns weekly/monthly/quarterly bars, so full
  history is requested as an explicit period, and every response's
  ``dataGranularity`` must equal the requested interval.
* Bars that had not ended when the response was fetched (an in-progress
  session) are dropped, as are bars with missing prices.
* Dividends are reported per split-adjusted share and are converted to per
  as-traded share; ``known_time`` of an action is 00:00 New York on its
  ex-date (announcements precede ex-dates, so this is an upper bound).
"""

import datetime as dt
import os
import re
import time
import urllib.parse
from collections.abc import Iterator, Mapping, Sequence
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
import pyarrow as pa

from automoonbot.data.http import HttpClient
from automoonbot.data.providers.base import KnowledgePolicy, Provider, ProviderInfo
from automoonbot.data.schemas import BARS, CORPORATE_ACTIONS, conform

NAME = "yahoo"
_HOSTS = frozenset({"query1.finance.yahoo.com", "query2.finance.yahoo.com"})
#: Tickers, plus Yahoo index symbols with a leading caret (``^VIX``).
_SYMBOL = re.compile(r"^\^?[A-Z0-9][A-Z0-9.\-=]{0,15}$")
_NY = ZoneInfo("America/New_York")
_SEC = 1_000_000_000
#: interval -> (seconds per bar, default range)
INTERVALS: dict[str, tuple[int, str]] = {
    "1h": (3600, "730d"),
    "1d": (86_400, "max"),
}
#: 1900-01-01 UTC: earlier than any listing, for full-history requests.
_EPOCH_1900 = -2_208_988_800
USER_AGENT = "automoonbot-research/0.2 (+https://github.com/SnowCheetos/AutoMoonBot)"


def security_id(symbol: str) -> str:
    s = symbol.strip().upper()
    if not _SYMBOL.match(s):
        raise ValueError(f"invalid symbol {symbol!r}")
    return f"ticker:{s}"


def _symbol(entity: str) -> str:
    return security_id(entity[7:] if entity.lower().startswith("ticker:") else entity)[7:]


class YahooProvider(Provider):
    info = ProviderInfo(
        name=NAME,
        datasets=frozenset({BARS.name, CORPORATE_ACTIONS.name}),
        policy=KnowledgePolicy.ESTIMATED,
        description="Yahoo Finance chart API bars and corporate actions (no knowledge times)",
    )

    def __init__(
        self,
        interval: str = "1d",
        *,
        range_: str | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        delay: dt.timedelta = dt.timedelta(minutes=1),
        client: HttpClient | None = None,
    ) -> None:
        if interval not in INTERVALS:
            raise ValueError(f"interval must be one of {sorted(INTERVALS)}")
        self.interval = interval
        self.step_s, default_range = INTERVALS[interval]
        self.range = range_ or default_range
        if not re.fullmatch(r"\d{1,5}[dmy]|max|ytd", self.range):
            raise ValueError(f"invalid range {self.range!r}")
        self.delay_ns = int(delay.total_seconds() * _SEC)
        if self.delay_ns < 0:
            raise ValueError("delay must be non-negative")
        self.client = client or HttpClient(
            USER_AGENT, allowed_hosts=_HOSTS, rate_per_host=2.0, cache_dir=cache_dir
        )

    def _span(self) -> str:
        """The requested history. ``range=max`` makes Yahoo silently coarsen
        the bars (to 1wk/1mo/3mo), so the full history is requested as an
        explicit period instead, ending at the next UTC midnight (a stable
        URL, and so a stable cache key, within a day)."""
        if self.range != "max":
            return f"range={self.range}"
        end = (int(time.time()) // 86_400 + 1) * 86_400
        return f"period1={_EPOCH_1900}&period2={end}"

    def chart(self, symbol: str) -> tuple[dict[str, Any], int]:
        """The chart result for ``symbol`` and its fetch time (ns)."""
        sym = _symbol(symbol)
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(sym)}"
            f"?{self._span()}&interval={self.interval}&events=div%2Csplits"
            "&includePrePost=false"
        )
        r = self.client.get(url)
        data = r.json()
        if not isinstance(data, dict) or "chart" not in data:
            raise ValueError(f"{sym}: unexpected response")
        chart = data["chart"]
        if chart.get("error"):
            raise ValueError(f"{sym}: {chart['error']}")
        res = chart.get("result") or []
        if len(res) != 1:
            raise ValueError(f"{sym}: expected one result")
        out: dict[str, Any] = res[0]
        meta = out.get("meta", {})
        got = str(meta.get("symbol", "")).upper()
        if got != sym:
            raise ValueError(f"{sym}: response is for {got!r}")
        # Yahoo may answer with coarser bars than requested.
        gran = meta.get("dataGranularity")
        if gran != self.interval:
            raise ValueError(f"{sym}: got {gran!r} bars, requested {self.interval!r}")
        return out, r.fetched_at_ns

    # --------------------------------------------------------- normalization
    @staticmethod
    def _splits(res: Mapping[str, Any]) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]:
        ev = (res.get("events") or {}).get("splits") or {}
        dates, ratios = [], []
        for v in ev.values():
            num, den = float(v["numerator"]), float(v["denominator"])
            if not (num > 0 and den > 0):
                raise ValueError(f"invalid split {v!r}")
            dates.append(int(v["date"]))
            ratios.append(num / den)
        order = np.argsort(dates, kind="stable")
        return np.asarray(dates, np.int64)[order], np.asarray(ratios, np.float64)[order]

    @staticmethod
    def _later_split_factor(
        at_s: npt.NDArray[np.int64], dates: npt.NDArray[np.int64], ratios: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Product of split ratios effective strictly after each time."""
        if dates.size == 0:
            return np.ones(at_s.size)
        suffix = np.concatenate([np.cumprod(ratios[::-1])[::-1], [1.0]])
        idx = np.searchsorted(dates, at_s, side="right")
        out: npt.NDArray[np.float64] = suffix[idx]
        return out

    def _session_end(
        self, res: Mapping[str, Any], starts: npt.NDArray[np.int64]
    ) -> npt.NDArray[np.int64]:
        step = np.int64(self.step_s)
        if self.interval == "1d":
            ends = []
            for s in starts.tolist():
                d = dt.datetime.fromtimestamp(s, _NY).date()
                ends.append(int(dt.datetime.combine(d, dt.time(16, 0), _NY).timestamp()))
            return np.asarray(ends, np.int64)
        periods = res.get("meta", {}).get("tradingPeriods") or []
        flat = sorted(
            (int(p["start"]), int(p["end"]))
            for day in periods
            for p in (day if isinstance(day, list) else [day])
        )
        out = starts + step
        if flat:
            ps = np.asarray([a for a, _ in flat], np.int64)
            pe = np.asarray([b for _, b in flat], np.int64)
            k = np.searchsorted(ps, starts, side="right") - 1
            inside = (k >= 0) & (starts < pe[np.maximum(k, 0)])
            out = np.where(inside, np.minimum(out, pe[np.maximum(k, 0)]), out)
        return out

    def bars_table(self, res: Mapping[str, Any], fetched_ns: int) -> pa.Table:
        sid = security_id(res["meta"]["symbol"])
        ts = np.asarray(res.get("timestamp") or [], dtype=np.int64)
        q = (res.get("indicators", {}).get("quote") or [{}])[0]

        def col(k: str) -> npt.NDArray[np.float64]:
            v = q.get(k) or [None] * ts.size
            if len(v) != ts.size:
                raise ValueError(f"{sid}: quote column {k!r} has inconsistent length")
            return np.asarray([np.nan if x is None else float(x) for x in v], np.float64)

        o, h, lo, c, vol = (col(k) for k in ("open", "high", "low", "close", "volume"))
        ends = self._session_end(res, ts)
        ok = (
            np.isfinite(o)
            & np.isfinite(h)
            & np.isfinite(lo)
            & np.isfinite(c)
            & (o > 0)
            & (c > 0)
            & (lo > 0)
            & (h >= lo)
            & (ends * _SEC <= fetched_ns)
        )
        dates, ratios = self._splits(res)
        f = self._later_split_factor(ts, dates, ratios)
        o, h, lo, c = (x * f for x in (o, h, lo, c))
        vol = np.nan_to_num(vol, nan=0.0) / f
        start_ns, end_ns = ts * _SEC, ends * _SEC
        if np.unique(start_ns[ok]).size != int(ok.sum()):
            raise ValueError(f"{sid}: duplicate bar timestamps")
        n = int(ok.sum())
        tbl = pa.table(
            {
                "security_id": [sid] * n,
                "start": pa.array(start_ns[ok], pa.int64()).cast(BARS.schema.field("start").type),
                "end": pa.array(end_ns[ok], pa.int64()).cast(BARS.schema.field("end").type),
                "open": o[ok],
                "high": h[ok],
                "low": lo[ok],
                "close": c[ok],
                "volume": vol[ok],
                "vwap": pa.array([None] * n, pa.float64()),
                "known_time": pa.array(end_ns[ok] + self.delay_ns, pa.int64()).cast(
                    BARS.schema.field("known_time").type
                ),
                "source": [NAME] * n,
            }
        )
        return conform(BARS, tbl)

    def actions_table(self, res: Mapping[str, Any]) -> pa.Table:
        sid = security_id(res["meta"]["symbol"])
        dates, ratios = self._splits(res)
        rows: list[tuple[str, dt.date, float]] = []
        for d, r in zip(dates.tolist(), ratios.tolist(), strict=True):
            rows.append(("split", dt.datetime.fromtimestamp(d, _NY).date(), r))
        divs = (res.get("events") or {}).get("dividends") or {}
        for v in divs.values():
            amt, d = float(v["amount"]), int(v["date"])
            if not amt > 0:
                raise ValueError(f"{sid}: invalid dividend {v!r}")
            factor = float(
                self._later_split_factor(np.asarray([d], dtype=np.int64), dates, ratios)[0]
            )
            rows.append(("cash_dividend", dt.datetime.fromtimestamp(d, _NY).date(), amt * factor))
        rows.sort(key=lambda r: (r[1], r[0]))
        known = [
            int(dt.datetime.combine(day, dt.time(0, 0), _NY).timestamp()) * _SEC
            for _, day, _ in rows
        ]
        tbl = pa.table(
            {
                "security_id": [sid] * len(rows),
                "action": [r[0] for r in rows],
                "ex_date": pa.array([r[1] for r in rows], pa.date32()),
                "value": [r[2] for r in rows],
                "known_time": pa.array(known, pa.int64()).cast(
                    CORPORATE_ACTIONS.schema.field("known_time").type
                ),
                "source": [NAME] * len(rows),
            }
        )
        return conform(CORPORATE_ACTIONS, tbl)

    # -------------------------------------------------------------- Provider
    def fetch(self, dataset: str, entities: Sequence[str]) -> Iterator[pa.Table]:
        for e in entities:
            res, fetched = self.chart(e)
            if dataset == BARS.name:
                yield self.bars_table(res, fetched)
            elif dataset == CORPORATE_ACTIONS.name:
                yield self.actions_table(res)
            else:
                raise ValueError(f"{NAME} does not provide {dataset!r}")

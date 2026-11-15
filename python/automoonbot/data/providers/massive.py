"""Massive (formerly Polygon.io) provider: listed US option contracts and
their daily bars.

The API key is read from ``MASSIVE_API_KEY`` (or passed in) and sent as an
``Authorization: Bearer`` header: it never appears in a URL, a cache key, a
stored row, a ``repr`` or an error message. The free plan allows 5 requests a
minute and two years of history, so the default rate is 5 per minute and a
chain is narrowed before any bars are fetched (see :class:`ChainSelection`).

Normalization
-------------
* Contract tickers ``O:SPY250117C00450000`` become security ids
  ``option:SPY250117C00450000``; underlyings become ``ticker:SPY``.
* Daily aggregates are requested unadjusted, matching the ``bars`` schema.
  A bar covers one New York session: it starts at 00:00 and ends at 16:15,
  the latest regular close of any US option class (ETF options trade 15
  minutes after the stock close), so the end is never early. Combine option
  and stock bars by session with ``MarketPanel.from_store(align="session")``.
* Massive publishes no knowledge times, so the policy is
  :attr:`KnowledgePolicy.ESTIMATED`: a bar's ``known_time`` is its end plus
  ``delay``.
* Contracts with extra deliverables (``additional_underlyings``, after
  mergers or special distributions) and non-standard contracts are skipped:
  their exercise value is not a function of one underlying's price.
* Bars are trade aggregates: an illiquid contract's close can be stale
  relative to its underlying's close. There are no quotes on the free plan.

Point-in-time universe
----------------------
Downloading only the strikes that were ever near the money would leak the
underlying's future path into the universe (a listed 120 strike while the
stock is at 100 says it will rally). :meth:`MassiveProvider.ingest_chain`
instead admits a contract on the first session whose underlying close is
within ``moneyness`` of the strike, and stores its bars and terms from that
session on, both known no earlier than that close. At any decision time the
universe depends only on closes already known.
"""

import datetime as dt
import os
import re
import urllib.parse
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa

from automoonbot.data.http import HttpClient
from automoonbot.data.providers.base import KnowledgePolicy, Provider, ProviderInfo
from automoonbot.data.providers.yahoo import security_id as stock_id
from automoonbot.data.schemas import BARS, OPTION_CONTRACTS, conform
from automoonbot.data.store import ParquetStore, WriteResult

NAME = "massive"
HOST = "api.massive.com"
_NY = ZoneInfo("America/New_York")
_SEC = 1_000_000_000
_HALF_DAY_MS = 12 * 3600 * 1000
#: OCC symbology: root, yymmdd, C/P, strike x 1000 (8 digits).
_OCC = re.compile(r"^O:([A-Z0-9.]{1,6})(\d{6})([CP])(\d{8})$")
USER_AGENT = "automoonbot-research/0.2 (+https://github.com/SnowCheetos/AutoMoonBot)"
#: Latest regular close of any US option class.
OPTION_CLOSE = dt.time(16, 15)
_CONTRACTS = "/v3/reference/options/contracts"


def option_id(ticker: str) -> str:
    """``O:SPY250117C00450000`` -> ``option:SPY250117C00450000``."""
    if not _OCC.match(ticker):
        raise ValueError(f"not an OCC option ticker: {ticker!r}")
    return f"option:{ticker[2:]}"


def option_ticker(security: str) -> str:
    """``option:SPY250117C00450000`` -> ``O:SPY250117C00450000``."""
    if not security.startswith("option:"):
        raise ValueError(f"not an option security id: {security!r}")
    ticker = f"O:{security[7:]}"
    option_id(ticker)  # validates
    return ticker


def underlying_symbol(entity: str) -> str:
    """``SPY`` or ``ticker:SPY`` -> ``SPY`` (validated)."""
    raw = entity[7:] if entity.lower().startswith("ticker:") else entity
    return stock_id(raw)[7:]


def monthly_candidates(year: int, month: int) -> tuple[dt.date, dt.date]:
    """The standard monthly expiration (the third Friday) and its fallback
    when that Friday is an exchange holiday (the Thursday before)."""
    first = dt.date(year, month, 1)
    friday = first + dt.timedelta(days=(4 - first.weekday()) % 7 + 14)
    return friday, friday - dt.timedelta(days=1)


def _months(start: dt.date, end: dt.date) -> Iterator[tuple[int, int]]:
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


@dataclass(frozen=True)
class ChainSelection:
    """Which contracts of a chain to download.

    * ``expirations``: ``"monthly"`` keeps the standard monthly expiration
      (the third Friday, or the Thursday before when the Friday is a
      holiday); ``"all"`` keeps every listed expiration.
    * ``moneyness``: a contract enters the universe on the first session,
      at most ``max_days`` before expiration, whose underlying close is
      within this fraction of the strike.
    * ``strike_step``: keep only strikes that are multiples of it (0: all).
    * ``rights``: ``("call", "put")`` or one of them.
    """

    expirations: str = "monthly"
    moneyness: float = 0.10
    max_days: int = 70
    strike_step: float = 0.0
    rights: tuple[str, ...] = ("call", "put")

    def __post_init__(self) -> None:
        if self.expirations not in ("monthly", "all"):
            raise ValueError("expirations must be 'monthly' or 'all'")
        if not 0 < self.moneyness <= 1:
            raise ValueError("moneyness must be in (0, 1]")
        if self.max_days < 1:
            raise ValueError("max_days must be at least 1")
        if not (self.strike_step >= 0 and np.isfinite(self.strike_step)):
            raise ValueError("strike_step must be finite and non-negative")
        if not self.rights or set(self.rights) - {"call", "put"}:
            raise ValueError("rights must be a non-empty subset of ('call', 'put')")

    def on_step(self, strike: float) -> bool:
        if not self.strike_step:
            return True
        q = strike / self.strike_step
        return abs(q - round(q)) <= 1e-9 * max(1.0, abs(q))


@dataclass(frozen=True)
class ChainResult:
    bars: WriteResult
    contracts: WriteResult
    #: Contracts listed in the requested range, before selection.
    listed: int
    #: Contracts that came near the money and were downloaded.
    selected: int


class MassiveProvider(Provider):
    info = ProviderInfo(
        name=NAME,
        datasets=frozenset({BARS.name, OPTION_CONTRACTS.name}),
        policy=KnowledgePolicy.ESTIMATED,
        description="Massive (Polygon) option contracts and daily option bars",
    )

    def __init__(
        self,
        api_key: str | None = None,
        *,
        requests_per_minute: float = 5.0,
        cache_dir: str | os.PathLike[str] | None = None,
        delay: dt.timedelta = dt.timedelta(minutes=15),
        client: HttpClient | None = None,
    ) -> None:
        key = (api_key if api_key is not None else os.environ.get("MASSIVE_API_KEY", "")).strip()
        if not key:
            raise ValueError("a Massive API key is required (set MASSIVE_API_KEY)")
        if not key.isascii() or not key.isprintable() or any(c.isspace() for c in key):
            raise ValueError("malformed Massive API key")
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self.delay_ns = int(delay.total_seconds() * _SEC)
        if self.delay_ns < 0:
            raise ValueError("delay must be non-negative")
        self._auth = {"Authorization": f"Bearer {key}"}
        self.client = client or HttpClient(
            USER_AGENT,
            allowed_hosts=frozenset({HOST}),
            rate_per_host=requests_per_minute / 60.0,
            cache_dir=cache_dir,
        )

    def __repr__(self) -> str:  # never show the key
        return f"MassiveProvider(host={HOST!r})"

    def _get(
        self, path_or_url: str, params: Mapping[str, Any] | None = None, *, cache: bool = True
    ) -> dict[str, Any]:
        if path_or_url.startswith("https://"):
            parts = urllib.parse.urlsplit(path_or_url)
            if parts.hostname != HOST:
                raise ValueError("refusing a URL outside the Massive API")
            url = path_or_url
        else:
            q = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
            url = f"https://{HOST}{path_or_url}" + (f"?{q}" if q else "")
        data = self.client.get(url, use_cache=cache, headers=self._auth).json()
        if not isinstance(data, dict):
            raise ValueError(f"unexpected Massive response for {path_or_url.split('?')[0]}")
        status = str(data.get("status", "")).upper()
        if status not in ("OK", "DELAYED"):
            detail = data.get("error") or data.get("message") or status or "no status"
            raise ValueError(f"Massive error: {str(detail)[:200]}")
        return data

    def _paged(
        self, path: str, params: Mapping[str, Any], *, cache: bool = True
    ) -> Iterator[dict[str, Any]]:
        data = self._get(path, params, cache=cache)
        while True:
            yield from data.get("results") or []
            nxt = data.get("next_url")
            if not nxt:
                return
            data = self._get(str(nxt), cache=cache)

    # --------------------------------------------------------------- contracts
    def contracts(
        self,
        underlying: str,
        *,
        expiration: dt.date | None = None,
        expiration_gte: dt.date | None = None,
        expiration_lte: dt.date | None = None,
        strike_gte: float | None = None,
        strike_lte: float | None = None,
    ) -> list[dict[str, Any]]:
        """Every contract on ``underlying`` (expired and live) matching the
        filters, in listing order, following pagination.

        Listings of contracts whose expiration is today or later are not
        cached: new strikes are listed until expiration.
        """
        if expiration is not None and (expiration_gte or expiration_lte):
            raise ValueError("give expiration or a range, not both")
        params: dict[str, Any] = {
            "underlying_ticker": underlying_symbol(underlying),
            "expiration_date": expiration.isoformat() if expiration else None,
            "expiration_date.gte": expiration_gte.isoformat() if expiration_gte else None,
            "expiration_date.lte": expiration_lte.isoformat() if expiration_lte else None,
            "strike_price.gte": strike_gte,
            "strike_price.lte": strike_lte,
            "limit": 1000,
            "sort": "expiration_date",
            "order": "asc",
        }
        today = dt.datetime.now(_NY).date()
        last = expiration or expiration_lte
        settled = last is not None and last < today
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        # Massive lists expired and live contracts separately.
        for expired in ("true", "false"):
            if expired == "false" and settled:
                continue
            for c in self._paged(_CONTRACTS, params | {"expired": expired}, cache=settled):
                t = c.get("ticker", "")
                if isinstance(t, str) and t not in seen and _OCC.match(t):
                    seen.add(t)
                    out.append(c)
        return out

    def monthly_contracts(
        self,
        underlying: str,
        start: dt.date,
        end: dt.date,
        *,
        strike_bounds: Callable[[dt.date], tuple[float, float] | None] | None = None,
    ) -> list[dict[str, Any]]:
        """Contracts of the standard monthly expirations in ``[start, end]``:
        one listing per month (two when the third Friday was a holiday),
        optionally limited to the strikes ``strike_bounds(expiration)``."""
        out: list[dict[str, Any]] = []
        for y, m in _months(start, end):
            for e in monthly_candidates(y, m):
                if not start <= e <= end:
                    continue
                bounds = strike_bounds(e) if strike_bounds else (None, None)
                if bounds is None:
                    break
                found = self.contracts(
                    underlying, expiration=e, strike_gte=bounds[0], strike_lte=bounds[1]
                )
                if found:
                    out.extend(found)
                    break
        return out

    def contract_row(self, c: Mapping[str, Any], known_ns: int) -> dict[str, Any] | None:
        """Canonical contract row, or ``None`` for contracts this layer does
        not model (extra deliverables, non-standard types)."""
        if c.get("additional_underlyings") or c.get("type", "standard") != "standard":
            return None
        right, style = c.get("contract_type"), c.get("exercise_style")
        if right not in ("call", "put") or style not in ("american", "european"):
            return None
        try:
            strike = float(c["strike_price"])
            spc = float(c.get("shares_per_contract") or 100)
            expiration = dt.date.fromisoformat(c["expiration_date"])
            underlying = stock_id(c["underlying_ticker"])
            sid = option_id(c["ticker"])
        except KeyError, TypeError, ValueError:
            return None
        if not (np.isfinite(strike) and strike > 0 and np.isfinite(spc) and spc > 0):
            return None
        return {
            "security_id": sid,
            "underlying_id": underlying,
            "right": right,
            "strike": strike,
            "expiration": expiration,
            "exercise_style": style,
            # Listed equity and ETF options settle physically; this provider
            # covers no index (cash-settled) options.
            "settlement": "physical",
            "multiplier": 100.0,
            "deliverable": spc,
            "known_time": known_ns,
            "source": NAME,
        }

    # -------------------------------------------------------------------- bars
    def daily_bars(self, security: str, first: dt.date, last: dt.date) -> pa.Table:
        """Unadjusted daily bars of one option contract, as ``bars`` rows.

        Ranges reaching today are not cached (the session may be open)."""
        if first > last:
            raise ValueError("first must not be after last")
        ticker = option_ticker(security)
        today = dt.datetime.now(_NY).date()
        data = self._get(
            f"/v2/aggs/ticker/{urllib.parse.quote(ticker, safe=':')}/range/1/day/"
            f"{first.isoformat()}/{last.isoformat()}",
            {"adjusted": "false", "sort": "asc", "limit": 50000},
            cache=last < today,
        )
        cols: dict[str, list[Any]] = {
            k: [] for k in ("start", "end", "open", "high", "low", "close", "volume", "vwap")
        }
        for r in data.get("results") or []:
            # ``t`` opens the session's window (midnight New York, or UTC on
            # some endpoints); half a day later is inside the session date.
            day = dt.datetime.fromtimestamp((int(r["t"]) + _HALF_DAY_MS) / 1000, _NY).date()
            if not first <= day <= last or day >= today:
                continue
            o, h, lo, c = (float(r[k]) for k in ("o", "h", "l", "c"))
            if not all(np.isfinite(x) and x > 0 for x in (o, h, lo, c)):
                continue
            if not lo <= min(o, c) <= max(o, c) <= h:
                continue
            vol = float(r.get("v") or 0.0)
            vw = r.get("vw")
            cols["start"].append(int(dt.datetime.combine(day, dt.time(0), _NY).timestamp()) * _SEC)
            cols["end"].append(int(dt.datetime.combine(day, OPTION_CLOSE, _NY).timestamp()) * _SEC)
            cols["open"].append(o)
            cols["high"].append(h)
            cols["low"].append(lo)
            cols["close"].append(c)
            cols["volume"].append(vol if np.isfinite(vol) and vol >= 0 else 0.0)
            cols["vwap"].append(float(vw) if vw is not None and np.isfinite(float(vw)) else None)
        if len(set(cols["start"])) != len(cols["start"]):
            raise ValueError(f"{security}: duplicate daily bars")
        n = len(cols["start"])
        ts = BARS.schema.field("start").type
        ends = np.asarray(cols["end"], dtype=np.int64)
        tbl = pa.table(
            {
                "security_id": [security] * n,
                "start": pa.array(cols["start"], pa.int64()).cast(ts),
                "end": pa.array(ends, pa.int64()).cast(ts),
                "open": pa.array(cols["open"], pa.float64()),
                "high": pa.array(cols["high"], pa.float64()),
                "low": pa.array(cols["low"], pa.float64()),
                "close": pa.array(cols["close"], pa.float64()),
                "volume": pa.array(cols["volume"], pa.float64()),
                "vwap": pa.array(cols["vwap"], pa.float64()),
                "known_time": pa.array(ends + self.delay_ns, pa.int64()).cast(ts),
                "source": [NAME] * n,
            }
        )
        return conform(BARS, tbl)

    # ------------------------------------------------------------------ chains
    def ingest_chain(
        self,
        store: ParquetStore,
        underlying: str,
        closes: Mapping[dt.date, float],
        *,
        start: dt.date,
        end: dt.date,
        selection: ChainSelection | None = None,
        progress: Callable[[int, int, str], None] | None = None,
    ) -> ChainResult:
        """Select contracts on ``underlying`` expiring in ``[start, end]``
        and write their daily bars and terms to ``store``.

        ``closes`` are the underlying's as-traded daily closes by session
        date (the same units as the strikes). A contract is admitted on the
        first session at most ``selection.max_days`` before its expiration
        whose close is within ``selection.moneyness`` of its strike, and its
        bars are stored from that session on; its terms row is known from its
        first stored bar. Each contract is written as soon as it is fetched,
        so an interrupted run resumes from the HTTP cache.
        """
        sel = selection or ChainSelection()
        days = sorted(d for d, p in closes.items() if np.isfinite(p) and p > 0)
        if not days:
            raise ValueError("no valid underlying closes")
        day_ord = np.array([d.toordinal() for d in days], dtype=np.int64)
        px = np.array([closes[d] for d in days], dtype=np.float64)

        def window(e: dt.date) -> slice:
            lo = np.searchsorted(day_ord, (e - dt.timedelta(days=sel.max_days)).toordinal())
            hi = np.searchsorted(day_ord, e.toordinal(), side="right")
            return slice(int(lo), int(hi))

        def bounds(e: dt.date) -> tuple[float, float] | None:
            w = px[window(e)]
            if w.size == 0:
                return None
            return float(w.min() * (1 - sel.moneyness)), float(w.max() * (1 + sel.moneyness))

        if sel.expirations == "monthly":
            chain = self.monthly_contracts(underlying, start, end, strike_bounds=bounds)
        else:
            span = px[window(start).start : window(end).stop]
            if span.size == 0:
                raise ValueError("no underlying closes in the selection window")
            chain = self.contracts(
                underlying,
                expiration_gte=start,
                expiration_lte=end,
                strike_gte=float(span.min() * (1 - sel.moneyness)),
                strike_lte=float(span.max() * (1 + sel.moneyness)),
            )
        plan: list[tuple[dict[str, Any], dt.date, dt.date]] = []
        for c in chain:
            try:
                e = dt.date.fromisoformat(c["expiration_date"])
                k = float(c["strike_price"])
            except KeyError, TypeError, ValueError:
                continue
            if not (start <= e <= end) or c.get("contract_type") not in sel.rights:
                continue
            if not (np.isfinite(k) and k > 0 and sel.on_step(k)):
                continue
            w = window(e)
            near = np.flatnonzero(np.abs(k / px[w] - 1.0) <= sel.moneyness)
            row = self.contract_row(c, 0) if near.size else None
            if row is not None:
                plan.append((row, days[w.start + int(near[0])], e))

        bars_new = bars_dup = con_new = con_dup = 0
        files: list[str] = []
        cfiles: list[str] = []
        ts = OPTION_CONTRACTS.schema.field("known_time").type
        for n, (row, admitted, e) in enumerate(plan):
            sid = row["security_id"]
            if progress:
                progress(n, len(plan), sid)
            bars = self.daily_bars(sid, admitted, e)
            if bars.num_rows == 0:
                continue
            r = store.write(BARS.name, bars)
            bars_new, bars_dup = bars_new + r.new, bars_dup + r.duplicate
            files.extend(r.files)
            known = bars.column("known_time").cast(pa.int64())[0].as_py()
            tbl = pa.table({k: [v] for k, v in row.items() if k != "known_time"})
            tbl = tbl.append_column("known_time", pa.array([known], pa.int64()).cast(ts))
            rc = store.write(OPTION_CONTRACTS.name, conform(OPTION_CONTRACTS, tbl))
            con_new, con_dup = con_new + rc.new, con_dup + rc.duplicate
            cfiles.extend(rc.files)
        return ChainResult(
            WriteResult(bars_new, bars_dup, tuple(files)),
            WriteResult(con_new, con_dup, tuple(cfiles)),
            listed=len(chain),
            selected=len(plan),
        )

    # -------------------------------------------------------------- Provider
    def fetch(self, dataset: str, entities: Sequence[str]) -> Iterator[pa.Table]:
        """``bars`` of option security ids over the last two years (the free
        plan's history). Chains are ingested with :meth:`ingest_chain`."""
        if dataset != BARS.name:
            raise ValueError(f"use ingest_chain for {dataset!r}")
        today = dt.datetime.now(_NY).date()
        for e in entities:
            yield self.daily_bars(e, today - dt.timedelta(days=730), today)

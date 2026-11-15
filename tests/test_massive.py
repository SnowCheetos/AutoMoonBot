"""The Massive options provider against a fake API, end to end through the
store, a session-aligned panel and the simulator."""

import datetime as dt
import json
import urllib.parse
from dataclasses import replace
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pytest

from automoonbot.backtest import MarketPanel, Simulator
from automoonbot.data import ParquetStore
from automoonbot.data.http import Response
from automoonbot.data.providers.massive import (
    HOST,
    ChainSelection,
    MassiveProvider,
    monthly_candidates,
    option_id,
    option_ticker,
)
from automoonbot.data.schemas import BARS, conform

NY = ZoneInfo("America/New_York")
KEY = "test-key-abc123"
EXPIRY = dt.date(2024, 6, 21)  # third Friday of June 2024
DAYS = [
    d.astype(dt.date)
    for d in np.arange(np.datetime64("2024-05-01"), np.datetime64("2024-06-22"))
    if np.is_busday(d)
]
# The underlying rallies from 100 to 115 over the window.
CLOSE = dict(zip(DAYS, np.linspace(100.0, 115.0, len(DAYS)), strict=True))


def occ(right: str, strike: float, expiry: dt.date = EXPIRY) -> str:
    return f"O:XYZ{expiry:%y%m%d}{right[0].upper()}{round(strike * 1000):08d}"


def contract(right: str, strike: float, expiry: dt.date = EXPIRY, **extra: Any) -> dict[str, Any]:
    return {
        "ticker": occ(right, strike, expiry),
        "underlying_ticker": "XYZ",
        "contract_type": right,
        "exercise_style": "american",
        "expiration_date": expiry.isoformat(),
        "strike_price": strike,
        "shares_per_contract": 100,
        **extra,
    }


CHAIN = [contract("call", k) for k in (100.0, 105.0, 110.0, 115.0, 125.0, 150.0)] + [
    contract("put", 95.0),
    contract("call", 107.0, additional_underlyings=[{"underlying": "ABC", "amount": 10}]),
    contract("call", 105.0, dt.date(2024, 6, 14)),  # a weekly: never requested
]


def option_price(ticker: str, day: dt.date) -> float:
    right, strike = ticker[-9], int(ticker[-8:]) / 1000
    s = CLOSE[day]
    return max(s - strike if right == "C" else strike - s, 0.0) + 1.0


class FakeApi:
    page = 2

    def __init__(self, *, utc_midnight: bool = False) -> None:
        self.calls: list[tuple[str, dict[str, str], bool]] = []
        self.utc_midnight = utc_midnight

    def get(
        self, url: str, *, use_cache: bool = True, headers: dict[str, str] | None = None
    ) -> Response:
        self.calls.append((url, dict(headers or {}), use_cache))
        parts = urllib.parse.urlsplit(url)
        assert parts.hostname == HOST
        q = {k: v[0] for k, v in urllib.parse.parse_qs(parts.query).items()}
        if parts.path == "/v3/reference/options/contracts":
            body = self.contracts(q)
        elif parts.path.startswith("/v2/aggs/ticker/"):
            body = self.aggs(urllib.parse.unquote(parts.path))
        else:
            raise AssertionError(url)
        return Response(url, 200, json.dumps(body).encode(), 0, False)

    def contracts(self, q: dict[str, str]) -> dict[str, Any]:
        rows = [c for c in CHAIN if q["underlying_ticker"] == c["underlying_ticker"]]
        if "expiration_date" in q:
            rows = [c for c in rows if c["expiration_date"] == q["expiration_date"]]
        for key, op in (("strike_price.gte", float.__ge__), ("strike_price.lte", float.__le__)):
            if key in q:
                rows = [c for c in rows if op(float(c["strike_price"]), float(q[key]))]
        if q["expired"] == "false":
            rows = []
        start = int(q.get("cursor", 0))
        body: dict[str, Any] = {"status": "OK", "results": rows[start : start + self.page]}
        if start + self.page < len(rows):
            nxt = q | {"cursor": str(start + self.page)}
            body["next_url"] = (
                f"https://{HOST}/v3/reference/options/contracts?{urllib.parse.urlencode(nxt)}"
            )
        return body

    def aggs(self, path: str) -> dict[str, Any]:
        parts = path.split("/")
        ticker, first, last = (
            parts[4],
            dt.date.fromisoformat(parts[8]),
            dt.date.fromisoformat(parts[9]),
        )
        out = []
        for d in DAYS:
            if first <= d <= last:
                tz = dt.UTC if self.utc_midnight else NY
                t = int(dt.datetime.combine(d, dt.time(0), tz).timestamp() * 1000)
                p = option_price(ticker, d)
                out.append({"t": t, "o": p, "h": p, "l": p, "c": p, "v": 10.0, "vw": p})
        # A corrupt bar is dropped.
        out.append({"t": out[-1]["t"] + 86_400_000, "o": 1.0, "h": 0.5, "l": 1.0, "c": 1.0})
        return {"status": "OK", "results": out}


def provider(api: FakeApi | None = None) -> MassiveProvider:
    return MassiveProvider(KEY, client=api or FakeApi())  # type: ignore[arg-type]


def test_identifiers_and_calendar() -> None:
    assert option_id("O:SPY250117C00450000") == "option:SPY250117C00450000"
    assert option_ticker("option:SPY250117C00450000") == "O:SPY250117C00450000"
    for bad in ("SPY250117C00450000", "O:SPY250117X00450000"):
        with pytest.raises(ValueError):
            option_id(bad)
    assert monthly_candidates(2024, 6) == (dt.date(2024, 6, 21), dt.date(2024, 6, 20))
    # Good Friday 2025 fell on the third Friday: the Thursday is the fallback.
    assert monthly_candidates(2025, 4) == (dt.date(2025, 4, 18), dt.date(2025, 4, 17))
    assert monthly_candidates(2024, 11)[0] == dt.date(2024, 11, 15)
    with pytest.raises(ValueError):
        ChainSelection(rights=("calls",))


def test_key_stays_in_the_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MASSIVE_API_KEY"):
        MassiveProvider(client=FakeApi())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="malformed"):
        MassiveProvider("two words", client=FakeApi())  # type: ignore[arg-type]
    monkeypatch.setenv("MASSIVE_API_KEY", KEY)
    api = FakeApi()
    m = MassiveProvider(client=api)  # type: ignore[arg-type]
    assert KEY not in repr(m)
    m.contracts("ticker:XYZ", expiration=EXPIRY)
    assert api.calls
    for url, headers, _ in api.calls:
        assert KEY not in url
        assert headers == {"Authorization": f"Bearer {KEY}"}


def test_errors_and_foreign_pagination() -> None:
    class Failing(FakeApi):
        def get(
            self, url: str, *, use_cache: bool = True, headers: dict[str, str] | None = None
        ) -> Response:
            body = {"status": "ERROR", "error": "Unknown API Key"}
            return Response(url, 200, json.dumps(body).encode(), 0, False)

    with pytest.raises(ValueError, match="Unknown API Key") as err:
        provider(Failing()).contracts("XYZ", expiration=EXPIRY)
    assert KEY not in str(err.value)

    class Hijack(FakeApi):
        def contracts(self, q: dict[str, str]) -> dict[str, Any]:
            return {"status": "OK", "results": [], "next_url": "https://evil.example/steal"}

    with pytest.raises(ValueError, match="outside the Massive API"):
        provider(Hijack()).contracts("XYZ", expiration=EXPIRY)


def test_contracts_paginate_and_settled_listings_are_cached() -> None:
    api = FakeApi()
    got = provider(api).contracts("XYZ", expiration=EXPIRY)
    assert [c["ticker"] for c in got] == [c["ticker"] for c in CHAIN[:8]]
    assert len(api.calls) == 4  # four pages, expired listings only
    assert all("expired=true" in u and cache for u, _, cache in api.calls)


@pytest.mark.parametrize("utc_midnight", [False, True])
def test_daily_bars(utc_midnight: bool) -> None:
    m = provider(FakeApi(utc_midnight=utc_midnight))
    sid = option_id(occ("call", 105.0))
    t = m.daily_bars(sid, dt.date(2024, 6, 3), EXPIRY)
    days = [d for d in DAYS if d >= dt.date(2024, 6, 3)]
    assert t.num_rows == len(days)  # the corrupt bar is dropped
    start = t.column("start").cast(pa.int64()).to_numpy()
    end = t.column("end").cast(pa.int64()).to_numpy()
    known = t.column("known_time").cast(pa.int64()).to_numpy()
    first = dt.datetime.combine(days[0], dt.time(0), NY)
    assert start[0] == int(first.timestamp()) * 10**9
    assert end[0] - start[0] == (16 * 3600 + 15 * 60) * 10**9
    assert np.all(known == end + 15 * 60 * 10**9)
    assert t.column("close").to_pylist()[-1] == pytest.approx(
        option_price(occ("call", 105.0), EXPIRY)
    )


def stock_bars() -> pa.Table:
    ts = BARS.schema.field("start").type
    start = [int(dt.datetime.combine(d, dt.time(9, 30), NY).timestamp()) * 10**9 for d in DAYS]
    end = [int(dt.datetime.combine(d, dt.time(16), NY).timestamp()) * 10**9 for d in DAYS]
    c = [CLOSE[d] for d in DAYS]
    n = len(DAYS)
    return conform(
        BARS,
        pa.table(
            {
                "security_id": ["ticker:XYZ"] * n,
                "start": pa.array(start, pa.int64()).cast(ts),
                "end": pa.array(end, pa.int64()).cast(ts),
                "open": c,
                "high": c,
                "low": c,
                "close": c,
                "volume": [1e9] * n,
                "vwap": c,
                "known_time": pa.array([e + 60 * 10**9 for e in end], pa.int64()).cast(ts),
                "source": ["test"] * n,
            }
        ),
    )


def test_chain_is_point_in_time_and_simulates(tmp_path: Any) -> None:
    api = FakeApi()
    store = ParquetStore(tmp_path)
    sel = ChainSelection(moneyness=0.05, max_days=40)
    res = provider(api).ingest_chain(
        store, "XYZ", CLOSE, start=dt.date(2024, 6, 1), end=dt.date(2024, 6, 30), selection=sel
    )
    # The listing is limited to strikes within 5% of the window's closes (no
    # 95 put, 125 or 150 call); the extra-deliverable call is not modeled.
    assert res.listed == 5 and res.selected == 4 and res.contracts.new == 4
    lists = [u for u, _, _ in api.calls if "/v3/" in u]
    assert all("expiration_date=2024-06-21" in u for u in lists)
    bars = {
        u.split("/")[6]: u.split("/")[10].split("?")[0] for u, _, _ in api.calls if "/aggs/" in u
    }
    window = [d for d in DAYS if d >= EXPIRY - dt.timedelta(days=40)]
    for strike in (100.0, 105.0, 110.0, 115.0):
        admitted = next(d for d in window if abs(strike / CLOSE[d] - 1) <= 0.05)
        assert bars[occ("call", strike)] == admitted.isoformat()
    assert occ("call", 125.0) not in bars and occ("call", 150.0) not in bars
    # Re-running writes nothing new.
    again = provider().ingest_chain(
        store, "XYZ", CLOSE, start=dt.date(2024, 6, 1), end=dt.date(2024, 6, 30), selection=sel
    )
    assert again.bars.new == 0 and again.contracts.new == 0

    store.write("bars", stock_bars())
    ids = ["ticker:XYZ"] + [option_id(occ("call", k)) for k in (105.0, 115.0)]
    p = MarketPanel.from_store(store, ids, align="session")
    assert p.shape[0] == len(DAYS)  # one row per session despite 16:00 and 16:15 ends
    p = p.with_contracts(store)
    assert p.instruments is not None
    c115 = p.instruments[2]
    assert c115["underlying"] == 0 and c115["strike"] == 115.0
    assert c115["expiry"] == len(DAYS) - 1
    first115 = int(np.argmax(p.listed[:, 2]))
    assert DAYS[first115] == next(d for d in window if abs(115 / CLOSE[d] - 1) <= 0.05)
    assert not p.listed[:first115, 2].any()
    # Contract terms are not known before the first bar.
    terms = store.read("option_contracts", int(p.decision_time[first115 - 1]))
    assert option_id(occ("call", 115.0)) not in terms.column("security_id").to_pylist()

    # Buy one 105 call when it first trades and hold it through exercise.
    p = replace(p, adv_shares=np.full(p.shape, 1e12))
    k0 = int(np.argmax(p.tradable[:, 1]))
    targets = np.full((p.shape[0] - 1 - k0, 3), np.nan)
    targets[0] = [0.0, 1.0, 0.0]
    free = dict(commission_bps=0.0, half_spread_bps=0.0, impact_coef=0.0)
    free |= dict(option_commission=0.0, option_half_spread_bps=0.0, option_min_half_spread=0.0)
    sim = Simulator(p.market(), start=k0, capital=10_000.0, max_weight=10.0, **free)
    out = sim.run(targets, units=True)
    assert out["exercised"][-1] == 1.0
    assert sim.shares().tolist() == [100.0, 0.0, 0.0]
    paid = 100 * p.open[k0 + 1, 1]
    assert out["equity"][-1] == pytest.approx(10_000.0 - paid + 100 * (CLOSE[EXPIRY] - 105.0))


def test_bars_sharing_a_row_are_rejected(tmp_path: Any) -> None:
    store = ParquetStore(tmp_path)
    bars = stock_bars()
    store.write("bars", bars)
    # A second, earlier bar in the first session (09:30-15:00).
    one = bars.slice(0, 1)
    end = one.column("end").cast(pa.int64())[0].as_py() - 3600 * 10**9
    ts = BARS.schema.field("end").type
    store.write("bars", conform(BARS, one.set_column(2, "end", pa.array([end]).cast(ts))))
    assert MarketPanel.from_store(store, ["ticker:XYZ"]).shape[0] == len(DAYS) + 1
    with pytest.raises(ValueError, match="share a row"):
        MarketPanel.from_store(store, ["ticker:XYZ"], align="session")
    with pytest.raises(ValueError, match="align"):
        MarketPanel.from_store(store, ["ticker:XYZ"], align="day")

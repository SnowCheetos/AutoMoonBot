"""Tests for the Yahoo chart provider on a synthetic response."""

import datetime as dt
import json
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest
from automoonbot.backtest import MarketPanel
from automoonbot.data import ParquetStore
from automoonbot.data.http import Response
from automoonbot.data.providers.yahoo import YahooProvider, security_id

NY = ZoneInfo("America/New_York")


def _ts(day: int, hour: int, minute: int = 30) -> int:
    return int(dt.datetime(2024, 6, day, hour, minute, tzinfo=NY).timestamp())


def chart() -> dict[str, Any]:
    # Two sessions of hourly bars; a 2:1 split effective on June 4. Yahoo
    # reports split-adjusted prices, so pre-split bars appear halved.
    starts = [_ts(3, h) for h in range(9, 16)] + [_ts(4, h) for h in range(9, 16)]
    n = len(starts)
    close = [50.0 + i for i in range(n)]
    close[3] = None  # a missing bar
    periods = [
        [{"start": _ts(d, 9), "end": int(dt.datetime(2024, 6, d, 16, tzinfo=NY).timestamp())}]
        for d in (3, 4)
    ]
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "meta": {"symbol": "XYZ", "tradingPeriods": periods},
                    "timestamp": starts,
                    "indicators": {
                        "quote": [
                            {
                                "open": [c and c - 0.5 for c in close],
                                "high": [c and c + 1 for c in close],
                                "low": [c and c - 1 for c in close],
                                "close": close,
                                "volume": [1000.0] * n,
                            }
                        ]
                    },
                    "events": {
                        "splits": {"s": {"date": _ts(4, 9), "numerator": 2, "denominator": 1}},
                        "dividends": {"d": {"date": _ts(3, 9), "amount": 0.25}},
                    },
                }
            ],
        }
    }


class FakeClient:
    def __init__(self, fetched: int) -> None:
        self.fetched = fetched

    def get(self, url: str, *, use_cache: bool = True) -> Response:
        assert "chart/XYZ" in url and "events=div%2Csplits" in url
        return Response(url, 200, json.dumps(chart()).encode(), self.fetched, False)


def provider(fetched: dt.datetime) -> YahooProvider:
    return YahooProvider("1h", client=FakeClient(int(fetched.timestamp()) * 10**9))  # type: ignore[arg-type]


def test_bars_are_unadjusted_and_end_at_session_close() -> None:
    y = provider(dt.datetime(2024, 6, 5, tzinfo=NY))
    res, fetched = y.chart("xyz")
    t = y.bars_table(res, fetched)
    assert t.num_rows == 13  # one missing bar dropped
    close = t.column("close").to_pylist()
    assert close[0] == 100.0  # pre-split bars restored to as-traded prices
    assert close[-1] == 50.0 + 13  # post-split unchanged
    vol = t.column("volume").to_pylist()
    assert vol[0] == 500.0 and vol[-1] == 1000.0
    ends = t.column("end").cast("int64").to_numpy() // 10**9
    assert ends[0] - _ts(3, 9) == 3600
    assert ends[5] == int(dt.datetime(2024, 6, 3, 16, tzinfo=NY).timestamp())  # 15:30 bar
    known = t.column("known_time").cast("int64").to_numpy() // 10**9
    assert np.all(known == ends + 60)


def test_unfinished_bars_are_dropped() -> None:
    y = provider(dt.datetime(2024, 6, 4, 12, 0, tzinfo=NY))
    res, fetched = y.chart("XYZ")
    t = y.bars_table(res, fetched)
    ends = t.column("end").cast("int64").to_numpy()
    assert ends.max() <= fetched and t.num_rows == 6 + 2  # 10:30 and 11:30 bars of June 4


def test_actions_and_panel_round_trip(tmp_path: Any) -> None:
    y = provider(dt.datetime(2024, 6, 5, tzinfo=NY))
    a = y.actions_table(y.chart("XYZ")[0])
    assert a.column("action").to_pylist() == ["cash_dividend", "split"]
    assert a.column("value").to_pylist() == [0.5, 2.0]  # dividend per as-traded share
    st = ParquetStore(tmp_path)
    y.ingest(st, "bars", ["XYZ"])
    y.ingest(st, "corporate_actions", ["XYZ"])
    p = MarketPanel.from_store(st, [security_id("XYZ")])
    k = int(np.flatnonzero(p.split[:, 0] != 1)[0])
    assert (
        p.split[k, 0] == 2.0 and p.times[k] - p.times[k - 1] > 3600 * 10**9
    )  # first bar of June 4
    r = p.log_returns()[:, 0]
    assert np.nanmax(np.abs(r)) < 0.1  # the split is not a return


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        security_id("bad symbol")
    with pytest.raises(ValueError):
        YahooProvider("5m")

"""Tests for point-in-time index membership."""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

from automoonbot.data.membership import Snapshots, coverage

NY_CLOSE_UTC = 21  # 16:00 New York in winter


def bar(d: dt.date) -> int:
    return int(dt.datetime(d.year, d.month, d.day, NY_CLOSE_UTC, tzinfo=dt.UTC).timestamp()) * 10**9


def test_snapshots_mask_and_coverage(tmp_path: Path) -> None:
    p = tmp_path / "hist.csv"
    p.write_text('date,tickers\n2020-01-06,"AAA,BBB"\n2020-01-02,"AAA,CCC"\n')
    snap = Snapshots.from_csv(p)  # rows are sorted on read
    assert snap.dates == [dt.date(2020, 1, 2), dt.date(2020, 1, 6)]
    days = [dt.date(2019, 12, 31), dt.date(2020, 1, 2), dt.date(2020, 1, 3), dt.date(2020, 1, 6)]
    t = np.array([bar(d) for d in days])
    secs = ["ticker:AAA", "ticker:BBB", "ticker:CCC", "ticker:ZZZ"]
    m = snap.mask(t, secs, lambda s: s.split(":")[1] if s != "ticker:ZZZ" else None)
    np.testing.assert_array_equal(
        m,
        [
            [False, False, False, False],  # before the first snapshot
            [True, False, True, False],
            [True, False, True, False],
            [True, True, False, False],  # the change takes effect on its date
        ],
    )
    tradable = np.ones_like(m)
    tradable[3, 1] = False  # BBB has no bar on its first member day
    cov = coverage(snap, t, m, tradable)
    assert cov == {2020: pytest.approx(5 / 6)}  # 2019 has no snapshot, so no member-days
    later = snap.extended(dt.date(2020, 2, 3), ["AAA"])
    assert later.symbols(dt.date(2020, 1, 10)) == {"AAA", "BBB"}
    with pytest.raises(ValueError):
        snap.extended(dt.date(2020, 1, 1), [])


def test_require_history_drops_runs_of_reused_symbols() -> None:
    from automoonbot.data.membership import require_history

    member = np.zeros((100, 3), dtype=bool)
    member[0:30, 0] = True  # old company; the series only starts at 60
    member[70:90, 0] = True  # the symbol's current owner, a member later
    member[10:50, 1] = True  # a spin-off added on its 12th bar: kept
    member[5:20, 2] = True
    tradable = np.zeros_like(member)
    tradable[60:, 0] = True
    tradable[22:, 1] = True
    tradable[:, 2] = True
    out = require_history(member, tradable, slack=21)
    assert not out[:30, 0].any() and out[70:90, 0].all()
    assert out[10:50, 1].all() and out[5:20, 2].all()

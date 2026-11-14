"""PGPerceptron's technical descriptors, computed causally and vectorized.

The original (``utils/descriptors.py`` in SnowCheetos/PGPerceptron) recomputed
every descriptor from a rolling queue of the last ``queue_size`` bars at each
step and used the final value. :func:`descriptors` produces exactly those
final values for every bar at once: recursive indicators (EMA, Wilder RSI)
are seeded at the start of the queue as in the original, which makes them
fixed-length linear filters, evaluated here as one matrix-vector product per
window. :func:`reference_descriptors` is a literal port of the original
``compute`` used to test the equivalence.

Differences from the original, all deliberate:

* Prices are the causal total-return index (splits and dividends applied as
  of each bar), so a split is not a 50% crash. Every descriptor is
  scale-free, so this equals back-adjusting as of the decision time.
* Each security's queue holds its last ``queue_size`` *bars*; a bar is
  produced only once the queue is full (the original also emitted states
  while the queue was filling, from a shorter, differently seeded queue).
* Degenerate divisions that produced NaN/inf in the original (flat windows
  in the z-score and the stochastic oscillator) yield the neutral value 0.
* Time of day / day of week are taken in the exchange time zone
  (``America/New_York`` by default) at the bar's end, rather than the
  machine's local zone.
"""

import datetime as dt
from collections.abc import Mapping, Sequence
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from automoonbot.backtest.panel import MarketPanel

F64 = npt.NDArray[np.float64]

#: ``feature_params`` from the original ``config.json`` (122 features).
FEATURE_PARAMS: dict[str, Any] = {
    "ntw": [True],
    "nts": [True],
    "cdl": [4, 8, 16, 32, 64, 128],
    "grd": [4, 8, 10, 14, 19, 26, 35, 47, 64, 128, 172, 256],
    "nrm": [4, 8, 10, 14, 19, 26, 35, 47, 64, 128, 172, 256],
    "zsc": [4, 8, 10, 14, 19, 26, 35, 47, 64, 128, 172, 256],
    "sma": [4, 8, 10, 14, 19, 26, 35, 47, 64, 128, 172, 256],
    "ema": [4, 8, 10, 14, 19, 26, 35, 47, 64, 128, 172, 256],
    "rsi": [4, 8, 10, 14, 19, 26, 35, 47, 64, 128, 172, 256],
    "sto": {
        "window": [8, 16, 31, 64, 128, 256, 8, 16, 31, 64, 128, 256],
        "k": [3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
        "d": [3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8],
    },
}
QUEUE_SIZE = 300
_KNOWN = frozenset({"sma", "ema", "rsi", "sto", "zsc", "nrm", "grd", "cov", "cdl", "nts", "ntw"})


def feature_dim(params: Mapping[str, Any] = FEATURE_PARAMS) -> int:
    n = 0
    for k, v in params.items():
        if k not in _KNOWN:
            raise ValueError(f"unknown descriptor {k!r}")
        if k == "sto":
            n += 2 * len(v["window"])
        elif k == "cdl":
            n += 4 * len(v)
        else:
            n += len(v)
    return n


def _min_queue(params: Mapping[str, Any]) -> int:
    need = 1
    for k, v in params.items():
        if k == "sto":
            for w, a, b in zip(v["window"], v["k"], v["d"], strict=True):
                need = max(need, w + a + b - 2)
        elif k == "rsi":
            need = max(need, *(w + 1 for w in v))
        elif k in ("nts", "ntw"):
            continue
        else:
            need = max(need, *v)
    return need


def _last(x: F64, w: int) -> F64:
    """``[n, w]`` view of the trailing ``w`` values ending at each row
    ``w - 1 .. len - 1``."""
    return sliding_window_view(x, w)


def _series_features(
    o: F64,
    h: F64,
    lo: F64,
    c: F64,
    end_ns: npt.NDArray[np.int64],
    params: Mapping[str, Any],
    queue: int,
    tz: ZoneInfo,
) -> tuple[F64, F64]:
    """Features ``[n, F]`` and coefficient of variation ``[n]`` for one
    gap-free series; rows before the queue fills are NaN."""
    n = c.size
    f = feature_dim(params)
    out = np.full((n, f), np.nan)
    cov = np.full(n, np.nan)
    if n < queue:
        return out, cov
    rows = slice(queue - 1, n)  # rows with a full queue
    m = n - queue + 1
    q = _last(c, queue)  # [m, queue]
    cov[rows] = q.std(axis=1) / q.mean(axis=1)
    cols: list[F64] = []
    last = c[rows]

    def tail(x: F64, w: int) -> F64:
        # Window of w values ending at each full-queue row.
        return _last(x, w)[queue - w :]

    for key, val in params.items():
        if key == "ntw" or key == "nts":
            local = [dt.datetime.fromtimestamp(int(t) / 1e9, tz) for t in end_ns[rows]]
            if key == "ntw":
                cols.append(np.array([2 * d.weekday() / 6 - 1 for d in local]))
            else:
                secs = np.array([d.hour * 3600 + d.minute * 60 + d.second for d in local])
                cols.append(2 * secs / 86_400 - 1)
        elif key == "cdl":
            qo, qh, ql = (
                o[: n - queue + 1],
                _last(h, queue).max(axis=1),
                _last(lo, queue).min(axis=1),
            )
            for w in val:
                mean = (
                    tail(o, w).sum(axis=1)
                    + tail(h, w).sum(axis=1)
                    + tail(lo, w).sum(axis=1)
                    + tail(c, w).sum(axis=1)
                ) / (4 * w)
                for x in (qo, qh, ql, last):
                    cols.append(x / mean - 1)
        elif key == "grd":
            for w in val:
                win = tail(c, w)
                cols.append((win[:, -1] - win[:, 0]) / win.mean(axis=1))
        elif key == "nrm":
            for w in val:
                cols.append(last / tail(c, w).mean(axis=1) - 1)
        elif key == "zsc":
            for w in val:
                win = tail(c, w)
                sd = win.std(axis=1)
                z = np.divide(last - win.mean(axis=1), sd, out=np.zeros(m), where=sd > 0)
                cols.append(z)
        elif key == "sma":
            for w in val:
                cols.append(tail(c, w).mean(axis=1) / last - 1)
        elif key == "ema":
            for w in val:
                a = 2 / (w + 1)
                j = np.arange(queue - 1)  # lag of each in-queue term
                kern = np.empty(queue)
                kern[1:] = (a * (1 - a) ** j)[::-1]  # oldest .. newest
                kern[0] = (1 - a) ** (queue - 1)  # the seed c[queue start]
                cols.append(q @ kern / last - 1)
        elif key == "rsi":
            d = np.diff(c)
            up, down = np.maximum(d, 0.0), np.maximum(-d, 0.0)
            qu, qd = _last(up, queue - 1), _last(down, queue - 1)
            for w in val:
                beta = (w - 1) / w
                updates = queue - w
                kern = np.zeros(queue - 1)
                kern[:w] += beta**updates / w
                idx = np.arange(w - 1, queue - 1)
                kern[idx] += beta ** (queue - 2 - idx) / w
                rs = (qu @ kern) / (qd @ kern + 1e-9)
                rsi = 100 - 100 / (1.0 + rs)
                cols.append(2 * (rsi / 100) - 1)
        elif key == "sto":
            for w, ks, ds in zip(val["window"], val["k"], val["d"], strict=True):
                hh = _last(h, w).max(axis=1)
                ll = _last(lo, w).min(axis=1)
                rng = hh - ll
                kv = np.divide(c[w - 1 :] - ll, rng, out=np.full(rng.size, 0.5), where=rng > 0)
                ksm = _last(kv, ks).mean(axis=1)  # ends at rows w+ks-2 ..
                dsm = _last(ksm, ds).mean(axis=1)
                cols.append(ksm[ksm.size - m :] - 0.5)
                cols.append(dsm[dsm.size - m :] - 0.5)
        elif key == "cov":
            for w in val:
                win = tail(c, w)
                cols.append(win.std(axis=1) / win.mean(axis=1))
    out[rows] = np.column_stack(cols)
    return out, cov


def descriptors(
    panel: MarketPanel,
    params: Mapping[str, Any] = FEATURE_PARAMS,
    *,
    queue: int = QUEUE_SIZE,
    tz: str = "America/New_York",
) -> tuple[F64, F64]:
    """``(features[T, S, F], coef_of_var[T, S])`` at every bar; NaN where the
    security has no bar or its queue is not full. Row ``t`` uses bars
    ``<= t`` only."""
    if queue < _min_queue(params):
        raise ValueError(f"queue_size must be >= {_min_queue(params)} for these windows")
    zone = ZoneInfo(tz)
    t, s = panel.shape
    tri = panel.total_return_index()
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = tri / panel.close
    feats = np.full((t, s, feature_dim(params)), np.nan)
    cov = np.full((t, s), np.nan)
    for i in range(s):
        rows = np.flatnonzero(panel.tradable[:, i])
        if rows.size < queue:
            continue
        k = scale[rows, i]
        f, cv = _series_features(
            panel.open[rows, i] * k,
            panel.high[rows, i] * k,
            panel.low[rows, i] * k,
            tri[rows, i],
            panel.times[rows],
            params,
            queue,
            zone,
        )
        feats[rows, i] = f
        cov[rows, i] = cv
    return feats, cov


def reference_descriptors(
    ohlc: F64,
    end_ns: Sequence[int],
    params: Mapping[str, Any] = FEATURE_PARAMS,
    tz: str = "America/New_York",
) -> F64:
    """Literal port of the original ``Descriptors.compute`` on one queue
    ``ohlc[n, 4]`` (with this module's documented deviations), for tests."""
    zone = ZoneInfo(tz)
    o, h, lo, c = (np.asarray(ohlc[:, k], dtype=np.float64) for k in range(4))
    feats: list[float] = []
    for key, val in params.items():
        if key == "sto":
            for w, ks, ds in zip(val["window"], val["k"], val["d"], strict=True):
                kv = []
                for i in range(w - 1, c.size):
                    lw, hg = np.min(lo[i - w + 1 : i + 1]), np.max(h[i - w + 1 : i + 1])
                    kv.append((c[i] - lw) / (hg - lw) if hg > lw else 0.5)
                ksm = np.convolve(np.array(kv), np.ones(ks) / ks, mode="valid")
                dsm = np.convolve(ksm, np.ones(ds) / ds, mode="valid")
                feats += [ksm[-1] - 0.5, dsm[-1] - 0.5]
        elif key == "cdl":
            for w in val:
                prices = np.stack([o, h, lo, c]).T[-w:]
                cand = np.array([o[0], np.max(h), np.min(lo), c[-1]])
                feats += (cand / prices.mean() - 1).tolist()
        elif key in ("nts", "ntw"):
            d = dt.datetime.fromtimestamp(int(end_ns[-1]) / 1e9, zone)
            if key == "nts":
                feats.append(2 * (d.hour * 3600 + d.minute * 60 + d.second) / 86_400 - 1)
            else:
                feats.append(2 * d.weekday() / 6 - 1)
        else:
            for w in val:
                arr = c[-w:]
                if key == "sma":
                    feats.append(arr.mean() / c[-1] - 1)
                elif key == "ema":
                    a, e = 2 / (w + 1), c[0]
                    for p in c[1:]:
                        e = a * p + (1 - a) * e
                    feats.append(e / c[-1] - 1)
                elif key == "rsi":
                    deltas = np.diff(c)
                    seed = deltas[:w]
                    up = seed[seed >= 0].sum() / w
                    down = -seed[seed < 0].sum() / w
                    for i in range(w, c.size):
                        dlt = deltas[i - 1]
                        uv, dv = (dlt, 0.0) if dlt > 0 else (0.0, -dlt)
                        up = (up * (w - 1) + uv) / w
                        down = (down * (w - 1) + dv) / w
                    rsi = 100 - 100 / (1.0 + up / (down + 1e-9))
                    feats.append(2 * (rsi / 100) - 1)
                elif key == "zsc":
                    sd = arr.std()
                    feats.append((arr[-1] - arr.mean()) / sd if sd > 0 else 0.0)
                elif key == "nrm":
                    feats.append(arr[-1] / arr.mean() - 1)
                elif key == "grd":
                    feats.append((arr[-1] - arr[0]) / arr.mean())
                elif key == "cov":
                    feats.append(arr.std() / arr.mean())
    return np.asarray(feats)

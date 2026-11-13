"""Pythonic front-end to the Rust :mod:`automoonbot._moonrs` store.

The Rust layer is strict about types and layout (contiguous ``int64`` /
``float32`` arrays, nanosecond timestamps). This module converts common
inputs (``datetime64``, ``pandas.Timestamp``, ``datetime`` with tzinfo,
Python ints, lists) into that form once, and never guesses a timezone.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

from automoonbot import _moonrs

#: Open interval bounds ("since forever" / "until further notice").
OPEN_START: int = np.iinfo(np.int64).min
OPEN_END: int = np.iinfo(np.int64).max

TimeLike: TypeAlias = "int | np.integer[Any] | np.datetime64 | _dt.datetime | str"
TimesLike: TypeAlias = "TimeLike | Sequence[TimeLike] | npt.ArrayLike"


def to_ns(t: TimeLike) -> int:
    """Convert one instant to UTC nanoseconds since the epoch.

    ``np.datetime64`` values and integers are taken to be UTC already.
    Naive ``datetime`` objects and offset-less strings are rejected: a
    timestamp without a timezone is exactly the kind of ambiguity that
    produces look-ahead bias around market opens and closes.
    """
    if isinstance(t, bool):
        raise TypeError("bool is not a timestamp")
    if isinstance(t, np.timedelta64):
        raise TypeError("timedelta64 is a duration, not a timestamp")
    if isinstance(t, int | np.integer):
        return int(t)
    if isinstance(t, _dt.datetime):
        if t.tzinfo is None or t.tzinfo.utcoffset(t) is None:
            raise ValueError(f"naive datetime {t!r}: attach a timezone")
        if hasattr(t, "value"):  # pandas.Timestamp keeps exact nanoseconds
            return int(t.value)
        delta = t - _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
        return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000
    if isinstance(t, np.datetime64):
        if np.isnat(t):
            raise ValueError("NaT is not a timestamp")
        return int(t.astype("datetime64[ns]").astype(np.int64))
    if isinstance(t, str):
        # ISO 8601 with an explicit offset; "Z" is accepted on every version.
        s = t[:-1] + "+00:00" if t.endswith(("Z", "z")) else t
        return to_ns(_dt.datetime.fromisoformat(s))
    raise TypeError(f"cannot interpret {type(t).__name__} as a timestamp")


def _times(t: TimesLike, n: int, name: str) -> npt.NDArray[np.int64]:
    """Broadcast a scalar or convert a sequence of instants to ``int64[n]``."""
    if isinstance(t, np.ndarray):
        if np.issubdtype(t.dtype, np.datetime64):
            if np.isnat(t).any():
                raise ValueError(f"{name} contains NaT")
            arr = t.astype("datetime64[ns]").astype(np.int64)
        elif np.issubdtype(t.dtype, np.integer):
            arr = t.astype(np.int64, copy=False)
        else:
            raise TypeError(f"{name} must be int64 nanoseconds or datetime64, got {t.dtype}")
    elif isinstance(t, (list, tuple)):
        arr = np.fromiter((to_ns(x) for x in t), dtype=np.int64, count=len(t))
    elif hasattr(t, "to_numpy"):  # pandas Series / Index
        return _times(np.asarray(t.to_numpy()), n, name)
    else:
        arr = np.full(n, to_ns(t), dtype=np.int64)  # type: ignore[arg-type]
    arr = np.ascontiguousarray(arr, dtype=np.int64)
    if arr.ndim != 1 or arr.shape[0] != n:
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    return arr


def _keys(keys: str | Sequence[str] | npt.ArrayLike) -> list[str]:
    if isinstance(keys, str):
        return [keys]
    out = [str(k) for k in keys]  # type: ignore[union-attr]
    return out


def _matrix(values: npt.ArrayLike, n: int, name: str) -> npt.NDArray[np.float32]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1 and n == 1:
        arr = arr[None, :]
    if arr.ndim == 1 and n > 1:
        arr = arr[:, None]
    if arr.ndim != 2 or arr.shape[0] != n:
        raise ValueError(f"{name} must have shape ({n}, dim), got {arr.shape}")
    return np.ascontiguousarray(arr)


@dataclass(frozen=True)
class Channel:
    """A fixed-width ``float32`` feature channel of a node kind."""

    name: str
    dim: int
    nullable: bool = False
    bounds: tuple[float, float] | None = None

    def as_tuple(self) -> tuple[str, int, bool, tuple[float, float] | None]:
        return (self.name, self.dim, self.nullable, self.bounds)


Homogeneous = Literal["forward", "reverse"]


class TemporalGraph:
    """Bitemporal heterogeneous graph store.

    All write methods are atomic per call and idempotent: re-sending rows that
    were already recorded is a no-op, and a batch containing any invalid or
    conflicting row leaves the store unchanged.
    """

    __slots__ = ("_g",)

    def __init__(self, _inner: _moonrs.TemporalGraph | None = None) -> None:
        self._g = _inner if _inner is not None else _moonrs.TemporalGraph()

    # ------------------------------------------------------------- schema
    def define_entity(self, name: str, channels: Sequence[Channel] = ()) -> int:
        """Declare a persistent node kind (company, security, publisher)."""
        return self._g.define_kind(name, [c.as_tuple() for c in channels], None)

    def define_event(
        self,
        name: str,
        window: int | np.timedelta64 | _dt.timedelta,
        channels: Sequence[Channel] = (),
    ) -> int:
        """Declare a point-like node kind (article, filing) with a default
        trailing snapshot window."""
        return self._g.define_kind(name, [c.as_tuple() for c in channels], _duration_ns(window))

    def define_relation(
        self,
        name: str,
        src: str,
        dst: str,
        attr_dim: int = 0,
        *,
        nullable: bool = False,
        bounds: tuple[float, float] | None = None,
    ) -> int:
        return self._g.define_relation(name, src, dst, attr_dim, nullable, bounds)

    @property
    def schema(self) -> dict[str, Any]:
        return self._g.schema()

    # ------------------------------------------------------------- writes
    def upsert_nodes(
        self,
        kind: str,
        keys: str | Sequence[str],
        known_time: TimesLike,
        valid_from: TimesLike = OPEN_START,
        valid_to: TimesLike = OPEN_END,
    ) -> dict[str, int]:
        k = _keys(keys)
        n = len(k)
        return self._g.upsert_nodes(
            kind,
            k,
            _times(valid_from, n, "valid_from"),
            _times(valid_to, n, "valid_to"),
            _times(known_time, n, "known_time"),
        )

    def retract_nodes(
        self,
        kind: str,
        keys: str | Sequence[str],
        known_time: TimesLike,
        valid_from: TimesLike,
        valid_to: TimesLike = OPEN_END,
    ) -> dict[str, int]:
        k = _keys(keys)
        n = len(k)
        return self._g.retract_nodes(
            kind,
            k,
            _times(valid_from, n, "valid_from"),
            _times(valid_to, n, "valid_to"),
            _times(known_time, n, "known_time"),
        )

    def observe(
        self,
        kind: str,
        channel: str,
        keys: str | Sequence[str],
        valid_time: TimesLike,
        known_time: TimesLike,
        values: npt.ArrayLike,
    ) -> dict[str, int]:
        k = _keys(keys)
        n = len(k)
        return self._g.observe(
            kind,
            channel,
            k,
            _times(valid_time, n, "valid_time"),
            _times(known_time, n, "known_time"),
            _matrix(values, n, "values"),
        )

    def assert_edges(
        self,
        relation: str,
        src: str | Sequence[str],
        dst: str | Sequence[str],
        known_time: TimesLike,
        valid_from: TimesLike = OPEN_START,
        valid_to: TimesLike = OPEN_END,
        attrs: npt.ArrayLike | None = None,
    ) -> dict[str, int]:
        s, d = _keys(src), _keys(dst)
        n = len(s)
        if attrs is None:
            attrs = np.zeros((n, 0), dtype=np.float32)
        return self._g.assert_edges(
            relation,
            s,
            d,
            _times(valid_from, n, "valid_from"),
            _times(valid_to, n, "valid_to"),
            _times(known_time, n, "known_time"),
            _matrix(attrs, n, "attrs"),
        )

    def retract_edges(
        self,
        relation: str,
        src: str | Sequence[str],
        dst: str | Sequence[str],
        known_time: TimesLike,
        valid_from: TimesLike,
        valid_to: TimesLike = OPEN_END,
    ) -> dict[str, int]:
        s, d = _keys(src), _keys(dst)
        n = len(s)
        return self._g.retract_edges(
            relation,
            s,
            d,
            _times(valid_from, n, "valid_from"),
            _times(valid_to, n, "valid_to"),
            _times(known_time, n, "known_time"),
        )

    # ------------------------------------------------------------- reads
    def snapshot(
        self,
        known_at: TimeLike,
        valid_at: TimeLike | None = None,
        *,
        history: Mapping[str, int] | None = None,
        windows: Mapping[str, int | np.timedelta64 | _dt.timedelta] | None = None,
        homogeneous: Homogeneous | None = None,
    ) -> dict[str, Any]:
        """Materialize the graph as known at ``known_at``.

        ``history`` maps ``"kind.channel"`` to a history length ``K``; feature
        tensors are ``[N, K, D]``, chronological and right-aligned. ``windows``
        overrides event-kind windows. ``homogeneous`` adds a flattened,
        transformer-ready view (``"reverse"`` adds typed reverse edges).
        """
        return self._g.snapshot(
            to_ns(known_at),
            None if valid_at is None else to_ns(valid_at),
            dict(history) if history else None,
            {k: _duration_ns(v) for k, v in windows.items()} if windows else None,
            homogeneous,
        )

    def node_count(self, kind: str) -> int:
        return self._g.node_count(kind)

    def edge_key_count(self, relation: str) -> int:
        return self._g.edge_key_count(relation)

    def observation_count(self, kind: str, channel: str) -> int:
        return self._g.observation_count(kind, channel)

    def max_known_time(self) -> int | None:
        return self._g.max_known_time()

    # ------------------------------------------------------------- I/O
    def save(self, path: str | PathLike[str]) -> None:
        """Atomically write a checksummed store file."""
        self._g.save(path)

    @classmethod
    def load(cls, path: str | PathLike[str]) -> TemporalGraph:
        """Load and fully re-validate a store file."""
        return cls(_moonrs.TemporalGraph.load(path))

    def __repr__(self) -> str:
        return repr(self._g)


def _duration_ns(d: int | np.timedelta64 | _dt.timedelta) -> int:
    if isinstance(d, bool):
        raise TypeError("bool is not a duration")
    # np.timedelta64 subclasses np.signedinteger, so test it first.
    if isinstance(d, np.timedelta64):
        if np.isnat(d):
            raise ValueError("NaT is not a duration")
        return int(d.astype("timedelta64[ns]").astype(np.int64))
    if isinstance(d, int | np.integer):
        return int(d)
    if isinstance(d, _dt.timedelta):
        return (d.days * 86_400 + d.seconds) * 1_000_000_000 + d.microseconds * 1_000
    raise TypeError(f"cannot interpret {type(d).__name__} as a duration")

"""Append-only, point-in-time Parquet store for canonical datasets.

Layout::

    <root>/<dataset>/known_year=<YYYY>/part-<sha256-prefix>.parquet

Semantics mirror the Rust core:

* Rows are *versions* identified by ``dataset.key + (known_time,)``.
* Writing a row identical to an existing version is a no-op (idempotent
  replays); writing a *different* row with the same identity is rejected
  (corrections must carry a later ``known_time``).
* A write is atomic: either the whole batch becomes visible or none of it.
* Reads can be restricted to what was known at ``as_of``.

Partitioning by the year of ``known_time`` keeps conflict checks local: two
versions can only conflict if they share a ``known_time``, hence a partition.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from automoonbot.data.schemas import DATASETS, TS, Dataset, conform

_PART = "known_year"


class ConflictError(ValueError):
    """A different row already exists with the same key and known_time."""


@dataclass(frozen=True)
class WriteResult:
    new: int
    duplicate: int
    files: tuple[str, ...]


def _resolve(dataset: str | Dataset) -> Dataset:
    if isinstance(dataset, Dataset):
        return dataset
    try:
        return DATASETS[dataset]
    except KeyError:
        raise KeyError(f"unknown dataset {dataset!r}; known: {sorted(DATASETS)}") from None


def _null_safe(col: pa.ChunkedArray) -> pa.ChunkedArray:
    """Make nulls joinable by replacing them with an out-of-band sentinel
    of the same logical type (Arrow joins never match nulls)."""
    if not col.null_count:
        return col
    t = col.type
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return pc.fill_null(col, "\x00null")
    if pa.types.is_timestamp(t):
        return pc.fill_null(col.cast(pa.int64()), np.iinfo(np.int64).min)
    if pa.types.is_date32(t):
        return pc.fill_null(col.cast(pa.int32()), np.iinfo(np.int32).min)
    if pa.types.is_integer(t):
        return pc.fill_null(col.cast(pa.int64()), np.iinfo(np.int64).min)
    raise TypeError(f"unsupported key column type {t}")


def _join_keys(table: pa.Table, cols: Sequence[str]) -> pa.Table:
    out = {}
    for c in cols:
        col = table.column(c)
        if pa.types.is_date32(col.type):
            col = col.cast(pa.int32()).cast(pa.int64())
        elif pa.types.is_timestamp(col.type):
            col = col.cast(pa.int64())
        out[f"__k_{c}"] = _null_safe(col)
    return pa.table(out)


def _values_equal(a: pa.ChunkedArray, b: pa.ChunkedArray) -> np.ndarray:
    """Row-wise equality where null == null and NaN == NaN."""
    if pa.types.is_list(a.type):
        return np.array(
            [
                x == y or _list_nan_eq(x, y)
                for x, y in zip(a.to_pylist(), b.to_pylist(), strict=True)
            ],
            dtype=bool,
        )
    eq = pc.fill_null(pc.equal(a, b), False).to_numpy(zero_copy_only=False)
    both_null = (pc.is_null(a).to_numpy(zero_copy_only=False)) & (
        pc.is_null(b).to_numpy(zero_copy_only=False)
    )
    out = eq | both_null
    if pa.types.is_floating(a.type):
        nan_a = pc.fill_null(pc.is_nan(a), False).to_numpy(zero_copy_only=False)
        nan_b = pc.fill_null(pc.is_nan(b), False).to_numpy(zero_copy_only=False)
        out |= nan_a & nan_b
    return np.asarray(out, dtype=bool)


def _list_nan_eq(x: list[float] | None, y: list[float] | None) -> bool:
    if x is None or y is None or len(x) != len(y):
        return False
    return all(a == b or (a != a and b != b) for a, b in zip(x, y, strict=True))


class ParquetStore:
    """See module docstring."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _writer_lock(self) -> Iterator[None]:
        """Serialize writers across processes (conflict checks read-then-write)."""
        if os.name != "posix":
            yield
            return
        import fcntl

        with open(self.root / ".write.lock", "a+b") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    # ----------------------------------------------------------------- write
    def write(self, dataset: str | Dataset, table: pa.Table) -> WriteResult:
        with self._writer_lock():
            return self._write(dataset, table)

    def _write(self, dataset: str | Dataset, table: pa.Table) -> WriteResult:
        d = _resolve(dataset)
        table = conform(d, table)
        if table.num_rows == 0:
            return WriteResult(0, 0, ())
        identity = [*d.key, "known_time"]

        n_in = table.num_rows
        # Reject conflicts inside the batch itself and collapse exact repeats.
        table = self._dedupe_batch(d, table, identity)

        years = pc.year(table.column("known_time")).to_numpy(zero_copy_only=False)
        written: list[str] = []
        staged: list[tuple[Path, Path]] = []
        new_total = 0
        try:
            for year in np.unique(years):
                part = table.filter(pa.array(years == year))
                existing = self._read_partition(d, int(year), columns=d.schema.names)
                part = self._drop_existing(d, part, existing, identity)
                if part.num_rows == 0:
                    continue
                part = part.sort_by([(c, "ascending") for c in identity])
                buf = io.BytesIO()
                pq.write_table(part, buf, compression="zstd", write_statistics=True)
                data = buf.getvalue()
                digest = hashlib.sha256(data).hexdigest()[:32]
                target = self.root / d.name / f"{_PART}={int(year)}" / f"part-{digest}.parquet"
                target.parent.mkdir(parents=True, exist_ok=True)
                fd, tmp_name = tempfile.mkstemp(
                    dir=target.parent, prefix=".tmp-", suffix=".parquet"
                )
                with os.fdopen(fd, "wb") as fh:
                    fh.write(data)
                    fh.flush()
                    os.fsync(fh.fileno())
                staged.append((Path(tmp_name), target))
                new_total += part.num_rows
            # All partitions validated and staged: publish.
            for tmp, target in staged:
                os.replace(tmp, target)
                written.append(str(target.relative_to(self.root)))
            for parent in {t.parent for _, t in staged}:
                _fsync_dir(parent)
        finally:
            for tmp, _ in staged:
                if tmp.exists():
                    tmp.unlink()
        return WriteResult(new_total, n_in - new_total, tuple(written))

    @staticmethod
    def _dedupe_batch(d: Dataset, table: pa.Table, identity: list[str]) -> pa.Table:
        keys = _join_keys(table, identity)
        idx = pa.array(np.arange(table.num_rows, dtype=np.int64))
        keyed = keys.append_column("__row", idx)
        first = keyed.group_by(keys.column_names, use_threads=False).aggregate(
            [("__row", "min"), ("__row", "count")]
        )
        if pc.max(first.column("__row_count")).as_py() == 1:
            return table
        # Map every row to its group's first row and compare values.
        joined = keyed.join(first, keys.column_names, join_type="inner", use_threads=False)
        rows = joined.column("__row").to_numpy()
        firsts = joined.column("__row_min").to_numpy()
        values = [c for c in d.schema.names if c not in identity]
        a = table.take(pa.array(rows))
        b = table.take(pa.array(firsts))
        for c in values:
            same = _values_equal(a.column(c), b.column(c))
            if not same.all():
                bad = int(rows[~same][0])
                raise ConflictError(
                    f"{d.name}: batch contains different rows for the same "
                    f"{identity} (row {bad}, column {c!r})"
                )
        keep = np.sort(np.unique(firsts))
        return table.take(pa.array(keep))

    @staticmethod
    def _drop_existing(
        d: Dataset, part: pa.Table, existing: pa.Table | None, identity: list[str]
    ) -> pa.Table:
        if existing is None or existing.num_rows == 0:
            return part
        lk = _join_keys(part, identity).append_column(
            "__l", pa.array(np.arange(part.num_rows, dtype=np.int64))
        )
        rk = _join_keys(existing, identity).append_column(
            "__r", pa.array(np.arange(existing.num_rows, dtype=np.int64))
        )
        names = [c for c in lk.column_names if c.startswith("__k_")]
        m = lk.join(rk, names, join_type="inner", use_threads=False)
        if m.num_rows == 0:
            return part
        li, ri = m.column("__l").to_numpy(), m.column("__r").to_numpy()
        a, b = part.take(pa.array(li)), existing.take(pa.array(ri))
        for c in d.schema.names:
            if c in identity:
                continue
            same = _values_equal(a.column(c), b.column(c))
            if not same.all():
                raise ConflictError(
                    f"{d.name}: a different version with the same {identity} already exists "
                    f"(column {c!r}); record corrections with a later known_time"
                )
        mask = np.ones(part.num_rows, dtype=bool)
        mask[li] = False
        return part.filter(pa.array(mask))

    # ------------------------------------------------------------------ read
    def _dataset(self, d: Dataset) -> ds.Dataset | None:
        path = self.root / d.name
        if not path.exists() or not any(path.rglob("part-*.parquet")):
            return None
        return ds.dataset(
            path,
            schema=d.schema,
            format="parquet",
            partitioning=ds.partitioning(pa.schema([(_PART, pa.int32())]), flavor="hive"),
            exclude_invalid_files=False,
            ignore_prefixes=[".tmp-", "."],
        )

    def _read_partition(self, d: Dataset, year: int, columns: list[str]) -> pa.Table | None:
        path = self.root / d.name / f"{_PART}={year}"
        files = sorted(path.glob("part-*.parquet")) if path.exists() else []
        if not files:
            return None
        return ds.dataset([str(f) for f in files], schema=d.schema, format="parquet").to_table(
            columns=columns
        )

    def read(
        self,
        dataset: str | Dataset,
        as_of: int | None = None,
        *,
        columns: Sequence[str] | None = None,
        filter: pc.Expression | None = None,
        latest: bool = False,
    ) -> pa.Table:
        """Rows known by ``as_of`` (UTC ns), sorted by key and ``known_time``.

        With ``latest=True`` only the most recently known version of each key
        is returned - what a participant would have believed at ``as_of``.
        """
        d = _resolve(dataset)
        data = self._dataset(d)
        if data is None:
            return (
                d.schema.empty_table()
                if columns is None
                else d.schema.empty_table().select(list(columns))
            )
        expr = filter
        if as_of is not None:
            cut = pc.field("known_time") <= pa.scalar(as_of, type=pa.int64()).cast(TS)
            expr = cut if expr is None else expr & cut
        table = data.to_table(filter=expr)
        identity = [*d.key, "known_time"]
        table = table.sort_by([(c, "ascending") for c in identity])
        if latest and table.num_rows > 1:
            # Sorted by key then known_time: a row is the latest version iff
            # its whole key differs from the next row's.
            keys = _join_keys(table, list(d.key))
            same = np.ones(table.num_rows - 1, dtype=bool)
            for c in keys.column_names:
                v = keys.column(c).to_numpy()
                same &= v[:-1] == v[1:]
            last = np.ones(table.num_rows, dtype=bool)
            last[:-1] = ~same
            table = table.filter(pa.array(last))
        if columns is not None:
            table = table.select(list(columns))
        return table

    def compact(self, dataset: str | Dataset) -> int:
        """Merge each partition's parts into one file. Returns files removed."""
        with self._writer_lock():
            return self._compact(dataset)

    def _compact(self, dataset: str | Dataset) -> int:
        d = _resolve(dataset)
        base = self.root / d.name
        removed = 0
        if not base.exists():
            return 0
        for part_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            files = sorted(part_dir.glob("part-*.parquet"))
            if len(files) <= 1:
                continue
            table = ds.dataset(
                [str(f) for f in files], schema=d.schema, format="parquet"
            ).to_table()
            table = table.sort_by([(c, "ascending") for c in [*d.key, "known_time"]])
            buf = io.BytesIO()
            pq.write_table(table, buf, compression="zstd", write_statistics=True)
            data = buf.getvalue()
            target = part_dir / f"part-{hashlib.sha256(data).hexdigest()[:32]}.parquet"
            fd, tmp_name = tempfile.mkstemp(dir=part_dir, prefix=".tmp-", suffix=".parquet")
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, target)
            _fsync_dir(part_dir)
            for old in files:
                if old != target:
                    old.unlink()
                    removed += 1
            _fsync_dir(part_dir)
        return removed


def _fsync_dir(path: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

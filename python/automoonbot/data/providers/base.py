"""Provider abstraction.

A provider turns some external source into canonical tables
(:mod:`automoonbot.data.schemas`). The only hard requirement beyond the
schema is honesty about time: :class:`KnowledgePolicy` states how a provider
derives ``known_time``, and downstream evaluation can refuse providers whose
policy is not point-in-time safe.
"""

import abc
import enum
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

import pyarrow as pa

from automoonbot.data.schemas import DATASETS, conform
from automoonbot.data.store import ParquetStore, WriteResult


class KnowledgePolicy(enum.Enum):
    #: ``known_time`` is the source's own first-publication timestamp.
    EXACT = "exact"
    #: ``known_time`` is a guaranteed upper bound on first availability
    #: (e.g. end of the business day when only a date is published).
    CONSERVATIVE = "conservative"
    #: ``known_time`` is a heuristic (vendor history without publication
    #: times). Usable for prototyping, never for evaluation.
    ESTIMATED = "estimated"

    @property
    def point_in_time_safe(self) -> bool:
        return self is not KnowledgePolicy.ESTIMATED


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    datasets: frozenset[str]
    policy: KnowledgePolicy
    description: str

    def __post_init__(self) -> None:
        unknown = self.datasets - DATASETS.keys()
        if unknown:
            raise ValueError(f"{self.name}: unknown datasets {sorted(unknown)}")


class Provider(abc.ABC):
    """Base class for data providers."""

    info: ProviderInfo

    @abc.abstractmethod
    def fetch(self, dataset: str, entities: Sequence[str]) -> Iterator[pa.Table]:
        """Yield canonical batches of ``dataset`` for ``entities``."""

    def ingest(self, store: ParquetStore, dataset: str, entities: Sequence[str]) -> WriteResult:
        """Fetch and write ``dataset`` for ``entities`` into ``store``.

        Each yielded batch is validated and written atomically; a failure
        leaves earlier batches in place (they are idempotent to replay).
        """
        if dataset not in self.info.datasets:
            raise ValueError(f"{self.info.name} does not provide {dataset!r}")
        new = dup = 0
        files: list[str] = []
        for batch in self.fetch(dataset, entities):
            batch = conform(DATASETS[dataset], batch)
            sources = set(batch.column("source").to_pylist())
            if sources - {self.info.name}:
                raise ValueError(f"{self.info.name} produced rows tagged {sorted(sources)}")
            r = store.write(dataset, batch)
            new, dup = new + r.new, dup + r.duplicate
            files.extend(r.files)
        return WriteResult(new, dup, tuple(files))

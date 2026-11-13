from os import PathLike
from typing import Any

import numpy as np
import numpy.typing as npt

FORMAT_VERSION: int
MISSING_TIME: int
__version__: str

_I64 = npt.NDArray[np.int64]
_F32 = npt.NDArray[np.float32]

class TemporalGraph:
    def __init__(self) -> None: ...
    def define_kind(
        self,
        name: str,
        channels: list[tuple[str, int, bool, tuple[float, float] | None]],
        window_ns: int | None = None,
    ) -> int: ...
    def define_relation(
        self,
        name: str,
        src: str,
        dst: str,
        attr_dim: int,
        nullable: bool = False,
        bounds: tuple[float, float] | None = None,
    ) -> int: ...
    def schema(self) -> dict[str, Any]: ...
    def upsert_nodes(
        self, kind: str, keys: list[str], valid_from: _I64, valid_to: _I64, known_time: _I64
    ) -> dict[str, int]: ...
    def retract_nodes(
        self, kind: str, keys: list[str], valid_from: _I64, valid_to: _I64, known_time: _I64
    ) -> dict[str, int]: ...
    def observe(
        self,
        kind: str,
        channel: str,
        keys: list[str],
        valid_time: _I64,
        known_time: _I64,
        values: _F32,
    ) -> dict[str, int]: ...
    def assert_edges(
        self,
        relation: str,
        src: list[str],
        dst: list[str],
        valid_from: _I64,
        valid_to: _I64,
        known_time: _I64,
        attrs: _F32,
    ) -> dict[str, int]: ...
    def retract_edges(
        self,
        relation: str,
        src: list[str],
        dst: list[str],
        valid_from: _I64,
        valid_to: _I64,
        known_time: _I64,
    ) -> dict[str, int]: ...
    def snapshot(
        self,
        known_at: int,
        valid_at: int | None = None,
        history: dict[str, int] | None = None,
        windows: dict[str, int] | None = None,
        homogeneous: str | None = None,
    ) -> dict[str, Any]: ...
    def node_count(self, kind: str) -> int: ...
    def edge_key_count(self, relation: str) -> int: ...
    def observation_count(self, kind: str, channel: str) -> int: ...
    def max_known_time(self) -> int | None: ...
    def save(self, path: str | PathLike[str]) -> None: ...
    @staticmethod
    def load(path: str | PathLike[str]) -> TemporalGraph: ...

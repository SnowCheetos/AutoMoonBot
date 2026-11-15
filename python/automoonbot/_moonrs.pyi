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

_B = npt.NDArray[np.bool_]
_F64 = npt.NDArray[np.float64]

class Market:
    def __init__(
        self,
        open: _F64,
        close: _F64,
        adv_shares: _F64,
        sigma: _F64,
        tradable: _B,
        listed: _B,
        split: _F64,
        dividend: _F64,
        *,
        instruments: list[dict[str, Any]] | None = None,
        half_spread: _F64 | None = None,
        rate: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """``instruments``: one dict per column (``kind`` is ``"equity"``,
        ``"index"``, ``"option"`` or ``"future"``); unknown keys are
        rejected. ``half_spread``: absolute per-bar half-spread, NaN for the
        bps default. ``rate``: ``[T]`` annual risk-free rate known at each
        close (cash accrues at it plus ``cash_spread``/``margin_spread``)."""
    @property
    def shape(self) -> tuple[int, int]: ...

class Simulator:
    def __init__(
        self,
        market: Market,
        start: int = 0,
        capital: float = 1.0,
        *,
        commission_bps: float | None = None,
        half_spread_bps: float | None = None,
        impact_coef: float | None = None,
        max_participation: float | None = None,
        borrow_rate: float | None = None,
        cash_rate: float | None = None,
        margin_rate: float | None = None,
        cash_spread: float | None = None,
        margin_spread: float | None = None,
        delist_haircut: float | None = None,
        periods_per_year: float | None = None,
        option_commission: float | None = None,
        future_commission: float | None = None,
        option_half_spread_bps: float | None = None,
        option_min_half_spread: float | None = None,
        long_only: bool = True,
        max_gross: float = 1.0,
        max_weight: float = 1.0,
        margin: str = "none",
        margin_params: dict[str, float] | None = None,
        early_exercise: bool = True,
        exercise_threshold: float | None = None,
        rate: float | None = None,
    ) -> None:
        """``margin``: ``"none"``, ``"reg_t"`` or ``"portfolio"``, with
        ``margin_params`` overriding that model's defaults."""
    @property
    def period(self) -> int: ...
    @property
    def done(self) -> bool: ...
    @property
    def equity(self) -> float: ...
    @property
    def cash(self) -> float: ...
    def shares(self) -> _F64: ...
    def weights(self) -> _F64: ...
    def option_terms(self, i: int) -> tuple[float, float] | None:
        """``(strike, deliverable)`` of an option column after split
        adjustments; ``None`` for other columns."""
    def step(self, target: _F64, units: bool = False) -> dict[str, float]:
        """Execute target weights (or, with ``units``, shares / contracts)
        at the next open. ``NaN`` holds a security's current position (no
        trade); infinities are rejected."""
    def run(self, targets: _F64, units: bool = False) -> dict[str, _F64]:
        """``step`` for each row of ``targets[N, S]``; stops on ruin."""

def option_value(
    call: npt.NDArray[np.bool_],
    spot: npt.NDArray[np.float64],
    strike: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
    rate: npt.NDArray[np.float64],
    carry: npt.NDArray[np.float64],
    vol: npt.NDArray[np.float64],
    *,
    american: bool = True,
) -> npt.NDArray[np.float64]: ...

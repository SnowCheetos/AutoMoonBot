"""AutoMoonBot: a point-in-time graph research platform for systematic trading.

The core is :class:`TemporalGraph`, a bitemporal heterogeneous graph store
implemented in Rust. Every fact carries a *valid time* (when it is true in the
world) and a *known time* (when it could first have been known); snapshots
taken "as of" a knowledge time can only see facts known by then.
"""

from automoonbot._moonrs import FORMAT_VERSION, MISSING_TIME
from automoonbot._moonrs import __version__ as core_version
from automoonbot.graph import (
    OPEN_END,
    OPEN_START,
    Channel,
    TemporalGraph,
    to_ns,
)

__all__ = [
    "FORMAT_VERSION",
    "MISSING_TIME",
    "OPEN_END",
    "OPEN_START",
    "Channel",
    "TemporalGraph",
    "core_version",
    "to_ns",
]

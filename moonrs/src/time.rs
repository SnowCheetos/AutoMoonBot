//! Time primitives.
//!
//! All times are UTC nanoseconds since the Unix epoch stored as `i64`, which
//! covers years 1677..2262. `std::time::Instant` (used by the previous version
//! of this crate) is monotonic process-local time: it cannot be persisted,
//! compared across runs, or mapped to a market calendar, so it made
//! point-in-time reconstruction impossible.
//!
//! Every fact in the store carries two times (a *bitemporal* model):
//!
//! * **valid time** – when the fact is true in the world (the bar's close, the
//!   filing's period, the relationship's effective dates);
//! * **known time** – the earliest moment the fact could have been known by a
//!   trader (the filing's acceptance timestamp, the vendor's publication time
//!   plus latency).
//!
//! A snapshot taken "as of" `T` may only use facts with `known_time <= T`.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A UTC instant with nanosecond resolution.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
pub struct Timestamp(i64);

impl Timestamp {
    /// Sentinel for "since forever" – only valid as an interval start.
    pub const NEG_INF: Timestamp = Timestamp(i64::MIN);
    /// Sentinel for "until further notice" – only valid as an interval end.
    pub const POS_INF: Timestamp = Timestamp(i64::MAX);

    /// A concrete (non-sentinel) instant.
    pub fn new(nanos: i64) -> Result<Self> {
        if nanos == i64::MIN || nanos == i64::MAX {
            Err(Error::ReservedTimestamp(nanos))
        } else {
            Ok(Timestamp(nanos))
        }
    }

    /// Construct without the sentinel check. Used for interval bounds, where
    /// sentinels are meaningful.
    pub const fn from_nanos_unchecked(nanos: i64) -> Self {
        Timestamp(nanos)
    }

    #[inline]
    pub const fn nanos(self) -> i64 {
        self.0
    }

    #[inline]
    pub fn saturating_sub(self, nanos: i64) -> Self {
        Timestamp(self.0.saturating_sub(nanos))
    }
}

/// A half-open validity interval `[start, end)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Interval {
    start: Timestamp,
    end: Timestamp,
}

impl Interval {
    /// `start` may be [`Timestamp::NEG_INF`] and `end` may be
    /// [`Timestamp::POS_INF`]; otherwise both must be concrete and
    /// `start < end`.
    pub fn new(start: i64, end: i64) -> Result<Self> {
        if start >= end || start == i64::MAX || end == i64::MIN {
            return Err(Error::InvalidInterval { start, end });
        }
        Ok(Interval {
            start: Timestamp(start),
            end: Timestamp(end),
        })
    }

    pub const ALWAYS: Interval = Interval {
        start: Timestamp::NEG_INF,
        end: Timestamp::POS_INF,
    };

    #[inline]
    pub fn start(&self) -> Timestamp {
        self.start
    }

    #[inline]
    pub fn end(&self) -> Timestamp {
        self.end
    }

    #[inline]
    pub fn contains(&self, t: Timestamp) -> bool {
        self.start <= t && t < self.end
    }

    #[inline]
    pub fn overlaps(&self, other: &Interval) -> bool {
        self.start < other.end && other.start < self.end
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentinels_rejected_as_instants() {
        assert!(Timestamp::new(i64::MIN).is_err());
        assert!(Timestamp::new(i64::MAX).is_err());
        assert_eq!(Timestamp::new(0).unwrap().nanos(), 0);
    }

    #[test]
    fn interval_semantics() {
        let i = Interval::new(10, 20).unwrap();
        assert!(!i.contains(Timestamp(9)));
        assert!(i.contains(Timestamp(10)));
        assert!(i.contains(Timestamp(19)));
        assert!(!i.contains(Timestamp(20)));
        assert!(Interval::new(5, 5).is_err());
        assert!(Interval::new(6, 5).is_err());
        assert!(Interval::new(i64::MAX, i64::MAX).is_err());
        assert!(Interval::ALWAYS.contains(Timestamp(0)));
        assert!(i.overlaps(&Interval::new(19, 30).unwrap()));
        assert!(!i.overlaps(&Interval::new(20, 30).unwrap()));
    }
}

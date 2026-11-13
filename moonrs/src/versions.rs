//! Bitemporal interval facts.
//!
//! A [`VersionList`] holds every recorded belief about one key (a node's
//! existence, or one `(src, relation, dst)` edge). Each version says "over
//! valid interval `[start, end)` the state is `payload`" (or, for a
//! retraction, "does not exist"), and was recorded at `known` time.
//!
//! Resolution at `(valid_at, known_at)` returns the state asserted by the
//! most recently *known* version (with `known <= known_at`) whose interval
//! contains `valid_at`. Corrections are therefore expressed by recording a new
//! version with a later knowledge time, and history before that knowledge time
//! is reconstructed exactly as it was believed back then.
//!
//! Two versions with the same knowledge time and overlapping intervals must
//! agree; otherwise the answer would depend on ingestion order, so such a
//! write is rejected.

use serde::{Deserialize, Serialize};
use smallvec::SmallVec;

use crate::error::{Error, Result};
use crate::time::{Interval, Timestamp};

/// Payload compared bit-for-bit (so `NaN == NaN` for nullable attributes and
/// `0.0 != -0.0`), which is what idempotent re-ingestion needs.
pub trait BitEq {
    fn bit_eq(&self, other: &Self) -> bool;
}

impl BitEq for () {
    fn bit_eq(&self, _: &Self) -> bool {
        true
    }
}

impl BitEq for Box<[f32]> {
    fn bit_eq(&self, other: &Self) -> bool {
        self.len() == other.len()
            && self
                .iter()
                .zip(other.iter())
                .all(|(a, b)| a.to_bits() == b.to_bits())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Version<P> {
    pub known: Timestamp,
    pub valid: Interval,
    /// `None` is a retraction over `valid`.
    pub payload: Option<P>,
}

impl<P: BitEq> Version<P> {
    fn same_assertion(&self, other: &Self) -> bool {
        match (&self.payload, &other.payload) {
            (None, None) => true,
            (Some(a), Some(b)) => a.bit_eq(b),
            _ => false,
        }
    }
}

/// Outcome of recording a version.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Recorded {
    New,
    Duplicate,
}

/// Versions sorted by `(known, valid.start, valid.end)`. Most keys have a
/// single version, which stays inline.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(serialize = "P: Serialize", deserialize = "P: Deserialize<'de>"))]
pub struct VersionList<P> {
    versions: SmallVec<[Version<P>; 1]>,
}

impl<P> Default for VersionList<P> {
    fn default() -> Self {
        VersionList {
            versions: SmallVec::new(),
        }
    }
}

impl<P: BitEq> VersionList<P> {
    pub fn record(
        &mut self,
        version: Version<P>,
        context: impl Fn() -> String,
    ) -> Result<Recorded> {
        let key = (version.known, version.valid.start(), version.valid.end());
        // Every version with the same knowledge time is a contiguous run;
        // check it for conflicts.
        let lo = self.versions.partition_point(|v| v.known < version.known);
        let hi = self.versions.partition_point(|v| v.known <= version.known);
        for existing in &self.versions[lo..hi] {
            if existing.valid == version.valid && existing.same_assertion(&version) {
                return Ok(Recorded::Duplicate);
            }
            if existing.valid.overlaps(&version.valid) && !existing.same_assertion(&version) {
                return Err(Error::Conflict { context: context() });
            }
        }
        let at = lo
            + self.versions[lo..hi]
                .partition_point(|v| (v.known, v.valid.start(), v.valid.end()) < key);
        self.versions.insert(at, version);
        Ok(Recorded::New)
    }

    /// Remove the version recorded with exactly this `(known, valid)`; used to
    /// roll back a failed batch. `(known, valid)` is unique by construction.
    pub(crate) fn remove(&mut self, known: Timestamp, valid: Interval) {
        if let Some(at) = self
            .versions
            .iter()
            .position(|v| v.known == known && v.valid == valid)
        {
            self.versions.remove(at);
        }
    }

    /// The state believed at `known_at` for valid time `valid_at`.
    ///
    /// Returns the deciding version (if it asserts existence), so callers can
    /// expose its interval and knowledge time.
    #[inline]
    pub fn resolve(&self, valid_at: Timestamp, known_at: Timestamp) -> Option<&Version<P>> {
        let end = self.versions.partition_point(|v| v.known <= known_at);
        self.versions[..end]
            .iter()
            .rev()
            .find(|v| v.valid.contains(valid_at))
            .filter(|v| v.payload.is_some())
    }

    pub fn versions(&self) -> &[Version<P>] {
        &self.versions
    }

    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }

    /// Check ordering and conflict invariants (used when loading files).
    pub(crate) fn check(&self) -> bool {
        self.versions.windows(2).all(|w| {
            (w[0].known, w[0].valid.start(), w[0].valid.end())
                <= (w[1].known, w[1].valid.start(), w[1].valid.end())
        }) && self.versions.iter().enumerate().all(|(i, a)| {
            self.versions[i + 1..]
                .iter()
                .take_while(|b| b.known == a.known)
                .all(|b| !a.valid.overlaps(&b.valid) || a.same_assertion(b))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(n: i64) -> Timestamp {
        Timestamp::new(n).unwrap()
    }

    fn v(known: i64, start: i64, end: i64, payload: Option<f32>) -> Version<Box<[f32]>> {
        Version {
            known: t(known),
            valid: Interval::new(start, end).unwrap(),
            payload: payload.map(|p| vec![p].into_boxed_slice()),
        }
    }

    fn val(list: &VersionList<Box<[f32]>>, valid: i64, known: i64) -> Option<f32> {
        list.resolve(t(valid), t(known))
            .map(|v| v.payload.as_ref().unwrap()[0])
    }

    #[test]
    fn corrections_apply_only_after_they_are_known() {
        let mut l = VersionList::default();
        let ctx = || String::new();
        l.record(v(10, 0, 100, Some(1.0)), ctx).unwrap();
        // At knowledge time 50 we learn the relationship ended at 40.
        l.record(v(50, 40, i64::MAX, None), ctx).unwrap();
        assert_eq!(val(&l, 45, 9), None, "not known yet");
        assert_eq!(val(&l, 45, 10), Some(1.0));
        assert_eq!(val(&l, 45, 49), Some(1.0), "retraction not yet known");
        assert_eq!(val(&l, 45, 50), None, "retraction known");
        assert_eq!(val(&l, 39, 50), Some(1.0), "earlier valid time unaffected");
        assert_eq!(val(&l, 100, 60), None, "outside original interval");
    }

    #[test]
    fn same_known_time_conflicts_are_rejected() {
        let mut l = VersionList::default();
        let ctx = || String::new();
        assert_eq!(
            l.record(v(10, 0, 100, Some(1.0)), ctx).unwrap(),
            Recorded::New
        );
        assert_eq!(
            l.record(v(10, 0, 100, Some(1.0)), ctx).unwrap(),
            Recorded::Duplicate
        );
        assert!(l.record(v(10, 50, 150, Some(2.0)), ctx).is_err());
        assert!(l.record(v(10, 50, 150, None), ctx).is_err());
        // Non-overlapping at the same knowledge time is fine.
        assert_eq!(
            l.record(v(10, 100, 150, Some(2.0)), ctx).unwrap(),
            Recorded::New
        );
        // Overlapping but agreeing is fine.
        assert_eq!(
            l.record(v(10, 20, 30, Some(1.0)), ctx).unwrap(),
            Recorded::New
        );
        assert!(l.check());
        assert_eq!(val(&l, 120, 10), Some(2.0));
    }

    #[test]
    fn nan_payloads_are_idempotent() {
        let mut l = VersionList::default();
        let ctx = || String::new();
        l.record(v(1, 0, 10, Some(f32::NAN)), ctx).unwrap();
        assert_eq!(
            l.record(v(1, 0, 10, Some(f32::NAN)), ctx).unwrap(),
            Recorded::Duplicate
        );
    }
}

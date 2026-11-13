//! Bitemporal observation series for one `(node, channel)` pair.
//!
//! Storage is columnar (`valid`, `known`, flat `values`) and sorted by
//! `(valid, known)`. Several rows may share a valid time: they are revisions
//! (a restated quarter, a corrected bar). A query "as of" knowledge time `T`
//! sees, for every valid time, only the latest revision known by `T`.
//!
//! Point lookups and K-step history are answered by scanning backwards for
//! the last row with `known <= T`. A per-block minimum of `known` lets the
//! scan skip whole blocks of not-yet-known rows, so even a series whose
//! recent rows were all back-filled late costs `O(n / BLOCK + BLOCK)` rather
//! than `O(n)` per lookup.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::time::Timestamp;
use crate::versions::Recorded;

const BLOCK: usize = 64;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Series {
    valid: Vec<Timestamp>,
    known: Vec<Timestamp>,
    values: Vec<f32>,
    #[serde(skip)]
    block_min: Vec<Timestamp>,
}

impl Series {
    pub fn len(&self) -> usize {
        self.valid.len()
    }

    pub fn is_empty(&self) -> bool {
        self.valid.is_empty()
    }

    #[inline]
    pub fn valid_time(&self, row: usize) -> Timestamp {
        self.valid[row]
    }

    #[inline]
    pub fn known_time(&self, row: usize) -> Timestamp {
        self.known[row]
    }

    #[inline]
    pub fn row(&self, row: usize, dim: usize) -> &[f32] {
        &self.values[row * dim..(row + 1) * dim]
    }

    /// Insert one observation. `values.len()` must equal the channel dim and
    /// be validated by the caller.
    pub fn record(
        &mut self,
        valid: Timestamp,
        known: Timestamp,
        values: &[f32],
        context: impl Fn() -> String,
    ) -> Result<Recorded> {
        let dim = values.len();
        let key = (valid, known);
        let n = self.len();
        // Fast path: in-order append (the overwhelmingly common case).
        let at = if n == 0 || (self.valid[n - 1], self.known[n - 1]) < key {
            n
        } else {
            let lo = self.valid.partition_point(|&v| v < valid);
            let hi = lo + self.valid[lo..].partition_point(|&v| v <= valid);
            let at = lo + self.known[lo..hi].partition_point(|&k| k < known);
            if at < n && (self.valid[at], self.known[at]) == key {
                let existing = self.row(at, dim);
                return if existing
                    .iter()
                    .zip(values)
                    .all(|(a, b)| a.to_bits() == b.to_bits())
                {
                    Ok(Recorded::Duplicate)
                } else {
                    Err(Error::Conflict { context: context() })
                };
            }
            at
        };
        if at == n {
            self.valid.push(valid);
            self.known.push(known);
            self.values.extend_from_slice(values);
            if n.is_multiple_of(BLOCK) {
                self.block_min.push(known);
            } else {
                let b = self.block_min.last_mut().expect("block exists");
                *b = (*b).min(known);
            }
        } else {
            self.valid.insert(at, valid);
            self.known.insert(at, known);
            self.values
                .splice(at * dim..at * dim, values.iter().copied());
            self.rebuild_blocks_from(at / BLOCK);
        }
        Ok(Recorded::New)
    }

    /// Remove the row recorded with exactly `(valid, known)`; used to roll
    /// back a failed batch.
    pub(crate) fn remove(&mut self, valid: Timestamp, known: Timestamp, dim: usize) {
        let lo = self.valid.partition_point(|&v| v < valid);
        let hi = lo + self.valid[lo..].partition_point(|&v| v <= valid);
        let at = lo + self.known[lo..hi].partition_point(|&k| k < known);
        if at < hi && self.known[at] == known {
            self.valid.remove(at);
            self.known.remove(at);
            self.values.drain(at * dim..(at + 1) * dim);
            self.rebuild_blocks_from(at / BLOCK);
        }
    }

    fn rebuild_blocks_from(&mut self, first_block: usize) {
        self.block_min.truncate(first_block);
        for chunk in self.known[first_block * BLOCK..].chunks(BLOCK) {
            self.block_min
                .push(*chunk.iter().min().expect("chunks are non-empty"));
        }
    }

    /// Recompute derived indexes after deserialization and verify ordering.
    pub(crate) fn finish_load(&mut self, dim: usize) -> bool {
        let n = self.valid.len();
        if self.known.len() != n || self.values.len() != n * dim {
            return false;
        }
        let sorted =
            (1..n).all(|i| (self.valid[i - 1], self.known[i - 1]) < (self.valid[i], self.known[i]));
        self.rebuild_blocks_from(0);
        sorted
    }

    /// Largest row index `< end` with `known <= known_at`.
    #[inline]
    fn last_known_before(&self, end: usize, known_at: Timestamp) -> Option<usize> {
        let mut i = end;
        while i > 0 {
            let block = (i - 1) / BLOCK;
            let block_start = block * BLOCK;
            if self.block_min[block] > known_at {
                i = block_start;
                continue;
            }
            while i > block_start {
                i -= 1;
                if self.known[i] <= known_at {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Row of the latest observation with `valid <= valid_at` as known at
    /// `known_at`.
    #[inline]
    pub fn latest(&self, valid_at: Timestamp, known_at: Timestamp) -> Option<usize> {
        let end = self.valid.partition_point(|&v| v <= valid_at);
        // Rows are sorted by (valid, known), so the last row with
        // `known <= known_at` is the latest revision of the latest valid time
        // that has any known revision.
        self.last_known_before(end, known_at)
    }

    /// Up to `k` rows (most recent first) for distinct valid times
    /// `<= valid_at`, each the latest revision known at `known_at`.
    pub fn history(
        &self,
        valid_at: Timestamp,
        known_at: Timestamp,
        k: usize,
        out: &mut Vec<usize>,
    ) {
        out.clear();
        let mut end = self.valid.partition_point(|&v| v <= valid_at);
        while out.len() < k {
            let Some(row) = self.last_known_before(end, known_at) else {
                break;
            };
            out.push(row);
            // Skip the remaining (earlier) revisions of this valid time.
            let v = self.valid[row];
            end = self.valid[..row].partition_point(|&x| x < v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(n: i64) -> Timestamp {
        Timestamp::new(n).unwrap()
    }

    fn series(rows: &[(i64, i64, f32)]) -> Series {
        let mut s = Series::default();
        for &(v, k, x) in rows {
            s.record(t(v), t(k), &[x], String::new).unwrap();
        }
        s
    }

    fn value_at(s: &Series, valid: i64, known: i64) -> Option<f32> {
        s.latest(t(valid), t(known)).map(|r| s.row(r, 1)[0])
    }

    #[test]
    fn revisions_respect_knowledge_time() {
        // Q1 reported at 10 as 1.0, restated at 50 as 1.5; Q2 reported at 30.
        let s = series(&[(5, 10, 1.0), (5, 50, 1.5), (25, 30, 2.0)]);
        assert_eq!(value_at(&s, 100, 9), None);
        assert_eq!(value_at(&s, 100, 10), Some(1.0));
        assert_eq!(value_at(&s, 100, 30), Some(2.0));
        assert_eq!(value_at(&s, 20, 30), Some(1.0), "Q2 not valid yet");
        assert_eq!(value_at(&s, 20, 50), Some(1.5), "restatement visible");
        let mut out = vec![];
        s.history(t(100), t(49), 5, &mut out);
        let vals: Vec<f32> = out.iter().map(|&r| s.row(r, 1)[0]).collect();
        assert_eq!(vals, vec![2.0, 1.0]);
        s.history(t(100), t(60), 5, &mut out);
        let vals: Vec<f32> = out.iter().map(|&r| s.row(r, 1)[0]).collect();
        assert_eq!(vals, vec![2.0, 1.5]);
    }

    #[test]
    fn out_of_order_and_duplicates() {
        let mut s = series(&[(30, 31, 3.0), (10, 11, 1.0), (20, 21, 2.0)]);
        assert_eq!(
            s.record(t(20), t(21), &[2.0], String::new).unwrap(),
            Recorded::Duplicate
        );
        assert!(s.record(t(20), t(21), &[9.0], String::new).is_err());
        let mut out = vec![];
        s.history(t(100), t(100), 10, &mut out);
        assert_eq!(out.len(), 3);
        assert_eq!(s.row(out[0], 1)[0], 3.0);
        assert_eq!(s.row(out[2], 1)[0], 1.0);
    }

    #[test]
    fn block_skipping_matches_linear_scan() {
        // 1000 rows; every third row was back-filled with a late known time.
        let mut s = Series::default();
        let mut rows = vec![];
        for i in 0..1000i64 {
            let known = if i % 3 == 0 { 10_000 + i } else { i + 1 };
            rows.push((i, known));
        }
        // Insert in a scrambled order to exercise block rebuilds.
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| (i * 7919) % rows.len());
        for &i in &order {
            let (v, k) = rows[i];
            s.record(t(v), t(k), &[v as f32], String::new).unwrap();
        }
        for &(valid_at, known_at) in &[(999, 500), (999, 10_500), (400, 300), (0, 0), (999, 1)] {
            let expected = rows
                .iter()
                .filter(|&&(v, k)| v <= valid_at && k <= known_at)
                .max_by_key(|&&(v, k)| (v, k))
                .map(|&(v, _)| v as f32);
            assert_eq!(
                value_at(&s, valid_at, known_at),
                expected,
                "{valid_at} {known_at}"
            );
        }
    }
}

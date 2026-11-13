//! The bitemporal heterogeneous graph store.
//!
//! Ingestion is batched and atomic: a batch is fully validated (shapes,
//! values, intervals, node keys) before anything is written, and if a
//! conflict is discovered while writing, every row already written by that
//! batch is rolled back. Re-ingesting identical rows is a no-op, so pipelines
//! can safely replay.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::schema::{ChannelId, KindId, NodeClass, NodeKindSpec, RelationId, Schema, ValueSpec};
use crate::series::Series;
use crate::time::{Interval, Timestamp};
use crate::versions::{Recorded, Version, VersionList};

/// Dense per-kind node id. Internal: never exposed, never relied on for
/// output order.
pub(crate) type Local = u32;

pub(crate) type Attrs = Box<[f32]>;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct KindStore {
    pub(crate) keys: Vec<Box<str>>,
    pub(crate) lifetimes: Vec<VersionList<()>>,
    /// `[channel][local]`, grown lazily.
    pub(crate) channels: Vec<Vec<Series>>,
    #[serde(skip)]
    pub(crate) key_index: HashMap<Box<str>, Local>,
    /// Event kinds only: `(version start, node)` for every asserted lifetime
    /// version. A superset index – entries are re-checked on use.
    #[serde(skip)]
    pub(crate) starts: BTreeSet<(Timestamp, Local)>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct RelationStore {
    pub(crate) edges: BTreeMap<(Local, Local), VersionList<Attrs>>,
    /// `(dst, src)` for every key in `edges`.
    #[serde(skip)]
    pub(crate) by_dst: BTreeSet<(Local, Local)>,
}

/// Counts returned from a batch write.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct BatchStats {
    /// Rows that added new information.
    pub new: usize,
    /// Rows identical to something already recorded.
    pub duplicate: usize,
}

impl BatchStats {
    fn add(&mut self, r: Recorded) {
        match r {
            Recorded::New => self.new += 1,
            Recorded::Duplicate => self.duplicate += 1,
        }
    }
}

/// Undo record for batch rollback.
enum Undo {
    Lifetime(KindId, Local, Timestamp, Interval),
    Observation(KindId, ChannelId, Local, Timestamp, Timestamp),
    Edge(RelationId, Local, Local, Timestamp, Interval),
}

#[derive(Clone, Debug, Default)]
pub struct TemporalGraph {
    pub(crate) schema: Schema,
    pub(crate) kinds: Vec<KindStore>,
    pub(crate) relations: Vec<RelationStore>,
}

fn check_len(context: &str, expected: usize, got: usize) -> Result<()> {
    if expected == got {
        Ok(())
    } else {
        Err(Error::BatchLength {
            context: format!("{context}: expected {expected} rows, got {got}"),
        })
    }
}

impl TemporalGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    // ---------------------------------------------------------------- schema

    pub fn define_kind(&mut self, spec: NodeKindSpec) -> Result<KindId> {
        let n_channels = spec.channels.len();
        let id = self.schema.define_kind(spec)?;
        if id.0 as usize == self.kinds.len() {
            self.kinds.push(KindStore {
                channels: vec![Vec::new(); n_channels],
                ..KindStore::default()
            });
        }
        Ok(id)
    }

    pub fn define_relation(
        &mut self,
        name: &str,
        src: &str,
        dst: &str,
        attrs: ValueSpec,
    ) -> Result<RelationId> {
        let id = self.schema.define_relation(name, src, dst, attrs)?;
        if id.0 as usize == self.relations.len() {
            self.relations.push(RelationStore::default());
        }
        Ok(id)
    }

    // ------------------------------------------------------------- inspection

    pub fn node_count(&self, kind: &str) -> Result<usize> {
        Ok(self.kinds[self.schema.kind_id(kind)?.0 as usize].keys.len())
    }

    pub fn edge_key_count(&self, relation: &str) -> Result<usize> {
        Ok(
            self.relations[self.schema.relation_id(relation)?.0 as usize]
                .edges
                .len(),
        )
    }

    pub fn observation_count(&self, kind: &str, channel: &str) -> Result<usize> {
        let k = self.schema.kind_id(kind)?;
        let c = self.schema.channel_id(k, channel)?;
        Ok(self.kinds[k.0 as usize].channels[c.0 as usize]
            .iter()
            .map(Series::len)
            .sum())
    }

    /// Latest knowledge time recorded anywhere in the store, i.e. the
    /// earliest `known_at` at which a snapshot reflects everything ingested.
    pub fn max_known_time(&self) -> Option<i64> {
        let nodes = self.kinds.iter().flat_map(|k| {
            k.lifetimes
                .iter()
                .flat_map(|l| l.versions().iter().map(|v| v.known))
                .chain(
                    k.channels
                        .iter()
                        .flatten()
                        .flat_map(|s| (0..s.len()).map(move |r| s.known_time(r))),
                )
        });
        let edges = self.relations.iter().flat_map(|r| {
            r.edges
                .values()
                .flat_map(|l| l.versions().iter().map(|v| v.known))
        });
        nodes.chain(edges).max().map(Timestamp::nanos)
    }

    // ------------------------------------------------------------- ingestion

    fn lookup(&self, kind: KindId, key: &str) -> Result<Local> {
        self.kinds[kind.0 as usize]
            .key_index
            .get(key)
            .copied()
            .ok_or_else(|| Error::UnknownNode {
                kind: self.schema.kind(kind).name.clone(),
                key: key.to_owned(),
            })
    }

    fn intern(&mut self, kind: KindId, key: &str) -> Result<(Local, bool)> {
        let store = &mut self.kinds[kind.0 as usize];
        if let Some(&local) = store.key_index.get(key) {
            return Ok((local, false));
        }
        if key.is_empty() || key.len() > 1024 {
            return Err(Error::InvalidName {
                name: key.chars().take(64).collect(),
                reason: "node keys must be 1..=1024 bytes",
            });
        }
        if store.keys.len() >= Local::MAX as usize {
            return Err(Error::Capacity(format!(
                "kind {:?} reached {} nodes",
                self.schema.kind(kind).name,
                Local::MAX
            )));
        }
        let local = store.keys.len() as Local;
        let boxed: Box<str> = key.into();
        store.keys.push(boxed.clone());
        store.key_index.insert(boxed, local);
        store.lifetimes.push(VersionList::default());
        Ok((local, true))
    }

    fn rollback(&mut self, undo: Vec<Undo>, created: &[(KindId, usize)]) {
        for u in undo.into_iter().rev() {
            match u {
                Undo::Lifetime(k, n, known, valid) => {
                    self.kinds[k.0 as usize].lifetimes[n as usize].remove(known, valid)
                }
                Undo::Observation(k, c, n, valid, known) => {
                    let dim = self.schema.kind(k).channel(c).value.dim;
                    self.kinds[k.0 as usize].channels[c.0 as usize][n as usize]
                        .remove(valid, known, dim)
                }
                Undo::Edge(r, s, d, known, valid) => {
                    let store = &mut self.relations[r.0 as usize];
                    if let Some(list) = store.edges.get_mut(&(s, d)) {
                        list.remove(known, valid);
                        if list.is_empty() {
                            store.edges.remove(&(s, d));
                            store.by_dst.remove(&(d, s));
                        }
                    }
                }
            }
        }
        // Drop node keys created by the failed batch (they were appended).
        for &(kind, first_new) in created {
            let store = &mut self.kinds[kind.0 as usize];
            for key in store.keys.drain(first_new..) {
                store.key_index.remove(&key);
            }
            store.lifetimes.truncate(first_new);
            for ch in &mut store.channels {
                ch.truncate(first_new);
            }
        }
        // `starts` may retain stale entries; it is a superset index.
    }

    fn intervals(
        context: &str,
        n: usize,
        valid_from: &[i64],
        valid_to: &[i64],
        known: &[i64],
    ) -> Result<Vec<(Interval, Timestamp)>> {
        check_len(&format!("{context} valid_from"), n, valid_from.len())?;
        check_len(&format!("{context} valid_to"), n, valid_to.len())?;
        check_len(&format!("{context} known_time"), n, known.len())?;
        (0..n)
            .map(|i| {
                Ok((
                    Interval::new(valid_from[i], valid_to[i])?,
                    Timestamp::new(known[i])?,
                ))
            })
            .collect()
    }

    fn lifetime_batch<K: AsRef<str>>(
        &mut self,
        kind: &str,
        keys: &[K],
        valid_from: &[i64],
        valid_to: &[i64],
        known: &[i64],
        assert: bool,
    ) -> Result<BatchStats> {
        let k = self.schema.kind_id(kind)?;
        let rows = Self::intervals(kind, keys.len(), valid_from, valid_to, known)?;
        let is_event = matches!(self.schema.kind(k).class, NodeClass::Event { .. });
        let first_new = self.kinds[k.0 as usize].keys.len();
        let mut undo = Vec::new();
        let mut stats = BatchStats::default();
        for (key, &(valid, known)) in keys.iter().zip(&rows) {
            let key = key.as_ref();
            let result = self.intern(k, key).and_then(|(local, _)| {
                let version = Version {
                    known,
                    valid,
                    payload: assert.then_some(()),
                };
                let store = &mut self.kinds[k.0 as usize];
                let r = store.lifetimes[local as usize]
                    .record(version, || format!("{kind} {key:?} lifetime"))?;
                if r == Recorded::New {
                    undo.push(Undo::Lifetime(k, local, known, valid));
                    if is_event && assert {
                        store.starts.insert((valid.start(), local));
                    }
                }
                Ok(r)
            });
            match result {
                Ok(r) => stats.add(r),
                Err(e) => {
                    self.rollback(undo, &[(k, first_new)]);
                    return Err(e);
                }
            }
        }
        Ok(stats)
    }

    /// Record that each node exists over `[valid_from, valid_to)`, as known
    /// at `known_time`. Creates node keys on first sight. Use `i64::MIN` /
    /// `i64::MAX` for open interval ends.
    pub fn upsert_nodes<K: AsRef<str>>(
        &mut self,
        kind: &str,
        keys: &[K],
        valid_from: &[i64],
        valid_to: &[i64],
        known: &[i64],
    ) -> Result<BatchStats> {
        self.lifetime_batch(kind, keys, valid_from, valid_to, known, true)
    }

    /// Record that each node does *not* exist over `[valid_from, valid_to)`
    /// (delisting, merger, retracted article), as known at `known_time`.
    pub fn retract_nodes<K: AsRef<str>>(
        &mut self,
        kind: &str,
        keys: &[K],
        valid_from: &[i64],
        valid_to: &[i64],
        known: &[i64],
    ) -> Result<BatchStats> {
        self.lifetime_batch(kind, keys, valid_from, valid_to, known, false)
    }

    /// Record observations for one channel. `values` is row-major
    /// `[keys.len(), dim]`. Nodes must already exist.
    pub fn observe<K: AsRef<str>>(
        &mut self,
        kind: &str,
        channel: &str,
        keys: &[K],
        valid_time: &[i64],
        known_time: &[i64],
        values: &[f32],
    ) -> Result<BatchStats> {
        let k = self.schema.kind_id(kind)?;
        let c = self.schema.channel_id(k, channel)?;
        let spec: ValueSpec = self.schema.kind(k).channel(c).value.clone();
        let n = keys.len();
        let ctx = format!("{kind}.{channel}");
        check_len(&format!("{ctx} valid_time"), n, valid_time.len())?;
        check_len(&format!("{ctx} known_time"), n, known_time.len())?;
        check_len(&format!("{ctx} values"), n * spec.dim, values.len())?;
        // Validate everything before writing anything.
        let mut rows = Vec::with_capacity(n);
        for i in 0..n {
            let key = keys[i].as_ref();
            let local = self.lookup(k, key)?;
            let (v, kn) = (
                Timestamp::new(valid_time[i])?,
                Timestamp::new(known_time[i])?,
            );
            let x = &values[i * spec.dim..(i + 1) * spec.dim];
            spec.validate(x, || format!("{ctx} {key:?} @ {}", v.nanos()))?;
            rows.push((local, v, kn));
        }
        let mut undo = Vec::new();
        let mut stats = BatchStats::default();
        let series_vec = &mut self.kinds[k.0 as usize].channels[c.0 as usize];
        for (i, &(local, v, kn)) in rows.iter().enumerate() {
            if series_vec.len() <= local as usize {
                series_vec.resize_with(local as usize + 1, Series::default);
            }
            let x = &values[i * spec.dim..(i + 1) * spec.dim];
            match series_vec[local as usize].record(v, kn, x, || {
                format!(
                    "{ctx} {:?} @ valid {} known {}",
                    keys[i].as_ref(),
                    v.nanos(),
                    kn.nanos()
                )
            }) {
                Ok(r) => {
                    if r == Recorded::New {
                        undo.push(Undo::Observation(k, c, local, v, kn));
                    }
                    stats.add(r)
                }
                Err(e) => {
                    self.rollback(undo, &[]);
                    return Err(e);
                }
            }
        }
        Ok(stats)
    }

    #[allow(clippy::too_many_arguments)]
    fn edge_batch<K: AsRef<str>>(
        &mut self,
        relation: &str,
        src: &[K],
        dst: &[K],
        valid_from: &[i64],
        valid_to: &[i64],
        known: &[i64],
        attrs: Option<&[f32]>,
    ) -> Result<BatchStats> {
        let r = self.schema.relation_id(relation)?;
        let spec = self.schema.relation(r).clone();
        let n = src.len();
        check_len(&format!("{relation} dst"), n, dst.len())?;
        let rows = Self::intervals(relation, n, valid_from, valid_to, known)?;
        let dim = spec.attrs.dim;
        if let Some(a) = attrs {
            check_len(&format!("{relation} attrs"), n * dim, a.len())?;
        }
        let mut resolved = Vec::with_capacity(n);
        for i in 0..n {
            let (s, d) = (src[i].as_ref(), dst[i].as_ref());
            let sl = self.lookup(spec.src, s)?;
            let dl = self.lookup(spec.dst, d)?;
            let payload: Option<Attrs> = match attrs {
                Some(a) => {
                    let x = &a[i * dim..(i + 1) * dim];
                    spec.attrs
                        .validate(x, || format!("{relation} {s:?}->{d:?}"))?;
                    Some(x.into())
                }
                None => None,
            };
            resolved.push((sl, dl, payload));
        }
        let mut undo = Vec::new();
        let mut stats = BatchStats::default();
        for (i, ((sl, dl, payload), &(valid, known))) in resolved.into_iter().zip(&rows).enumerate()
        {
            let store = &mut self.relations[r.0 as usize];
            let list = store.edges.entry((sl, dl)).or_default();
            let fresh = list.is_empty();
            let result = list.record(
                Version {
                    known,
                    valid,
                    payload,
                },
                || format!("{relation} {:?}->{:?}", src[i].as_ref(), dst[i].as_ref()),
            );
            match result {
                Ok(rec) => {
                    if fresh {
                        store.by_dst.insert((dl, sl));
                    }
                    if rec == Recorded::New {
                        undo.push(Undo::Edge(r, sl, dl, known, valid));
                    }
                    stats.add(rec);
                }
                Err(e) => {
                    if fresh {
                        store.edges.remove(&(sl, dl));
                    }
                    self.rollback(undo, &[]);
                    return Err(e);
                }
            }
        }
        Ok(stats)
    }

    /// Assert edges `src --relation--> dst` over `[valid_from, valid_to)` as
    /// known at `known_time`. `attrs` is row-major `[n, attr_dim]`.
    #[allow(clippy::too_many_arguments)]
    pub fn assert_edges<K: AsRef<str>>(
        &mut self,
        relation: &str,
        src: &[K],
        dst: &[K],
        valid_from: &[i64],
        valid_to: &[i64],
        known: &[i64],
        attrs: &[f32],
    ) -> Result<BatchStats> {
        self.edge_batch(relation, src, dst, valid_from, valid_to, known, Some(attrs))
    }

    /// Record that edges do not exist over `[valid_from, valid_to)`.
    pub fn retract_edges<K: AsRef<str>>(
        &mut self,
        relation: &str,
        src: &[K],
        dst: &[K],
        valid_from: &[i64],
        valid_to: &[i64],
        known: &[i64],
    ) -> Result<BatchStats> {
        self.edge_batch(relation, src, dst, valid_from, valid_to, known, None)
    }

    /// Rebuild derived indexes (after deserialization).
    pub(crate) fn rebuild_indexes(&mut self) -> Result<()> {
        for (ki, store) in self.kinds.iter_mut().enumerate() {
            let spec = self.schema.kind(KindId(ki as u16));
            store.key_index = HashMap::with_capacity(store.keys.len());
            for (i, key) in store.keys.iter().enumerate() {
                if store.key_index.insert(key.clone(), i as Local).is_some() {
                    return Err(Error::Format(format!(
                        "duplicate key {key:?} in {}",
                        spec.name
                    )));
                }
            }
            store.starts.clear();
            if matches!(spec.class, NodeClass::Event { .. }) {
                for (i, l) in store.lifetimes.iter().enumerate() {
                    for v in l.versions().iter().filter(|v| v.payload.is_some()) {
                        store.starts.insert((v.valid.start(), i as Local));
                    }
                }
            }
        }
        for store in &mut self.relations {
            store.by_dst = store.edges.keys().map(|&(s, d)| (d, s)).collect();
        }
        Ok(())
    }
}

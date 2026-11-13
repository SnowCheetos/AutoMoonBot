//! Point-in-time materialization.
//!
//! [`TemporalGraph::snapshot`] answers "what did the graph look like at valid
//! time `valid_at`, using only facts known by `known_at`?". Every fact it
//! returns satisfies `known_time <= known_at`, which the property tests in
//! `tests/leakage.rs` verify against an independent reference model.
//!
//! Output is deterministic and independent of ingestion order: node rows are
//! sorted by key, edges by `(src row, dst row)`.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::schema::{KindId, NodeClass, RelationId};
use crate::store::{Attrs, Local, TemporalGraph};
use crate::time::Timestamp;
use crate::versions::VersionList;

/// Padding value for timestamps of missing history slots.
pub const MISSING_TIME: i64 = i64::MIN;

/// Rows per rayon task; below this the work is too small to split.
const PAR_MIN: usize = 256;

#[derive(Clone, Debug, Default)]
pub struct SnapshotRequest {
    pub known_at: i64,
    /// Defaults to `known_at`.
    pub valid_at: Option<i64>,
    /// History length per `"kind.channel"`; default 1.
    pub history: HashMap<String, usize>,
    /// Event window override per event kind, in nanoseconds.
    pub windows: HashMap<String, i64>,
}

impl SnapshotRequest {
    pub fn at(known_at: i64) -> Self {
        SnapshotRequest {
            known_at,
            ..Default::default()
        }
    }
}

/// Feature tensor for one channel of one node kind.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelBlock {
    pub name: String,
    pub dim: usize,
    /// History length `K`.
    pub history: usize,
    /// `[N, K, D]`, chronological and right-aligned: slot `K-1` is the latest
    /// observation. Missing slots and null elements are `0.0`.
    pub values: Vec<f32>,
    /// `[N, K]`, 1 where the slot holds an observation.
    pub present: Vec<u8>,
    /// `[N, K, D]`, 1 where the element is non-null. Only for nullable
    /// channels.
    pub element_present: Option<Vec<u8>>,
    /// `[N, K]` valid time of each slot, [`MISSING_TIME`] when absent.
    pub valid_time: Vec<i64>,
    /// `[N, K]` knowledge time of each slot, [`MISSING_TIME`] when absent.
    pub known_time: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeBlock {
    pub kind: String,
    pub keys: Vec<String>,
    /// Start of the lifetime interval that makes each node present
    /// (`i64::MIN` when unbounded).
    pub since: Vec<i64>,
    /// Knowledge time of that lifetime assertion.
    pub known_time: Vec<i64>,
    pub channels: Vec<ChannelBlock>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EdgeBlock {
    pub relation: String,
    pub src_kind: String,
    pub dst_kind: String,
    /// Row indices into the source / destination node blocks.
    pub src: Vec<i64>,
    pub dst: Vec<i64>,
    pub attr_dim: usize,
    /// `[E, A]`; null elements are `0.0`.
    pub attrs: Vec<f32>,
    /// `[E, A]`, only for nullable attributes.
    pub attr_present: Option<Vec<u8>>,
    /// Start of the deciding validity interval (`i64::MIN` when unbounded).
    pub since: Vec<i64>,
    pub known_time: Vec<i64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Snapshot {
    pub known_at: i64,
    pub valid_at: i64,
    pub nodes: Vec<NodeBlock>,
    pub edges: Vec<EdgeBlock>,
}

impl Snapshot {
    pub fn node_block(&self, kind: &str) -> Option<&NodeBlock> {
        self.nodes.iter().find(|b| b.kind == kind)
    }

    pub fn edge_block(&self, relation: &str) -> Option<&EdgeBlock> {
        self.edges.iter().find(|b| b.relation == relation)
    }
}

impl NodeBlock {
    pub fn channel(&self, name: &str) -> Option<&ChannelBlock> {
        self.channels.iter().find(|c| c.name == name)
    }

    pub fn row(&self, key: &str) -> Option<usize> {
        self.keys.binary_search_by(|k| k.as_str().cmp(key)).ok()
    }
}

/// Mutable output slices for one node row of a channel block.
type RowSlots<'a> = (
    &'a mut [f32],
    (
        &'a mut [u8],
        (&'a mut [i64], (&'a mut [i64], Option<&'a mut [u8]>)),
    ),
);

struct Present {
    /// Locals in output-row order.
    locals: Vec<Local>,
    since: Vec<i64>,
    known: Vec<i64>,
    /// `local -> row`, `u32::MAX` when absent.
    row_of: Vec<u32>,
}

impl TemporalGraph {
    fn present_nodes(
        &self,
        kind: KindId,
        valid_at: Timestamp,
        known_at: Timestamp,
        window: Option<i64>,
    ) -> Present {
        let store = &self.kinds[kind.0 as usize];
        let resolve = |local: Local| {
            store.lifetimes[local as usize]
                .resolve(valid_at, known_at)
                .map(|v| (local, v.valid.start(), v.known))
        };
        let mut found: Vec<(Local, Timestamp, Timestamp)> = match window {
            None => (0..store.keys.len() as Local)
                .into_par_iter()
                .with_min_len(PAR_MIN)
                .filter_map(resolve)
                .collect(),
            Some(w) => {
                let lo = valid_at.saturating_sub(w);
                let mut candidates: Vec<Local> = store
                    .starts
                    .range((lo, 0)..=(valid_at, Local::MAX))
                    .map(|&(_, l)| l)
                    .collect();
                candidates.sort_unstable();
                candidates.dedup();
                candidates
                    .into_par_iter()
                    .with_min_len(PAR_MIN)
                    .filter_map(resolve)
                    // The deciding version must itself have started inside
                    // the window.
                    .filter(|&(_, since, _)| since >= lo)
                    .collect()
            }
        };
        found.par_sort_unstable_by(|a, b| store.keys[a.0 as usize].cmp(&store.keys[b.0 as usize]));
        let mut row_of = vec![u32::MAX; store.keys.len()];
        for (row, &(local, _, _)) in found.iter().enumerate() {
            row_of[local as usize] = row as u32;
        }
        Present {
            locals: found.iter().map(|f| f.0).collect(),
            since: found.iter().map(|f| f.1.nanos()).collect(),
            known: found.iter().map(|f| f.2.nanos()).collect(),
            row_of,
        }
    }

    pub fn snapshot(&self, req: &SnapshotRequest) -> Result<Snapshot> {
        let known_at = Timestamp::new(req.known_at)?;
        let valid_at = Timestamp::new(req.valid_at.unwrap_or(req.known_at))?;
        let schema = &self.schema;

        // Validate request keys up front so typos fail loudly.
        let mut history: HashMap<(KindId, u16), usize> = HashMap::new();
        for (name, &k) in &req.history {
            let (kind, channel) = name.split_once('.').ok_or_else(|| Error::Unknown {
                what: "channel (expected \"kind.channel\")",
                name: name.clone(),
            })?;
            let kid = schema.kind_id(kind)?;
            let cid = schema.channel_id(kid, channel)?;
            if k == 0 || k > 1 << 16 {
                return Err(Error::Schema(format!(
                    "history for {name} must be in 1..=65536"
                )));
            }
            history.insert((kid, cid.0), k);
        }
        let mut windows: HashMap<KindId, i64> = HashMap::new();
        for (name, &w) in &req.windows {
            let kid = schema.kind_id(name)?;
            if !matches!(schema.kind(kid).class, NodeClass::Event { .. }) {
                return Err(Error::Schema(format!("{name} is not an event kind")));
            }
            if w <= 0 {
                return Err(Error::Schema(format!("window for {name} must be positive")));
            }
            windows.insert(kid, w);
        }

        let mut present = Vec::with_capacity(schema.kinds().len());
        let mut nodes = Vec::with_capacity(schema.kinds().len());
        for (ki, spec) in schema.kinds().iter().enumerate() {
            let kid = KindId(ki as u16);
            let window = match spec.class {
                NodeClass::Entity => None,
                NodeClass::Event { window_ns } => {
                    Some(windows.get(&kid).copied().unwrap_or(window_ns))
                }
            };
            let p = self.present_nodes(kid, valid_at, known_at, window);
            let store = &self.kinds[ki];
            let channels = spec
                .channels
                .iter()
                .enumerate()
                .map(|(ci, cspec)| {
                    let k = history.get(&(kid, ci as u16)).copied().unwrap_or(1);
                    self.channel_block(
                        &store.channels[ci],
                        &p.locals,
                        &cspec.name,
                        cspec.value.dim,
                        cspec.value.nullable,
                        k,
                        valid_at,
                        known_at,
                    )
                })
                .collect();
            nodes.push(NodeBlock {
                kind: spec.name.clone(),
                keys: p
                    .locals
                    .iter()
                    .map(|&l| store.keys[l as usize].to_string())
                    .collect(),
                since: p.since.clone(),
                known_time: p.known.clone(),
                channels,
            });
            present.push(p);
        }

        let edges = (0..schema.relations().len())
            .map(|ri| self.edge_block(RelationId(ri as u16), &present, valid_at, known_at))
            .collect();

        Ok(Snapshot {
            known_at: known_at.nanos(),
            valid_at: valid_at.nanos(),
            nodes,
            edges,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn channel_block(
        &self,
        series: &[crate::series::Series],
        locals: &[Local],
        name: &str,
        dim: usize,
        nullable: bool,
        k: usize,
        valid_at: Timestamp,
        known_at: Timestamp,
    ) -> ChannelBlock {
        let n = locals.len();
        let mut values = vec![0f32; n * k * dim];
        let mut present = vec![0u8; n * k];
        let mut element_present = nullable.then(|| vec![0u8; n * k * dim]);
        let mut valid_time = vec![MISSING_TIME; n * k];
        let mut known_time = vec![MISSING_TIME; n * k];

        let fill = |(i, (vals, (pres, (vt, (kt, mut ep))))): (usize, RowSlots<'_>)| {
            let Some(s) = series.get(locals[i] as usize) else {
                return;
            };
            let mut rows = Vec::with_capacity(k);
            if k == 1 {
                rows.extend(s.latest(valid_at, known_at));
            } else {
                s.history(valid_at, known_at, k, &mut rows);
            }
            // rows are most-recent-first; place right-aligned chronologically.
            for (j, &row) in rows.iter().enumerate() {
                let slot = k - 1 - j;
                pres[slot] = 1;
                vt[slot] = s.valid_time(row).nanos();
                kt[slot] = s.known_time(row).nanos();
                let src = s.row(row, dim);
                let dst = &mut vals[slot * dim..(slot + 1) * dim];
                for (e, (&x, out)) in src.iter().zip(dst.iter_mut()).enumerate() {
                    if x.is_nan() {
                        *out = 0.0;
                    } else {
                        *out = x;
                        if let Some(ep) = ep.as_deref_mut() {
                            ep[slot * dim + e] = 1;
                        }
                    }
                }
            }
        };

        if n > 0 {
            let ep_chunks: Vec<Option<&mut [u8]>> = match element_present.as_mut() {
                Some(ep) => ep.chunks_mut(k * dim).map(Some).collect(),
                None => (0..n).map(|_| None).collect(),
            };
            values
                .par_chunks_mut(k * dim)
                .zip(
                    present.par_chunks_mut(k).zip(
                        valid_time
                            .par_chunks_mut(k)
                            .zip(known_time.par_chunks_mut(k).zip(ep_chunks.into_par_iter())),
                    ),
                )
                .enumerate()
                .with_min_len(PAR_MIN)
                .for_each(fill);
        }

        ChannelBlock {
            name: name.to_owned(),
            dim,
            history: k,
            values,
            present,
            element_present,
            valid_time,
            known_time,
        }
    }

    fn edge_block(
        &self,
        rel: RelationId,
        present: &[Present],
        valid_at: Timestamp,
        known_at: Timestamp,
    ) -> EdgeBlock {
        let spec = self.schema.relation(rel);
        let store = &self.relations[rel.0 as usize];
        let (sp, dp) = (&present[spec.src.0 as usize], &present[spec.dst.0 as usize]);
        let is_event = |k: KindId| matches!(self.schema.kind(k).class, NodeClass::Event { .. });

        // Choose the cheapest enumeration: iterate from a windowed (event)
        // endpoint when there is one, otherwise from whichever endpoint set is
        // smaller relative to the number of edge keys.
        type Candidate<'a> = ((Local, Local), &'a VersionList<Attrs>);
        let candidates: Vec<Candidate<'_>> = if is_event(spec.src)
            || (!is_event(spec.dst) && sp.locals.len() * 8 < store.edges.len())
        {
            sp.locals
                .iter()
                .flat_map(|&s| {
                    store
                        .edges
                        .range((s, 0)..=(s, Local::MAX))
                        .map(|(&k, l)| (k, l))
                })
                .collect()
        } else if is_event(spec.dst) || dp.locals.len() * 8 < store.edges.len() {
            dp.locals
                .iter()
                .flat_map(|&d| {
                    store
                        .by_dst
                        .range((d, 0)..=(d, Local::MAX))
                        .map(|&(d, s)| ((s, d), &store.edges[&(s, d)]))
                })
                .collect()
        } else {
            store.edges.iter().map(|(&k, l)| (k, l)).collect()
        };

        let dim = spec.attrs.dim;
        let mut found: Vec<(u32, u32, i64, i64, &[f32])> = candidates
            .into_par_iter()
            .with_min_len(PAR_MIN)
            .filter_map(|((s, d), list)| {
                let (rs, rd) = (sp.row_of[s as usize], dp.row_of[d as usize]);
                if rs == u32::MAX || rd == u32::MAX {
                    return None;
                }
                let v = list.resolve(valid_at, known_at)?;
                let attrs: &[f32] = v
                    .payload
                    .as_deref()
                    .expect("resolve only returns assertions");
                Some((rs, rd, v.valid.start().nanos(), v.known.nanos(), attrs))
            })
            .collect();
        found.par_sort_unstable_by_key(|f| (f.0, f.1));

        let e = found.len();
        let mut attrs = Vec::with_capacity(e * dim);
        let mut attr_present = spec.attrs.nullable.then(|| Vec::with_capacity(e * dim));
        for f in &found {
            for &x in f.4 {
                attrs.push(if x.is_nan() { 0.0 } else { x });
                if let Some(ap) = attr_present.as_mut() {
                    ap.push(u8::from(!x.is_nan()));
                }
            }
        }
        EdgeBlock {
            relation: spec.name.clone(),
            src_kind: self.schema.kind(spec.src).name.clone(),
            dst_kind: self.schema.kind(spec.dst).name.clone(),
            src: found.iter().map(|f| f.0 as i64).collect(),
            dst: found.iter().map(|f| f.1 as i64).collect(),
            attr_dim: dim,
            attrs,
            attr_present,
            since: found.iter().map(|f| f.2).collect(),
            known_time: found.iter().map(|f| f.3).collect(),
        }
    }
}

/// A snapshot flattened to a single typed graph: the input layout for graph
/// transformers (one token per node, sparse attention over edges, type
/// embeddings) and for structural encodings computed over the whole graph.
#[derive(Clone, Debug, PartialEq)]
pub struct Homogeneous {
    /// `[K + 1]`: nodes of kind `k` are tokens `kind_offsets[k]..kind_offsets[k + 1]`,
    /// in the same row order as the corresponding [`NodeBlock`].
    pub kind_offsets: Vec<i64>,
    /// `[N]` kind index of each token.
    pub node_type: Vec<i64>,
    /// `[E]` global source/destination tokens, sorted by `(src, dst, type)`.
    pub src: Vec<i64>,
    pub dst: Vec<i64>,
    /// `[E]` relation index; reversed edges (when requested) use
    /// `relation + R` where `R` is the number of relations.
    pub edge_type: Vec<i64>,
    /// `[E]` row of the edge in its relation's [`EdgeBlock`] (to gather
    /// attributes and temporal fields).
    pub edge_row: Vec<i64>,
    /// `[N + 1]` CSR row pointer over `src`; `dst` doubles as the column
    /// index array.
    pub indptr: Vec<i64>,
}

impl Snapshot {
    /// Flatten into a [`Homogeneous`] graph. With `add_reverse`, every edge
    /// also appears reversed under its own type, so information can flow in
    /// both directions without conflating the two roles.
    pub fn homogeneous(&self, add_reverse: bool) -> Homogeneous {
        let mut kind_offsets = Vec::with_capacity(self.nodes.len() + 1);
        let mut total = 0i64;
        let mut offset_of: HashMap<&str, i64> = HashMap::new();
        for b in &self.nodes {
            kind_offsets.push(total);
            offset_of.insert(b.kind.as_str(), total);
            total += b.keys.len() as i64;
        }
        kind_offsets.push(total);
        let mut node_type = Vec::with_capacity(total as usize);
        for (k, b) in self.nodes.iter().enumerate() {
            node_type.extend(std::iter::repeat_n(k as i64, b.keys.len()));
        }

        let r = self.edges.len() as i64;
        let count: usize =
            self.edges.iter().map(|e| e.src.len()).sum::<usize>() * if add_reverse { 2 } else { 1 };
        let mut edges: Vec<(i64, i64, i64, i64)> = Vec::with_capacity(count);
        for (ri, e) in self.edges.iter().enumerate() {
            let (so, dof) = (
                offset_of[e.src_kind.as_str()],
                offset_of[e.dst_kind.as_str()],
            );
            for (row, (&s, &d)) in e.src.iter().zip(&e.dst).enumerate() {
                edges.push((so + s, dof + d, ri as i64, row as i64));
                if add_reverse {
                    edges.push((dof + d, so + s, ri as i64 + r, row as i64));
                }
            }
        }
        edges.par_sort_unstable();

        let mut indptr = vec![0i64; total as usize + 1];
        for &(s, ..) in &edges {
            indptr[s as usize + 1] += 1;
        }
        for i in 0..total as usize {
            indptr[i + 1] += indptr[i];
        }
        Homogeneous {
            kind_offsets,
            node_type,
            src: edges.iter().map(|e| e.0).collect(),
            dst: edges.iter().map(|e| e.1).collect(),
            edge_type: edges.iter().map(|e| e.2).collect(),
            edge_row: edges.iter().map(|e| e.3).collect(),
            indptr,
        }
    }
}

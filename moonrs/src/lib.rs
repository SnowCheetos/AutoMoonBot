//! # moonrs
//!
//! A bitemporal, point-in-time heterogeneous graph store for leakage-free
//! market research.
//!
//! Every fact (a node's existence, a feature observation, an edge) carries a
//! *valid time* (when it is true in the world) and a *known time* (when it
//! could first have been known). [`TemporalGraph::snapshot`] materializes the
//! graph as believed at a given knowledge time, using only facts known by
//! then, into dense per-kind feature tensors and per-relation edge lists ready
//! for a GNN.
//!
//! See [`time`] for the temporal model, [`schema`] for declaring node kinds,
//! channels and relations, and [`snapshot`] for the output layout.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod error;
pub mod persist;
pub mod schema;
pub mod series;
pub mod snapshot;
pub mod store;
pub mod time;
pub mod versions;

#[cfg(feature = "python")]
mod python;

pub use error::{Error, Result};
pub use schema::{ChannelSpec, NodeClass, NodeKindSpec, Schema, ValueSpec};
pub use snapshot::{ChannelBlock, EdgeBlock, MISSING_TIME, NodeBlock, Snapshot, SnapshotRequest};
pub use store::{BatchStats, TemporalGraph};
pub use time::{Interval, Timestamp};

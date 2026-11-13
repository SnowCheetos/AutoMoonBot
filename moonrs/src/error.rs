//! Error type shared by every fallible operation in the store.

use thiserror::Error;

/// Everything that can go wrong when defining a schema, ingesting facts,
/// building a snapshot, or (de)serializing a store.
///
/// Ingestion is strict by design: a record that would make a snapshot depend on
/// ingestion order, or that violates the schema, is rejected instead of being
/// silently coerced. Silent coercion is how look-ahead bias creeps in.
#[derive(Debug, Error, PartialEq)]
pub enum Error {
    #[error("invalid name {name:?}: {reason}")]
    InvalidName { name: String, reason: &'static str },

    #[error("{what} {name:?} is already defined with a different specification")]
    Redefinition { what: &'static str, name: String },

    #[error("unknown {what} {name:?}")]
    Unknown { what: &'static str, name: String },

    #[error("unknown {kind} node {key:?}")]
    UnknownNode { kind: String, key: String },

    #[error("invalid interval [{start}, {end}): start must be strictly before end")]
    InvalidInterval { start: i64, end: i64 },

    #[error("timestamp {0} is reserved (i64::MIN / i64::MAX are sentinels)")]
    ReservedTimestamp(i64),

    #[error("{context}: expected dimension {expected}, got {got}")]
    Dimension {
        context: String,
        expected: usize,
        got: usize,
    },

    #[error("{context}: element {index} = {value} is {reason}")]
    InvalidValue {
        context: String,
        index: usize,
        value: f32,
        reason: &'static str,
    },

    #[error("batch length mismatch: {context}")]
    BatchLength { context: String },

    #[error(
        "conflicting fact for {context}: a different value was already recorded with the \
         same knowledge time; record a correction with a later known_time instead"
    )]
    Conflict { context: String },

    #[error("{0}")]
    Schema(String),

    #[error("store file is corrupt or incompatible: {0}")]
    Format(String),

    #[error("i/o error: {0}")]
    Io(String),

    #[error("capacity exceeded: {0}")]
    Capacity(String),
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, Error>;

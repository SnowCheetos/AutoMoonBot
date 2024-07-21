pub mod data;
pub mod edges;
pub mod graph;
pub mod nodes;
pub mod utils;

#[macro_use]
extern crate lazy_static;
extern crate nalgebra as na;

pub use crate::{
    data::{aggregate::*, buffer::*, queue::*},
    utils::helpers::*,
};
pub use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};
pub use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    time::{Duration, Instant},
};
pub use uuid::Uuid;

#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
mod moonrs {
    use super::*;

    #[pyfunction]
    fn hello_python() -> &'static str {
        "Hello From Rust"
    }

    #[pymodule_export]
    use super::graph::hetero::HeteroGraph;
}

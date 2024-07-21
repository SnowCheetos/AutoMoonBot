#[macro_use]
extern crate lazy_static;
extern crate nalgebra as na;

#[cfg(feature = "python")]
use pyo3::prelude::*;

pub mod data;
pub mod edges;
pub mod graph;
pub mod nodes;
pub mod utils;

use data::{aggregate::*, buffer::*};
use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    time::{Duration, Instant},
};

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

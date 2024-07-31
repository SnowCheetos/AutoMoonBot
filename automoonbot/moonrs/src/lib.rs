#[cfg(feature = "python")]
use {
    pyo3::{
        exceptions::PyTypeError,
        prelude::*,
        types::{IntoPyDict, PyAny, PyDict, PyList},
    },
    std::time::{SystemTime, UNIX_EPOCH},
};

#[macro_use]
extern crate lazy_static;
extern crate nalgebra as na;

pub mod data;
pub mod edges;
pub mod graph;
pub mod nodes;
pub mod utils;

use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};
use std::{
    any::Any,
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

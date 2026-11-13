//! Python bindings (`automoonbot._moonrs`).
//!
//! Batches cross the boundary as contiguous NumPy arrays, which are read in
//! place. Snapshot tensors are handed to NumPy by moving the Rust buffers, so
//! nothing is copied on the way out. The GIL is released for ingestion,
//! snapshots and I/O.

use std::collections::HashMap;

use numpy::{
    IntoPyArray, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIOError, PyKeyError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::error::Error;
use crate::schema::{ChannelSpec, NodeClass, NodeKindSpec, ValueSpec};
use crate::snapshot::{Homogeneous, Snapshot, SnapshotRequest};
use crate::store::{BatchStats, TemporalGraph};

fn to_py(e: Error) -> PyErr {
    match e {
        Error::Io(_) => PyIOError::new_err(e.to_string()),
        Error::Unknown { .. } | Error::UnknownNode { .. } => PyKeyError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

fn slice<'a, T: numpy::Element>(a: &'a PyReadonlyArray1<'_, T>, name: &str) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a C-contiguous 1-D array")))
}

fn matrix<'a>(a: &'a PyReadonlyArray2<'_, f32>, name: &str, rows: usize) -> PyResult<&'a [f32]> {
    if a.shape()[0] != rows {
        return Err(PyValueError::new_err(format!(
            "{name} has {} rows, expected {rows}",
            a.shape()[0]
        )));
    }
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{name} must be a C-contiguous float32 array")))
}

fn stats(py: Python<'_>, s: BatchStats) -> PyResult<Bound<'_, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("new", s.new)?;
    d.set_item("duplicate", s.duplicate)?;
    Ok(d)
}

/// Parse `(name, dim, nullable, bounds)` tuples.
type ChannelArg = (String, usize, bool, Option<(f32, f32)>);

/// Bitemporal heterogeneous graph store.
#[pyclass(name = "TemporalGraph", module = "automoonbot._moonrs")]
#[derive(Debug, Default)]
pub struct PyTemporalGraph {
    inner: TemporalGraph,
}

#[pymethods]
impl PyTemporalGraph {
    #[new]
    fn new() -> Self {
        Self::default()
    }

    /// Declare a node kind. `window_ns` makes it an event kind with that
    /// default snapshot window; `None` makes it a persistent entity kind.
    #[pyo3(signature = (name, channels, window_ns=None))]
    fn define_kind(
        &mut self,
        name: String,
        channels: Vec<ChannelArg>,
        window_ns: Option<i64>,
    ) -> PyResult<u16> {
        let channels = channels
            .into_iter()
            .map(|(name, dim, nullable, bounds)| {
                Ok(ChannelSpec {
                    name,
                    value: ValueSpec::new(dim, nullable, bounds)?,
                })
            })
            .collect::<crate::Result<Vec<_>>>()
            .map_err(to_py)?;
        let class = match window_ns {
            None => NodeClass::Entity,
            Some(window_ns) => NodeClass::Event { window_ns },
        };
        self.inner
            .define_kind(NodeKindSpec {
                name,
                class,
                channels,
            })
            .map(|k| k.0)
            .map_err(to_py)
    }

    #[pyo3(signature = (name, src, dst, attr_dim, nullable=false, bounds=None))]
    fn define_relation(
        &mut self,
        name: &str,
        src: &str,
        dst: &str,
        attr_dim: usize,
        nullable: bool,
        bounds: Option<(f32, f32)>,
    ) -> PyResult<u16> {
        let attrs = ValueSpec::new(attr_dim, nullable, bounds).map_err(to_py)?;
        self.inner
            .define_relation(name, src, dst, attrs)
            .map(|r| r.0)
            .map_err(to_py)
    }

    /// The schema as plain Python data.
    fn schema<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let schema = self.inner.schema();
        let kinds = PyList::empty(py);
        for k in schema.kinds() {
            let d = PyDict::new(py);
            d.set_item("name", &k.name)?;
            d.set_item(
                "window_ns",
                match k.class {
                    NodeClass::Entity => None,
                    NodeClass::Event { window_ns } => Some(window_ns),
                },
            )?;
            let chans = PyList::empty(py);
            for c in &k.channels {
                chans.append((&c.name, c.value.dim, c.value.nullable, c.value.bounds))?;
            }
            d.set_item("channels", chans)?;
            kinds.append(d)?;
        }
        let rels = PyList::empty(py);
        for r in schema.relations() {
            let d = PyDict::new(py);
            d.set_item("name", &r.name)?;
            d.set_item("src", &schema.kind(r.src).name)?;
            d.set_item("dst", &schema.kind(r.dst).name)?;
            d.set_item("attr_dim", r.attrs.dim)?;
            d.set_item("nullable", r.attrs.nullable)?;
            d.set_item("bounds", r.attrs.bounds)?;
            rels.append(d)?;
        }
        let out = PyDict::new(py);
        out.set_item("kinds", kinds)?;
        out.set_item("relations", rels)?;
        Ok(out)
    }

    fn upsert_nodes<'py>(
        &mut self,
        py: Python<'py>,
        kind: &str,
        keys: Vec<String>,
        valid_from: PyReadonlyArray1<'py, i64>,
        valid_to: PyReadonlyArray1<'py, i64>,
        known_time: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (vf, vt, kt) = (
            slice(&valid_from, "valid_from")?,
            slice(&valid_to, "valid_to")?,
            slice(&known_time, "known_time")?,
        );
        let g = &mut self.inner;
        let s = py
            .detach(|| g.upsert_nodes(kind, &keys, vf, vt, kt))
            .map_err(to_py)?;
        stats(py, s)
    }

    fn retract_nodes<'py>(
        &mut self,
        py: Python<'py>,
        kind: &str,
        keys: Vec<String>,
        valid_from: PyReadonlyArray1<'py, i64>,
        valid_to: PyReadonlyArray1<'py, i64>,
        known_time: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (vf, vt, kt) = (
            slice(&valid_from, "valid_from")?,
            slice(&valid_to, "valid_to")?,
            slice(&known_time, "known_time")?,
        );
        let g = &mut self.inner;
        let s = py
            .detach(|| g.retract_nodes(kind, &keys, vf, vt, kt))
            .map_err(to_py)?;
        stats(py, s)
    }

    #[allow(clippy::too_many_arguments)]
    fn observe<'py>(
        &mut self,
        py: Python<'py>,
        kind: &str,
        channel: &str,
        keys: Vec<String>,
        valid_time: PyReadonlyArray1<'py, i64>,
        known_time: PyReadonlyArray1<'py, i64>,
        values: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (v, k) = (
            slice(&valid_time, "valid_time")?,
            slice(&known_time, "known_time")?,
        );
        let x = matrix(&values, "values", keys.len())?;
        let g = &mut self.inner;
        let s = py
            .detach(|| g.observe(kind, channel, &keys, v, k, x))
            .map_err(to_py)?;
        stats(py, s)
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_edges<'py>(
        &mut self,
        py: Python<'py>,
        relation: &str,
        src: Vec<String>,
        dst: Vec<String>,
        valid_from: PyReadonlyArray1<'py, i64>,
        valid_to: PyReadonlyArray1<'py, i64>,
        known_time: PyReadonlyArray1<'py, i64>,
        attrs: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (vf, vt, kt) = (
            slice(&valid_from, "valid_from")?,
            slice(&valid_to, "valid_to")?,
            slice(&known_time, "known_time")?,
        );
        let a = matrix(&attrs, "attrs", src.len())?;
        let g = &mut self.inner;
        let s = py
            .detach(|| g.assert_edges(relation, &src, &dst, vf, vt, kt, a))
            .map_err(to_py)?;
        stats(py, s)
    }

    #[allow(clippy::too_many_arguments)]
    fn retract_edges<'py>(
        &mut self,
        py: Python<'py>,
        relation: &str,
        src: Vec<String>,
        dst: Vec<String>,
        valid_from: PyReadonlyArray1<'py, i64>,
        valid_to: PyReadonlyArray1<'py, i64>,
        known_time: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let (vf, vt, kt) = (
            slice(&valid_from, "valid_from")?,
            slice(&valid_to, "valid_to")?,
            slice(&known_time, "known_time")?,
        );
        let g = &mut self.inner;
        let s = py
            .detach(|| g.retract_edges(relation, &src, &dst, vf, vt, kt))
            .map_err(to_py)?;
        stats(py, s)
    }

    /// Materialize the graph as known at `known_at` (valid at `valid_at`,
    /// default `known_at`). `homogeneous` adds the flattened view; pass
    /// `"reverse"` to include reversed edges.
    #[pyo3(signature = (known_at, valid_at=None, history=None, windows=None, homogeneous=None))]
    fn snapshot<'py>(
        &self,
        py: Python<'py>,
        known_at: i64,
        valid_at: Option<i64>,
        history: Option<HashMap<String, usize>>,
        windows: Option<HashMap<String, i64>>,
        homogeneous: Option<&str>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let add_reverse = match homogeneous {
            None => None,
            Some("forward") => Some(false),
            Some("reverse") => Some(true),
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "homogeneous must be None, 'forward' or 'reverse', got {other:?}"
                )));
            }
        };
        let req = SnapshotRequest {
            known_at,
            valid_at,
            history: history.unwrap_or_default(),
            windows: windows.unwrap_or_default(),
        };
        let g = &self.inner;
        let (snap, homo) = py
            .detach(|| {
                let s = g.snapshot(&req)?;
                let h = add_reverse.map(|r| s.homogeneous(r));
                Ok::<_, Error>((s, h))
            })
            .map_err(to_py)?;
        snapshot_to_py(py, snap, homo)
    }

    fn node_count(&self, kind: &str) -> PyResult<usize> {
        self.inner.node_count(kind).map_err(to_py)
    }

    fn edge_key_count(&self, relation: &str) -> PyResult<usize> {
        self.inner.edge_key_count(relation).map_err(to_py)
    }

    fn observation_count(&self, kind: &str, channel: &str) -> PyResult<usize> {
        self.inner.observation_count(kind, channel).map_err(to_py)
    }

    fn max_known_time(&self) -> Option<i64> {
        self.inner.max_known_time()
    }

    fn save(&self, py: Python<'_>, path: std::path::PathBuf) -> PyResult<()> {
        let g = &self.inner;
        py.detach(|| g.save(&path)).map_err(to_py)
    }

    #[staticmethod]
    fn load(py: Python<'_>, path: std::path::PathBuf) -> PyResult<Self> {
        py.detach(|| TemporalGraph::load(&path))
            .map(|inner| Self { inner })
            .map_err(to_py)
    }

    fn __repr__(&self) -> String {
        let s = self.inner.schema();
        format!(
            "TemporalGraph(kinds={:?}, relations={:?})",
            s.kinds().iter().map(|k| &k.name).collect::<Vec<_>>(),
            s.relations().iter().map(|r| &r.name).collect::<Vec<_>>()
        )
    }
}

fn snapshot_to_py(
    py: Python<'_>,
    snap: Snapshot,
    homo: Option<Homogeneous>,
) -> PyResult<Bound<'_, PyDict>> {
    let out = PyDict::new(py);
    out.set_item("known_at", snap.known_at)?;
    out.set_item("valid_at", snap.valid_at)?;

    let nodes = PyDict::new(py);
    for block in snap.nodes {
        let n = block.keys.len();
        let d = PyDict::new(py);
        d.set_item("keys", block.keys)?;
        d.set_item("since", block.since.into_pyarray(py))?;
        d.set_item("known_time", block.known_time.into_pyarray(py))?;
        let channels = PyDict::new(py);
        for c in block.channels {
            let (k, dim) = (c.history, c.dim);
            let cd = PyDict::new(py);
            cd.set_item("values", c.values.into_pyarray(py).reshape([n, k, dim])?)?;
            cd.set_item("present", c.present.into_pyarray(py).reshape([n, k])?)?;
            cd.set_item(
                "element_present",
                c.element_present
                    .map(|e| e.into_pyarray(py).reshape([n, k, dim]))
                    .transpose()?,
            )?;
            cd.set_item("valid_time", c.valid_time.into_pyarray(py).reshape([n, k])?)?;
            cd.set_item("known_time", c.known_time.into_pyarray(py).reshape([n, k])?)?;
            channels.set_item(c.name, cd)?;
        }
        d.set_item("channels", channels)?;
        nodes.set_item(block.kind, d)?;
    }
    out.set_item("nodes", nodes)?;

    let edges = PyDict::new(py);
    for e in snap.edges {
        let m = e.src.len();
        let d = PyDict::new(py);
        d.set_item("src_kind", e.src_kind)?;
        d.set_item("dst_kind", e.dst_kind)?;
        let mut index = e.src;
        index.extend_from_slice(&e.dst);
        d.set_item("edge_index", index.into_pyarray(py).reshape([2, m])?)?;
        d.set_item("attrs", e.attrs.into_pyarray(py).reshape([m, e.attr_dim])?)?;
        d.set_item(
            "attr_present",
            e.attr_present
                .map(|a| a.into_pyarray(py).reshape([m, e.attr_dim]))
                .transpose()?,
        )?;
        d.set_item("since", e.since.into_pyarray(py))?;
        d.set_item("known_time", e.known_time.into_pyarray(py))?;
        edges.set_item(e.relation, d)?;
    }
    out.set_item("edges", edges)?;

    if let Some(h) = homo {
        let d = PyDict::new(py);
        let m = h.src.len();
        d.set_item("kind_offsets", h.kind_offsets.into_pyarray(py))?;
        d.set_item("node_type", h.node_type.into_pyarray(py))?;
        let mut index = h.src;
        index.extend_from_slice(&h.dst);
        d.set_item("edge_index", index.into_pyarray(py).reshape([2, m])?)?;
        d.set_item("edge_type", h.edge_type.into_pyarray(py))?;
        d.set_item("edge_row", h.edge_row.into_pyarray(py))?;
        d.set_item("indptr", h.indptr.into_pyarray(py))?;
        out.set_item("homogeneous", d)?;
    }
    Ok(out)
}

#[pymodule]
fn _moonrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTemporalGraph>()?;
    m.add("FORMAT_VERSION", crate::persist::FORMAT_VERSION)?;
    m.add("MISSING_TIME", crate::snapshot::MISSING_TIME)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

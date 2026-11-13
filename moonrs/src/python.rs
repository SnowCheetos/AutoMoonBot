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

// ------------------------------------------------------------- backtest

use crate::backtest::{Constraints, Costs, Market, Simulator, StepReport};
use numpy::PyReadonlyArray2 as Ro2;
use std::sync::Arc;

fn panel_f64(a: &Ro2<'_, f64>, name: &str, shape: (usize, usize)) -> PyResult<Vec<f64>> {
    let sh = a.shape();
    if (sh[0], sh[1]) != shape {
        return Err(PyValueError::new_err(format!(
            "{name} has shape {sh:?}, expected {shape:?}"
        )));
    }
    Ok(a.as_array().iter().copied().collect())
}

/// Immutable `[T, S]` market panel shared by any number of simulators.
#[pyclass(name = "Market", module = "automoonbot._moonrs", frozen)]
#[derive(Debug)]
pub struct PyMarket {
    inner: Arc<Market>,
}

#[pymethods]
impl PyMarket {
    #[new]
    #[allow(clippy::too_many_arguments)]
    fn new(
        open: Ro2<'_, f64>,
        close: Ro2<'_, f64>,
        adv_shares: Ro2<'_, f64>,
        sigma: Ro2<'_, f64>,
        tradable: Ro2<'_, bool>,
        listed: Ro2<'_, bool>,
        split: Ro2<'_, f64>,
        dividend: Ro2<'_, f64>,
    ) -> PyResult<Self> {
        let sh = open.shape();
        let shape = (sh[0], sh[1]);
        for (name, a) in [("tradable", &tradable), ("listed", &listed)] {
            let tsh = a.shape();
            if (tsh[0], tsh[1]) != shape {
                return Err(PyValueError::new_err(format!("{name} has the wrong shape")));
            }
        }
        let m = Market {
            periods: shape.0,
            securities: shape.1,
            open: panel_f64(&open, "open", shape)?,
            close: panel_f64(&close, "close", shape)?,
            adv_shares: panel_f64(&adv_shares, "adv_shares", shape)?,
            sigma: panel_f64(&sigma, "sigma", shape)?,
            tradable: tradable.as_array().iter().copied().collect(),
            listed: listed.as_array().iter().copied().collect(),
            split: panel_f64(&split, "split", shape)?,
            dividend: panel_f64(&dividend, "dividend", shape)?,
        };
        m.validate().map_err(to_py)?;
        Ok(Self { inner: Arc::new(m) })
    }

    #[getter]
    fn shape(&self) -> (usize, usize) {
        (self.inner.periods, self.inner.securities)
    }
}

/// Costs and constraints as keyword arguments with conservative defaults.
#[pyclass(name = "Simulator", module = "automoonbot._moonrs")]
#[derive(Debug)]
pub struct PySimulator {
    inner: Simulator,
}

const REPORT_FIELDS: [&str; 15] = [
    "equity_open",
    "equity",
    "log_return",
    "gross",
    "net",
    "turnover",
    "commission",
    "spread_cost",
    "impact_cost",
    "borrow_cost",
    "financing",
    "dividends",
    "projection",
    "delisted",
    "clipped",
];

fn report_values(r: &StepReport) -> [f64; 15] {
    [
        r.equity_open,
        r.equity,
        r.log_return,
        r.gross,
        r.net,
        r.turnover,
        r.commission,
        r.spread_cost,
        r.impact_cost,
        r.borrow_cost,
        r.financing,
        r.dividends,
        r.projection,
        r.delisted as f64,
        r.clipped as f64,
    ]
}

#[pymethods]
impl PySimulator {
    #[new]
    #[pyo3(signature = (
        market, start=0, capital=1.0, *,
        commission_bps=None, half_spread_bps=None, impact_coef=None, max_participation=None,
        borrow_rate=None, cash_rate=None, margin_rate=None, delist_haircut=None,
        periods_per_year=None, long_only=true, max_gross=1.0, max_weight=1.0,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        market: &PyMarket,
        start: usize,
        capital: f64,
        commission_bps: Option<f64>,
        half_spread_bps: Option<f64>,
        impact_coef: Option<f64>,
        max_participation: Option<f64>,
        borrow_rate: Option<f64>,
        cash_rate: Option<f64>,
        margin_rate: Option<f64>,
        delist_haircut: Option<f64>,
        periods_per_year: Option<f64>,
        long_only: bool,
        max_gross: f64,
        max_weight: f64,
    ) -> PyResult<Self> {
        let d = Costs::default();
        let costs = Costs {
            commission_bps: commission_bps.unwrap_or(d.commission_bps),
            half_spread_bps: half_spread_bps.unwrap_or(d.half_spread_bps),
            impact_coef: impact_coef.unwrap_or(d.impact_coef),
            max_participation: max_participation.unwrap_or(d.max_participation),
            borrow_rate: borrow_rate.unwrap_or(d.borrow_rate),
            cash_rate: cash_rate.unwrap_or(d.cash_rate),
            margin_rate: margin_rate.unwrap_or(d.margin_rate),
            delist_haircut: delist_haircut.unwrap_or(d.delist_haircut),
            periods_per_year: periods_per_year.unwrap_or(d.periods_per_year),
        };
        let c = Constraints {
            long_only,
            max_gross,
            max_weight,
        };
        Simulator::new(Arc::clone(&market.inner), costs, c, start, capital)
            .map(|inner| Self { inner })
            .map_err(to_py)
    }

    #[getter]
    fn period(&self) -> usize {
        self.inner.period()
    }

    #[getter]
    fn done(&self) -> bool {
        self.inner.done()
    }

    #[getter]
    fn equity(&self) -> f64 {
        self.inner.equity()
    }

    #[getter]
    fn cash(&self) -> f64 {
        self.inner.cash()
    }

    fn shares<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        self.inner.shares().to_vec().into_pyarray(py)
    }

    fn weights<'py>(&mut self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        self.inner.weights().to_vec().into_pyarray(py)
    }

    /// Execute one step; returns a dict of floats plus `period` and `ruined`.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        target: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let w = slice(&target, "target")?;
        let r = self.inner.step(w).map_err(to_py)?;
        let d = PyDict::new(py);
        for (k, v) in REPORT_FIELDS.iter().zip(report_values(&r)) {
            d.set_item(*k, v)?;
        }
        d.set_item("period", r.period)?;
        d.set_item("ruined", r.ruined)?;
        Ok(d)
    }

    /// Run consecutive steps for each row of `targets` `[N, S]` (stops early
    /// on ruin). Returns `{field: float64[n]}` for the steps taken.
    fn run<'py>(
        &mut self,
        py: Python<'py>,
        targets: Ro2<'py, f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let sh = targets.shape();
        let (n, s) = (sh[0], sh[1]);
        let flat = targets
            .as_slice()
            .map_err(|_| PyValueError::new_err("targets must be C-contiguous float64"))?;
        let sim = &mut self.inner;
        let (rows, periods) = py
            .detach(|| {
                let mut rows: Vec<[f64; 15]> = Vec::with_capacity(n);
                let mut periods = Vec::with_capacity(n);
                for k in 0..n {
                    if sim.done() {
                        break;
                    }
                    let r = sim.step(&flat[k * s..(k + 1) * s])?;
                    rows.push(report_values(&r));
                    periods.push(r.period as i64);
                    if r.ruined {
                        break;
                    }
                }
                Ok::<_, Error>((rows, periods))
            })
            .map_err(to_py)?;
        let d = PyDict::new(py);
        for (j, k) in REPORT_FIELDS.iter().enumerate() {
            let col: Vec<f64> = rows.iter().map(|r| r[j]).collect();
            d.set_item(*k, col.into_pyarray(py))?;
        }
        d.set_item("period", periods.into_pyarray(py))?;
        Ok(d)
    }
}

#[pymodule]
fn _moonrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTemporalGraph>()?;
    m.add_class::<PyMarket>()?;
    m.add_class::<PySimulator>()?;
    m.add("FORMAT_VERSION", crate::persist::FORMAT_VERSION)?;
    m.add("MISSING_TIME", crate::snapshot::MISSING_TIME)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

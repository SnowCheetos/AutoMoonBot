#[cfg(feature = "python")]
use crate::graph::{hetero::HeteroGraph, *};

#[cfg(feature = "python")]
#[pymethods]
impl HeteroGraph {
    #[new]
    pub fn init() -> Self {
        Self::new()
    }

    #[staticmethod]
    pub fn hello_python() -> &'static str {
        "Hello From HeteroGraph"
    }

    #[pyo3(name = "node_count")]
    pub fn node_count_py(&self) -> usize {
        self.node_count()
    }

    #[pyo3(name = "edge_count")]
    pub fn edge_count_py(&self) -> usize {
        self.edge_count()
    }

    #[pyo3(name = "remove_node")]
    pub fn remove_node_py(&mut self, name: String) {
        self.remove_node_by_name(name);
    }

    pub fn add_test_node(&mut self, name: String, value: f64) {
        let node = TestNode::new(name, value);
        let index = self.add_node(Box::new(node));
        self.compute_valid_edges(index);
    }
}

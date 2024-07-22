use crate::nodes::*;

#[derive(Debug)]
#[cfg_attr(feature = "python", pyclass)]
pub struct TestNode {
    pub(super) name: String,
    pub(super) value: f64,
}

impl TestNode {
    pub fn new(name: String, value: f64) -> Self {
        TestNode { name, value }
    }

    pub fn value(&self) -> f64 {
        self.value
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl TestNode {
    #[new]
    pub fn init(name: String, value: f64) -> Self {
        Self::new(name, value)
    }

    #[pyo3(name = "value")]
    pub fn value_py(&self) -> f64 {
        self.value
    }
}
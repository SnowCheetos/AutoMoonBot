use crate::nodes::*;

#[derive(Debug)]
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

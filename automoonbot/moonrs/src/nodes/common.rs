use crate::nodes::*;

#[derive(Debug)]
pub struct TestNode {
    pub(super) name: String,
    pub(super) value: f64,
    pub(super) buffer: TemporalDeque<f64>,
}

impl TestNode {
    pub fn new(name: String, value: f64, capacity: usize) -> Self {
        TestNode {
            name,
            value,
            buffer: TemporalDeque::new(capacity),
        }
    }

    pub fn value(&self) -> f64 {
        self.value
    }
}

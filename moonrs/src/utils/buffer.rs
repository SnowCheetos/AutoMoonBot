use crate::utils::{aggregate::Aggregate, queue::FixedSizeQueue, *};
use nalgebra::MatrixXx4;

#[derive(Debug, Clone)]
pub struct AggregateBuffer {
    queue: FixedSizeQueue<Aggregate, Instant>,
}

impl AggregateBuffer {
    pub fn new(capacity: usize) -> Self {
        AggregateBuffer {
            queue: FixedSizeQueue::new(capacity),
        }
    }

    pub fn push(&mut self, item: Aggregate) {
        let index = item.timestamp().clone();
        self.queue.push(index, item)
    }

    pub fn last(&self) -> Option<Aggregate> {
        self.queue.back()
    }

    pub fn get(&self, row: usize) -> Option<Aggregate> {
        self.queue.get(row)
    }

    pub fn loc(&self, index: Instant) -> Option<Aggregate> {
        self.queue.loc(index)
    }

    pub fn between(&self, a: Instant, b: Instant) -> Option<Vec<&Aggregate>> {
        self.queue.between(&a, &b)
    }

    pub fn slice(&self, a: usize, b: usize) -> Option<Vec<&Aggregate>> {
        self.queue.slice(a, b)
    }

    pub fn vec(&self) -> Vec<&Aggregate> {
        self.queue.vec()
    }

    pub fn pmat(&self) -> Option<MatrixXx4<f64>> {
        let vec = self.vec();
        if vec.is_empty() {
            None
        } else {
            let data: Vec<f64> = vec
                .iter()
                .flat_map(|item| item.vec4().as_slice().to_vec())
                .collect();
            Some(MatrixXx4::from_row_slice(&data))
        }
    }
}

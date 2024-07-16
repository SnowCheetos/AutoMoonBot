use crate::utils::{queue::FixedSizeQueue, aggregate::Aggregate};

#[derive(Debug, Clone)]
pub struct History {
    queue: FixedSizeQueue<Aggregate>,
}

impl History {
    pub fn new(queue_size: usize) -> Self {
        History {
            queue: FixedSizeQueue::new(queue_size),
        }
    }

    pub fn push(&mut self, data: Aggregate) {
        self.queue.push(data)
    }

    pub fn last(&self) -> Option<Aggregate> {
        self.queue.back()
    }
}

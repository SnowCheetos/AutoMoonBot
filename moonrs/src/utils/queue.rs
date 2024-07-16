use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct FixedSizeQueue<T> {
    deque: VecDeque<T>,
    capacity: usize,
}

impl<T> FixedSizeQueue<T>
where
    T: Clone,
{
    pub fn new(capacity: usize) -> Self {
        FixedSizeQueue {
            deque: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, item: T) {
        if self.deque.len() == self.capacity {
            self.deque.pop_front();
        }
        self.deque.push_back(item);
    }

    pub fn back(&self) -> Option<T> {
        self.deque.back().cloned()
    }

    pub fn get_elements(&self) -> Vec<&T> {
        self.deque.iter().collect()
    }
}

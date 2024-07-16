use indexmap::IndexMap;
use std::{collections::VecDeque, hash::Hash};

#[derive(Debug, Clone)]
pub struct FixedSizeDeque<T, Ix> {
    deque: VecDeque<T>,
    index: IndexMap<Ix, usize>,
    capacity: usize,
}

impl<T, Ix> FixedSizeDeque<T, Ix>
where
    T: Clone,
    Ix: Hash + Eq + PartialOrd + Clone,
{
    pub fn new(capacity: usize) -> Self {
        FixedSizeDeque {
            index: IndexMap::with_capacity(capacity),
            deque: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, id: Ix, item: T) {
        if self.deque.len() == self.capacity {
            self.deque.pop_front();
            self.index.shift_remove_index(0);
        }
        self.deque.push_back(item);
        self.index.insert(id, self.deque.len() - 1);
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.deque.get(index)
    }

    pub fn slice(&self, a: usize, b: usize) -> Option<Vec<&T>> {
        if a >= b || b > self.deque.len() {
            None
        } else {
            Some(self.deque.range(a..b).collect())
        }
    }

    pub fn between(&self, a: &Ix, b: &Ix) -> Option<Vec<&T>> {
        let start = self.index.get(a)?;
        let end = self.index.get(b)?;
        if start >= end || *end >= self.deque.len() {
            None
        } else {
            Some(self.deque.range(*start..=*end).collect())
        }
    }

    pub fn len(&self) -> usize {
        self.deque.len()
    }
}

use indexmap::IndexMap;
use std::{collections::{VecDeque, vec_deque::Iter}, hash::Hash};

#[derive(Debug, Clone)]
pub struct FixedSizeQueue<T, Ix> {
    deque: VecDeque<T>,
    index: IndexMap<Ix, usize>,
    capacity: usize,
}

impl<T, Ix> FixedSizeQueue<T, Ix>
where
    T: Clone,
    Ix: Hash + Eq + PartialOrd,
{
    pub fn new(capacity: usize) -> Self {
        FixedSizeQueue {
            deque: VecDeque::with_capacity(capacity),
            index: IndexMap::new(),
            capacity,
        }
    }

    pub fn push(&mut self, index: Ix, item: T) {
        if self.deque.len() == self.capacity {
            if let Some(_) = self.deque.pop_front() {
                self.index.shift_remove_index(0);
            }
        }
        self.deque.push_back(item);
        self.index.insert(index, self.deque.len() - 1);
    }

    pub fn back(&self) -> Option<T> {
        self.deque.back().cloned()
    }

    pub fn get(&self, row: usize) -> Option<T> {
        self.deque.get(row).cloned()
    }

    pub fn loc(&self, index: Ix) -> Option<T> {
        if let Some(&row) = self.index.get(&index) {
            self.deque.get(row).cloned()
        } else {
            None
        }
    }

    pub fn between(&self, a: &Ix, b: &Ix) -> Option<Vec<&T>> {
        if a > b {
            return None;
        }

        let start = self.index.get_index_of(a);
        let end = self.index.get_index_of(b);

        match (start, end) {
            (Some(start), Some(end)) if start <= end => Some(
                self.index
                    .iter()
                    .skip(start)
                    .take(end - start + 1)
                    .filter_map(|(_, &pos)| self.deque.get(pos))
                    .collect(),
            ),
            _ => None,
        }
    }

    pub fn slice(&self, a: usize, b: usize) -> Option<Vec<&T>> {
        if a >= b || b > self.deque.len() {
            None
        } else {
            Some(self.deque.iter().skip(a).take(b - a).collect())
        }
    }

    pub fn vec(&self) -> Vec<&T> {
        self.deque.iter().collect()
    }

    pub fn iter(&self) -> Option<Iter<'_, T>> {
        if self.deque.is_empty() {
            None
        } else {
            Some(self.deque.iter())
        }
    }
}

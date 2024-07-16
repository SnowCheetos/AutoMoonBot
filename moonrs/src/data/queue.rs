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
    Ix: Hash + Eq + PartialOrd,
{
    pub fn new(capacity: usize) -> Self {
        FixedSizeDeque {
            index: IndexMap::with_capacity(capacity),
            deque: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, id: Ix, item: T) -> bool {
        if self.index.contains_key(&id) {
            return false;
        }
        if self.deque.len() == self.capacity {
            self.deque.pop_front();
            self.index.shift_remove_index(0);
        }
        self.deque.push_back(item);
        self.index.insert(id, self.deque.len() - 1);
        true
    }

    pub fn len(&self) -> usize {
        self.deque.len()
    }

    pub fn first(&self) -> Option<&T> {
        self.deque.front()
    }

    pub fn last(&self) -> Option<&T> {
        self.deque.back()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.deque.get(index)
    }

    pub fn loc(&self, key: &Ix) -> Option<&T> {
        if let Some(&index) = self.index.get(key) {
            self.get(index)
        } else {
            None
        }
    }

    pub fn range(&self, a: usize, b: usize) -> Option<Vec<&T>> {
        if a >= b || b > self.deque.len() {
            None
        } else {
            Some(self.deque.range(a..b).collect())
        }
    }

    pub fn between(&self, i: &Ix, j: &Ix) -> Option<Vec<&T>> {
        let start = self
            .index
            .iter()
            .find(|(key, _)| *key >= i)
            .map(|(_, &index)| index)?;
        let end = self
            .index
            .iter()
            .rev()
            .find(|(key, _)| *key <= j)
            .map(|(_, &index)| index)?;
        self.range(start, end)
    }

    pub fn slice(&mut self, a: usize, b: usize) -> Option<&[T]> {
        self.deque.make_contiguous();
        if let (slice, &[]) = self.deque.as_slices() {
            Some(&slice[a..b])
        } else {
            None
        }
    }
}

// pub fn mat(&self) -> DMatrix<f64> {
//     let rows = self.deque.len();
//     let cols = self
//         .deque
//         .front()
//         .map_or(0, |item| item.into_iter().count());
//     let iter = self.deque.iter().flat_map(|item| item.into_iter());
//     DMatrix::from_iterator(rows, cols, iter)
// }

use crate::data::{aggregate::*, buffer::*, *};
use indexmap::IndexMap;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct TemporalDeque<T> {
    deque: VecDeque<T>,
    index: IndexMap<Instant, usize>,
    capacity: usize,
}

impl<T> DataBuffer for TemporalDeque<T>
where
    T: Clone,
{
    fn new(capacity: usize) -> Self {
        TemporalDeque {
            index: IndexMap::with_capacity(capacity),
            deque: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn len(&self) -> usize {
        self.deque.len()
    }

    fn empty(&self) -> bool {
        self.deque.is_empty()
    }

    fn clear(&mut self) {
        self.deque.clear()
    }
}

impl<T> RingIndexBuffer<Instant, T> for TemporalDeque<T>
where
    T: Clone,
{
    fn push(&mut self, id: Instant, item: T) -> bool {
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

    fn first(&self) -> Option<&T> {
        self.deque.front()
    }

    fn last(&self) -> Option<&T> {
        self.deque.back()
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.deque.get(index)
    }

    fn loc(&self, key: &Instant) -> Option<&T> {
        if let Some(&index) = self.index.get(key) {
            self.get(index)
        } else {
            None
        }
    }

    fn range(&self, a: usize, b: usize) -> Option<Vec<&T>> {
        if a >= b || b > self.deque.len() {
            None
        } else {
            Some(self.deque.range(a..b).collect())
        }
    }

    fn between(&self, i: &Instant, j: &Instant) -> Option<Vec<&T>> {
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

    fn slice(&mut self, a: usize, b: usize) -> Option<&[T]> {
        self.deque.make_contiguous();
        if let (slice, &[]) = self.deque.as_slices() {
            Some(&slice[a..b])
        } else {
            None
        }
    }
}

impl BlockRingIndexBuffer<Instant, Aggregate, f64> for TemporalDeque<Aggregate> {
    fn rows(&self) -> usize {
        self.deque.len()
    }

    fn cols(&self) -> usize {
        self.deque
            .front()
            .map_or(0, |item| item.into_iter().count())
    }

    fn mat(&self) -> Option<na::DMatrix<f64>> {
        let rows = self.deque.len();
        if rows == 0 {
            return None
        }
        let cols = self
            .deque
            .front()
            .map_or(0, |item| item.into_iter().count());
        let iter = self.deque.iter().flat_map(|item| item.into_iter());
        Some(na::DMatrix::from_iterator(rows, cols, iter))
    }
}

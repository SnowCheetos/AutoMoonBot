use crate::data::*;
use aggregates::Aggregates;
use indexmap::IndexMap;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct TimeSeries<T> {
    deque: VecDeque<T>,
    index: IndexMap<Instant, usize>,
    capacity: usize,
}

impl<T> DataBuffer for TimeSeries<T>
where
    T: Clone,
{
    fn new(capacity: usize) -> Self {
        TimeSeries {
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

impl<T> RingIndexBuffer<Instant, T> for TimeSeries<T>
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

    fn to_vec(&self) -> Vec<&T> {
        self.deque.range(..).collect()
    }

    fn range(&self, a: usize, b: usize) -> Option<Vec<&T>> {
        if a >= b || b > self.deque.len() {
            None
        } else {
            Some(self.deque.range(a..b).collect())
        }
    }

    fn between(&self, i: &Instant, j: &Instant) -> Option<Vec<&T>> {
        let a = self
            .index
            .iter()
            .find(|(key, _)| *key >= i)
            .map(|(_, &index)| index)?;
        let b = self
            .index
            .iter()
            .rev()
            .find(|(key, _)| *key <= j)
            .map(|(_, &index)| index)?;
        if a >= b || b > self.deque.len() {
            None
        } else {
            Some(self.deque.range(a..=b).collect())
        }
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

impl<T> BlockRingIndexBuffer<Instant, T, f64> for TimeSeries<T>
where
    T: Aggregates,
{
    fn rows(&self) -> usize {
        self.deque.len()
    }

    fn cols(&self) -> usize {
        self.deque
            .front()
            .map_or(0, |item| item.into_iter().count())
    }

    fn mat(&self) -> Option<na::DMatrix<f64>> {
        let rows = self.rows();
        let cols = self.cols();
        if rows == 0 || cols == 0 {
            return None;
        }
        let data: Vec<f64> = self.deque.iter().flat_map(|item| item.to_vec()).collect();
        Some(na::DMatrix::from_row_slice(rows, cols, &data))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let buffer: TimeSeries<i32> = TimeSeries::new(5);
        assert_eq!(buffer.capacity, 5);
        assert!(buffer.deque.is_empty());
        assert!(buffer.index.is_empty());
    }

    #[test]
    fn test_push_and_get() {
        let mut buffer = TimeSeries::new(3);
        let now = Instant::now();

        assert!(buffer.push(now, 1));
        assert!(buffer.push(now + Duration::new(1, 0), 2));
        assert!(buffer.push(now + Duration::new(2, 0), 3));

        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.get(0), Some(&1));
        assert_eq!(buffer.get(1), Some(&2));
        assert_eq!(buffer.get(2), Some(&3));

        assert!(buffer.push(now + Duration::new(3, 0), 4));
        assert_eq!(buffer.len(), 3);
        assert_eq!(buffer.get(0), Some(&2));
    }

    #[test]
    fn test_first_last() {
        let mut buffer = TimeSeries::new(3);
        let now = Instant::now();

        buffer.push(now, 1);
        buffer.push(now + Duration::new(1, 0), 2);
        buffer.push(now + Duration::new(2, 0), 3);

        assert_eq!(buffer.first(), Some(&1));
        assert_eq!(buffer.last(), Some(&3));
    }

    #[test]
    fn test_loc() {
        let mut buffer = TimeSeries::new(3);
        let now = Instant::now();

        buffer.push(now, 1);
        buffer.push(now + Duration::new(1, 0), 2);
        buffer.push(now + Duration::new(2, 0), 3);

        assert_eq!(buffer.loc(&(now + Duration::new(1, 0))), Some(&2));
        assert_eq!(buffer.loc(&(now + Duration::new(3, 0))), None);
    }

    #[test]
    fn test_range() {
        let mut buffer = TimeSeries::new(3);
        let now = Instant::now();

        buffer.push(now, 1);
        buffer.push(now + Duration::new(1, 0), 2);
        buffer.push(now + Duration::new(2, 0), 3);

        let range = buffer.range(0, 2).unwrap();
        assert_eq!(range, vec![&1, &2]);
    }

    #[test]
    fn test_between() {
        let mut buffer = TimeSeries::new(3);
        let now = Instant::now();

        buffer.push(now, 1);
        buffer.push(now + Duration::new(1, 0), 2);
        buffer.push(now + Duration::new(2, 0), 3);

        let between = buffer
            .between(&(now + Duration::new(1, 0)), &(now + Duration::new(2, 0)))
            .unwrap();
        assert_eq!(between, vec![&2, &3]);
    }

    #[test]
    fn test_clear() {
        let mut buffer = TimeSeries::new(3);
        let now = Instant::now();

        buffer.push(now, 1);
        buffer.push(now + Duration::new(1, 0), 2);
        buffer.push(now + Duration::new(2, 0), 3);

        buffer.clear();
        assert!(buffer.empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_aggregate_mat() {
        let mut buffer = TimeSeries::new(3);
        let now = Instant::now();
        let span = Duration::new(60, 0);
        let next = now + span;

        let aggregate1 = PriceAggregate::new(now, span, true, 1.0, 2.0, 0.5, 1.5, 100.0);
        let aggregate2 = PriceAggregate::new(next, span, false, 1.1, 2.1, 0.6, 1.6, 200.0);

        let mat = buffer.mat();
        assert!(mat.is_none(), "did not none for matrix");

        buffer.push(aggregate1.timestamp(), aggregate1);
        buffer.push(aggregate2.timestamp(), aggregate2);

        let mat = buffer.mat();
        assert!(mat.as_ref().is_some_and(|mat| mat.shape() == (2, 5)), "returned none for matrix");

        let mat = mat.unwrap();
        assert_eq!(mat[(0, 0)], 1.0, "unmatched value at (0, 0)");
        assert_eq!(mat[(0, 1)], 2.0, "unmatched value at (0, 1)");
        assert_eq!(mat[(0, 2)], 0.5, "unmatched value at (0, 2)");
        assert_eq!(mat[(0, 3)], 1.5, "unmatched value at (0, 3)");
        assert_eq!(mat[(0, 4)], 100.0, "unmatched value at (0, 4)");
        assert_eq!(mat[(1, 0)], 1.1, "unmatched value at (1, 0)");
        assert_eq!(mat[(1, 1)], 2.1, "unmatched value at (1, 1)");
        assert_eq!(mat[(1, 2)], 0.6, "unmatched value at (1, 2)");
        assert_eq!(mat[(1, 3)], 1.6, "unmatched value at (1, 3)");
        assert_eq!(mat[(1, 4)], 200.0, "unmatched value at (1, 4)");
    }
}

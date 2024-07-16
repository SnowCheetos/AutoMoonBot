use crate::data::Hash;
use crate::na::{DMatrix, DVector, Scalar};

pub trait Buffer: Clone {
    fn new(capacity: usize) -> Self;
    fn len(&self) -> usize;
    fn empty(&self) -> bool;
    fn clear(&mut self);
}

pub trait MapBuffer<K, V>: Buffer
where
    K: Clone + Hash + Eq,
{
    fn loc(&self, index: &K) -> Option<&V>;
    fn put(&mut self, key: K, item: V) -> bool;
    fn pop(&mut self, key: K) -> Option<&V>;
}

pub trait RingIndexBuffer<Ix, T>: Buffer
where
    Ix: Clone + Hash + Eq + PartialOrd,
{
    fn first(&self) -> Option<&T>;
    fn last(&self) -> Option<&T>;
    fn loc(&self, index: &T) -> Option<&T>;
    fn get(&self, index: usize) -> Option<&T>;
    fn push(&mut self, index: Ix, item: T) -> bool;
    fn range(&self, a: usize, b: usize) -> Option<Vec<&T>>;
    fn between(&self, i: &Ix, j: &Ix) -> Option<Vec<&T>>;
    fn slice(&mut self, a: usize, b: usize) -> Option<&[T]>;
}

pub trait BlockRingIndexBuffer<Ix, T, N>: RingIndexBuffer<Ix, T>
where
    N: Copy + Scalar,
    T: Copy + IntoIterator<Item = N>,
    Ix: Clone + Hash + Eq + PartialOrd,
{
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn row(&self, r: usize) -> DVector<N>;
    fn col(&self, c: usize) -> DVector<N>;
    fn mat(&self) -> DMatrix<N>;
}

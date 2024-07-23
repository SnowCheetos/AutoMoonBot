use crate::data::*;

pub trait DataBuffer {
    fn new(capacity: usize) -> Self;
    fn len(&self) -> usize;
    fn empty(&self) -> bool;
    fn clear(&mut self);
}

pub trait MapBuffer<K, V>: DataBuffer
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    fn get(&self, index: &K) -> Option<&V>;
    fn put(&mut self, key: K, item: V) -> bool;
    fn pop(&mut self, key: K) -> Option<&V>;
}

pub trait RingIndexBuffer<Ix, T>: DataBuffer
where
    Ix: Clone + Hash + Eq + PartialOrd,
    T: Clone,
{
    fn to_vec(&self) -> Vec<&T>;
    fn first(&self) -> Option<&T>;
    fn last(&self) -> Option<&T>;
    fn loc(&self, index: &Ix) -> Option<&T>;
    fn get(&self, index: usize) -> Option<&T>;
    fn push(&mut self, index: Ix, item: T) -> bool;
    fn range(&self, a: usize, b: usize) -> Option<Vec<&T>>;
    fn between(&self, i: &Ix, j: &Ix) -> Option<Vec<&T>>;
    fn slice(&mut self, a: usize, b: usize) -> Option<&[T]>;
}

pub trait BlockRingIndexBuffer<Ix, T, N>: RingIndexBuffer<Ix, T>
where
    N: Copy + na::Scalar,
    T: Clone + IntoIterator<Item = N>,
    Ix: Clone + Hash + Eq + PartialOrd,
{
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn mat(&self) -> Option<na::DMatrix<N>>;
}

// pub trait TimeSeriesBuffer<T>: BlockRingIndexBuffer<Instant, T, f64>
// where
//     T: Clone + IntoIterator<Item = f64>,
// {
//     fn autocorrelation(&self, lag: usize) -> na::DVector<f64>;
//     fn zscore(&self, period: usize) -> na::DVector<f64>;
//     fn skew(&self, period: usize) -> na::DVector<f64>;
//     fn kurtosis(&self, period: usize) -> na::DVector<f64>;
//     fn momentum(&self, period: usize) -> na::DVector<f64>;
// }
use crate::edges::*;

pub trait MutualDynEdge<S, T, Ix, X, Y>: StaticEdge
where
    S: DynamicNode<Ix, Y>,
    T: DynamicNode<Ix, X>,
    Ix: Clone + Hash + Eq + PartialOrd,
    X: Clone,
    Y: Clone,
{
    fn covariance(&self) -> Option<f64>;
    fn correlation(&self) -> Option<f64>;
    fn compute_covariance(&self, src: &S, tgt: &T) -> Option<f64>;
    fn compute_correlation(&self, src: &S, tgt: &T) -> Option<f64>;
    fn update(&mut self, src: &S, tgt: &T);
}

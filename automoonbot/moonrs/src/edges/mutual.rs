use crate::edges::*;

pub trait MutualDynEdge<S, T, X, Y>
where
    S: DynamicNode<Y>,
    T: DynamicNode<X>,
    X: Clone,
    Y: Clone,
{
    fn index_match(&self, src: &S, tgt: &T) -> Vec<usize>;
    fn covariance(&self, src: &S, tgt: &T) -> f64;
    fn correlation(&self, src: &S, tgt: &T) -> f64;
}

mod tests {
    use super::*;

    #[test]
    fn test() {}
}

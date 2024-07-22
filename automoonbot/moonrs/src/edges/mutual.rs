use crate::edges::*;

pub trait MutualDynEdge<S, T, Ix, X, Y>: StaticEdge
where
    S: DynamicNode<Ix, Y>,
    T: DynamicNode<Ix, X>,
    Ix: Clone + Hash + Eq + PartialOrd,
    X: Clone,
    Y: Clone,
{
    fn covariance(&self, src: &S, tgt: &T) -> f64;
    fn correlation(&self, src: &S, tgt: &T) -> f64;
}

impl MutualDynEdge<TestNode, TestNode, Instant, f64, f64> for TestEdge {
    fn covariance(&self, src: &TestNode, tgt: &TestNode) -> f64 {
        todo!()
    }

    fn correlation(&self, src: &TestNode, tgt: &TestNode) -> f64 {
        todo!()
    }
}

mod tests {
    use super::*;

    #[test]
    fn test() {}
}

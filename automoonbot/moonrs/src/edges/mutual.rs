use crate::edges::*;

pub trait MutualDynEdge<S, T, Ix, X, Y>: StaticEdge
where
    S: DynamicNode<Ix, Y>,
    T: DynamicNode<Ix, X>,
    Ix: Clone + Hash + Eq + PartialOrd,
    X: Clone,
    Y: Clone,
{
    fn covariance(&self, src: &S, tgt: &T) -> Option<f64>;
    fn correlation(&self, src: &S, tgt: &T) -> f64;
}

impl MutualDynEdge<TestNode, TestNode, Instant, f64, f64> for TestEdge {
    fn covariance(&self, src: &TestNode, tgt: &TestNode) -> Option<f64> {
        let vec1 = src.to_vec();
        let vec2 = tgt.to_vec();
        if vec1.len() == vec2.len() && vec1.len() > 0 && vec2.len() > 0 {
            Some(vec1.covariance(vec2))
        } else {
            None
        }
    }

    fn correlation(&self, src: &TestNode, tgt: &TestNode) -> f64 {
        0.0
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_covariance() {
        let name1 = "Node1".to_owned();
        let name2 = "Node2".to_owned();
        let mut node1 = TestNode::new(name1.clone(), 0.0, 5);
        let mut node2 = TestNode::new(name2.clone(), 0.0, 5);
        assert!(node1.empty() && node2.empty());

        let now = Instant::now();
        let value1 = 1.0;
        let value2 = 2.0;
        node1.update(now, value1);
        node2.update(now, value2);

        let later = Instant::now();
        node1.update(later, value2);
        node2.update(later, value1);

        let index1 = NodeIndex::new(0);
        let index2 = NodeIndex::new(1);
        let edge = TestEdge::new(index1, index2, &node1, &node2);

        assert!(edge.covariance(&node1, &node2).is_some_and(|val| val < 0.0));
    }
}

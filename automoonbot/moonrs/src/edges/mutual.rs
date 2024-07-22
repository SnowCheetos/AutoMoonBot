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

impl MutualDynEdge<TestNode, TestNode, Instant, f64, f64> for TestEdge {
    fn covariance(&self) -> Option<f64> {
        self.covariance
    }

    fn correlation(&self) -> Option<f64> {
        self.correlation
    }

    fn compute_covariance(&self, src: &TestNode, tgt: &TestNode) -> Option<f64> {
        let vec1 = src.to_vec();
        let vec2 = tgt.to_vec();
        if vec1.len() == vec2.len() && vec1.len() > 1 && vec2.len() > 1 {
            Some(vec1.covariance(vec2))
        } else {
            None
        }
    }

    fn compute_correlation(&self, src: &TestNode, tgt: &TestNode) -> Option<f64> {
        let vec1 = src.to_vec();
        let vec2 = tgt.to_vec();
        if vec1.len() == vec2.len() && vec1.len() > 1 && vec2.len() > 1 {
            let std1 = vec1.clone().population_std_dev();
            let std2 = vec2.clone().population_std_dev();
            let cov = vec1.population_covariance(vec2);
            Some(cov / (std1 * std2))
        } else {
            None
        }
    }

    fn update(&mut self, src: &TestNode, tgt: &TestNode) {
        if let (Some(covariance), Some(correlation)) = (
            self.compute_covariance(src, tgt),
            self.compute_correlation(src, tgt),
        ) {
            self.covariance = Some(covariance);
            self.correlation = Some(correlation);
        }
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_edge() {
        let src_name = "src_node".to_owned();
        let tgt_name = "tgt_node".to_owned();
        let mut src_node = TestNode::new(src_name.clone(), 0.0, 5);
        let mut tgt_node = TestNode::new(tgt_name.clone(), 0.0, 5);
        assert!(src_node.empty() && tgt_node.empty());

        let now = Instant::now();
        let value1 = 1.0;
        let value2 = 2.0;
        src_node.update(now, value1);
        tgt_node.update(now, value2);

        let later = Instant::now();
        src_node.update(later, value2);
        tgt_node.update(later, value1);

        let src_index = NodeIndex::new(0);
        let tgt_index = NodeIndex::new(1);
        let mut edge = TestEdge::new(src_index, tgt_index, &src_node, &tgt_node);

        assert!(edge.covariance().is_none());
        assert!(edge.correlation().is_none());

        edge.update(&src_node, &tgt_node);
        assert!(edge.covariance().is_some_and(|val| val < 0.0));
        assert!(edge.correlation().is_some_and(|val| val < 0.0));

        assert_eq!(edge.covariance().unwrap(), -0.5);
        assert_eq!(edge.correlation().unwrap(), -1.0);
    }
}

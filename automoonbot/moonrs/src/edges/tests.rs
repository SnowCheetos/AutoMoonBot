use crate::edges::*;

#[derive(Debug)]
pub struct TestEdge {
    pub(super) src_index: NodeIndex,
    pub(super) tgt_index: NodeIndex,
    pub(super) value: f64,
    pub(super) covariance: Option<na::DMatrix<f64>>,
    pub(super) correlation: Option<na::DMatrix<f64>>,
}

impl TestEdge {
    pub fn new(
        src_index: NodeIndex,
        tgt_index: NodeIndex,
        src_node: &dyn StaticNode,
        tgt_node: &dyn StaticNode,
    ) -> Self {
        let value = Self::difference(src_node, tgt_node);
        TestEdge {
            src_index,
            tgt_index,
            value,
            covariance: None,
            correlation: None,
        }
    }

    pub fn try_new(
        src_index: NodeIndex,
        tgt_index: NodeIndex,
        src_node: &dyn StaticNode,
        tgt_node: &dyn StaticNode,
    ) -> Option<Self> {
        if src_index == tgt_index {
            return None;
        }
        let source = src_node.as_any().downcast_ref::<TestNode>();
        let target = tgt_node.as_any().downcast_ref::<TestNode>();
        if source.is_none() || target.is_none() {
            return None;
        }
        let (source, target) = (source.unwrap(), target.unwrap());
        let value = Self::difference(source, target);
        if value > 0.0 {
            Some(TestEdge {
                src_index,
                tgt_index,
                value,
                covariance: None,
                correlation: None,
            })
        } else {
            None
        }
    }

    pub fn difference(src_node: &dyn StaticNode, tgt_node: &dyn StaticNode) -> f64 {
        (src_node.value().unwrap() - tgt_node.value().unwrap()).abs()
    }
}

impl StaticEdge for TestEdge {
    fn value(&self) -> f64 {
        self.value
    }

    fn src_index(&self) -> &NodeIndex {
        &self.src_index
    }

    fn tgt_index(&self) -> &NodeIndex {
        &self.tgt_index
    }
}

impl MutualDynEdge<TestNode, TestNode, Instant, f64, f64> for TestEdge {
    fn covariance(&self) -> Option<&na::DMatrix<f64>> {
        self.covariance.as_ref()
    }

    fn correlation(&self) -> Option<&na::DMatrix<f64>> {
        self.correlation.as_ref()
    }

    fn compute_covariance(&self, src: &TestNode, tgt: &TestNode) -> Option<na::DMatrix<f64>> {
        let vec1 = src.to_vec();
        let vec2 = tgt.to_vec();
        if vec1.len() == vec2.len() && vec1.len() > 1 && vec2.len() > 1 {
            todo!()
        } else {
            None
        }
    }

    fn compute_correlation(&self, src: &TestNode, tgt: &TestNode) -> Option<na::DMatrix<f64>> {
        let vec1 = src.to_vec();
        let vec2 = tgt.to_vec();
        if vec1.len() == vec2.len() && vec1.len() > 1 && vec2.len() > 1 {
            todo!()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_edge() {
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
        // assert!(edge.covariance().is_some_and(|val| val < 0.0));
        // assert!(edge.correlation().is_some_and(|val| val < 0.0));

        // assert_eq!(edge.covariance().unwrap(), -0.5);
        // assert_eq!(edge.correlation().unwrap(), -1.0);
    }

    #[test]
    fn test_mutual_edge() {
        let src_name = "src_node".to_owned();
        let tgt_name = "tgt_node".to_owned();
        let src_value = 1.0;
        let tgt_value = 2.0;
        let src_node = TestNode::new(src_name.clone(), src_value, 1);
        let tgt_node = TestNode::new(tgt_name.clone(), tgt_value, 1);

        let src_index = NodeIndex::new(0);
        let tgt_index = NodeIndex::new(1);

        let edge = TestEdge::new(src_index, tgt_index, &src_node, &tgt_node);
        let edge_value = 1.0;
        assert_eq!(edge.value(), edge_value);
        assert_eq!(*edge.src_index(), src_index);
        assert_eq!(*edge.tgt_index(), tgt_index);
    }
}

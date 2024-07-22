use crate::edges::*;

pub trait StaticEdge: Send + Sync {
    fn value(&self) -> f64;
    fn src_index(&self) -> &NodeIndex;
    fn tgt_index(&self) -> &NodeIndex;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge() {
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
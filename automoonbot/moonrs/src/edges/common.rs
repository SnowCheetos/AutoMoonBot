use crate::edges::*;

#[derive(Debug)]
#[cfg_attr(feature = "python", pyclass)]
pub struct TestEdge {
    pub(super) src_index: NodeIndex,
    pub(super) tgt_index: NodeIndex,
    pub(super) value: f64,
}

impl TestEdge {
    pub fn new(src_index: NodeIndex, tgt_index: NodeIndex, src_node: &dyn StaticNode, tgt_node: &dyn StaticNode) -> Self {
        let value = Self::difference(src_node, tgt_node);
        TestEdge {
            src_index,
            tgt_index,
            value,
        }
    }

    pub fn difference(src_node: &dyn StaticNode, tgt_node: &dyn StaticNode) -> f64 {
        (src_node.value() - tgt_node.value()).abs()
    }
}
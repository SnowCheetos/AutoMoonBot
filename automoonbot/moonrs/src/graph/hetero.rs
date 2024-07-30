use crate::graph::*;

#[derive(Default)]
#[cfg_attr(feature = "python", pyclass(subclass))]
pub struct HeteroGraph {
    pub(super) graph: StableDiGraph<NodeType, EdgeType>,
    pub(super) node_memo: HashMap<String, NodeIndex>,
    pub(super) edge_memo: HashMap<(NodeIndex, NodeIndex), EdgeIndex>,
}

impl HeteroGraph {
    pub fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            node_memo: HashMap::new(),
            edge_memo: HashMap::new(),
        }
    }

    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    pub fn get_node(&self, index: NodeIndex) -> Option<&NodeType> {
        self.graph.node_weight(index)
    }

    pub fn get_edge(&self, index: EdgeIndex) -> Option<&EdgeType> {
        self.graph.edge_weight(index)
    }

    pub fn get_node_index(&self, name: String) -> Option<&NodeIndex> {
        self.node_memo.get(&name)
    }

    pub fn get_edge_index(&self, src: NodeIndex, tgt: NodeIndex) -> Option<&EdgeIndex> {
        self.edge_memo.get(&(src, tgt))
    }

    pub fn get_node_by_name(&self, name: String) -> Option<&NodeType> {
        let index = self.node_memo.get(&name)?;
        self.get_node(*index)
    }

    pub fn get_edge_by_pair(&self, src: NodeIndex, tgt: NodeIndex) -> Option<&EdgeType> {
        let index = self.edge_memo.get(&(src, tgt))?;
        self.get_edge(*index)
    }

    pub fn get_edge_by_names(&self, src: String, tgt: String) -> Option<&EdgeType> {
        let (source, target) = (self.get_node_index(src)?, self.get_node_index(tgt)?);
        self.get_edge_by_pair(*source, *target)
    }

    pub fn add_node(&mut self, node: NodeType) -> NodeIndex {
        let name = node.name().to_string();
        let index = self.graph.add_node(node);
        self.node_memo.entry(name).or_insert(index);
        index
    }

    pub fn add_edge(&mut self, src: NodeIndex, tgt: NodeIndex, edge: EdgeType) {
        let index = self.graph.add_edge(src, tgt, edge);
        self.edge_memo.entry((src, tgt)).or_insert(index);
    }

    pub fn remove_node(&mut self, index: NodeIndex) {
        if let Some(node) = self.graph.remove_node(index) {
            self.node_memo.remove(node.name());
        }
    }

    pub fn remove_edge(&mut self, index: EdgeIndex) {
        if let Some(edge) = self.graph.remove_edge(index) {
            self.edge_memo
                .remove(&(*edge.src_index(), *edge.tgt_index()));
        }
    }

    pub fn remove_node_by_name(&mut self, name: String) {
        if let Some(index) = self.get_node_index(name) {
            self.remove_node(*index);
        }
    }

    pub fn remove_edge_by_pair(&mut self, src: NodeIndex, tgt: NodeIndex) {
        if let Some(index) = self.get_edge_index(src, tgt) {
            self.remove_edge(*index);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let graph = HeteroGraph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_add_node() {
        let mut graph = HeteroGraph::new();

        let name = "node".to_owned();
        let value = 0.0;
        let node = TestNode::new(name.clone(), value, 1);

        graph.add_node(node.into());
        assert_eq!(graph.node_count(), 1);

        let node = graph.get_node_by_name(name.clone());
        assert!(node
            .map(|n| n.name())
            .map(|name_| *name_ == name)
            .unwrap_or(false));
    }

    #[test]
    fn test_remove_node() {
        let mut graph = HeteroGraph::new();

        let name = "node".to_owned();
        let value = 0.0;
        let node = TestNode::new(name.clone(), value, 1);

        graph.add_node(node.into());
        assert_eq!(graph.node_count(), 1);

        let node = graph.get_node_by_name(name.clone());
        assert!(node.is_some());

        graph.remove_node_by_name(name);
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_add_edge() {
        let mut graph = HeteroGraph::new();
        let src_name = "src_node".to_owned();
        let tgt_name = "tgt_node".to_owned();
        let src_value = 1.0;
        let tgt_value = 2.0;
        let src_node = TestNode::new(src_name.clone(), src_value, 1);
        let tgt_node = TestNode::new(tgt_name.clone(), tgt_value, 1);

        let src_index = graph.add_node(src_node.into());
        let tgt_index = graph.add_node(tgt_node.into());
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0);

        let (source, target) = (
            graph.get_node(src_index.clone()).unwrap(),
            graph.get_node(tgt_index.clone()).unwrap(),
        );

        match (source, target) {
            (NodeType::TestNode(source), NodeType::TestNode(target)) => {
                let edge = TestEdge::try_new(src_index, tgt_index, source, target);
                assert!(edge.is_some());
                graph.add_edge(src_index, tgt_index, edge.unwrap().into());
                assert_eq!(graph.edge_count(), 1);
            }
            _ => panic!("Failed to get nodes"),
        }

        let edge_index = graph.get_edge_index(src_index, tgt_index);
        assert!(edge_index.is_some());
    }

    #[test]
    fn test_remove_edge_by_node() {
        let mut graph = HeteroGraph::new();

        let src_name = "src_node".to_owned();
        let tgt_name = "tgt_node".to_owned();
        let src_value = 1.0;
        let tgt_value = 2.0;
        let src_node = TestNode::new(src_name.clone(), src_value, 1);
        let tgt_node = TestNode::new(tgt_name.clone(), tgt_value, 1);

        let src_index = graph.add_node(src_node.into());
        let tgt_index = graph.add_node(tgt_node.into());
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0);

        let (source, target) = (
            graph.get_node(src_index.clone()).unwrap(),
            graph.get_node(tgt_index.clone()).unwrap(),
        );

        match (source, target) {
            (NodeType::TestNode(source), NodeType::TestNode(target)) => {
                let edge = TestEdge::try_new(src_index, tgt_index, source, target);
                assert!(edge.is_some());
                graph.add_edge(src_index, tgt_index, edge.unwrap().into());
                assert_eq!(graph.edge_count(), 1);
            }
            _ => panic!("Failed to get nodes"),
        }

        graph.remove_node_by_name(src_name.clone());
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_remove_edge_by_edge() {
        let mut graph = HeteroGraph::new();

        let src_name = "src_node".to_owned();
        let tgt_name = "tgt_node".to_owned();
        let src_value = 1.0;
        let tgt_value = 2.0;
        let src_node = TestNode::new(src_name.clone(), src_value, 1);
        let tgt_node = TestNode::new(tgt_name.clone(), tgt_value, 1);

        let src_index = graph.add_node(src_node.into());
        let tgt_index = graph.add_node(tgt_node.into());
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0);

        let (source, target) = (
            graph.get_node(src_index.clone()).unwrap(),
            graph.get_node(tgt_index.clone()).unwrap(),
        );

        match (source, target) {
            (NodeType::TestNode(source), NodeType::TestNode(target)) => {
                let edge = TestEdge::try_new(src_index, tgt_index, source, target);
                assert!(edge.is_some());
                graph.add_edge(src_index, tgt_index, edge.unwrap().into());
                assert_eq!(graph.edge_count(), 1);
            }
            _ => panic!("Failed to get nodes"),
        }

        graph.remove_edge_by_pair(src_index, tgt_index);
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0);
    }
}

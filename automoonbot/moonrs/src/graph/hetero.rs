use crate::graph::*;

#[derive(Default)]
#[cfg_attr(feature = "python", pyclass(subclass))]
pub struct HeteroGraph {
    pub(super) graph: StableDiGraph<Box<dyn StaticNode>, Box<dyn StaticEdge>>,
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

    pub fn get_node(&self, index: NodeIndex) -> Option<&dyn StaticNode> {
        self.graph.node_weight(index).map(|boxed| &**boxed)
    }

    pub fn get_edge(&self, index: EdgeIndex) -> Option<&dyn StaticEdge> {
        self.graph.edge_weight(index).map(|boxed| &**boxed)
    }

    pub fn get_node_index(&self, name: String) -> Option<&NodeIndex> {
        self.node_memo.get(&name)
    }

    pub fn get_edge_index(&self, src: NodeIndex, tgt: NodeIndex) -> Option<&EdgeIndex> {
        self.edge_memo.get(&(src, tgt))
    }

    pub fn get_node_by_name(&self, name: String) -> Option<&dyn StaticNode> {
        let index = self.node_memo.get(&name)?;
        self.get_node(*index)
    }

    pub fn get_edge_by_pair(&self, src: NodeIndex, tgt: NodeIndex) -> Option<&dyn StaticEdge> {
        let index = self.edge_memo.get(&(src, tgt))?;
        self.get_edge(*index)
    }

    pub fn add_node(&mut self, node: Box<dyn StaticNode>) -> NodeIndex {
        let name = node.name().to_string();
        let index = self.graph.add_node(node);
        self.node_memo.entry(name).or_insert(index);
        index
    }

    pub fn add_edge(&mut self, src: NodeIndex, tgt: NodeIndex, edge: Box<dyn StaticEdge>) {
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
        let node = TestNode::new(name.clone(), value);

        graph.add_node(Box::new(node));
        assert_eq!(graph.node_count(), 1);

        let node = graph.get_node_by_name(name.clone());
        assert!(node.is_some());

        let node = node.unwrap();
        assert_eq!(node.value(), value);
    }

    #[test]
    fn test_remove_node() {
        let mut graph = HeteroGraph::new();

        let name = "node".to_owned();
        let value = 0.0;
        let node = TestNode::new(name.clone(), value);

        graph.add_node(Box::new(node));
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
        let src_node = TestNode::new(src_name.clone(), src_value);
        let tgt_node = TestNode::new(tgt_name.clone(), tgt_value);

        graph.add_node(Box::new(src_node));
        assert_eq!(graph.node_count(), 1);

        graph.add_node(Box::new(tgt_node));
        assert_eq!(graph.node_count(), 2);

        let src_index = graph.get_node_index(src_name);
        let tgt_index = graph.get_node_index(tgt_name);
        assert!(src_index.is_some());
        assert!(tgt_index.is_some());
        let src_index = src_index.unwrap().to_owned();
        let tgt_index = tgt_index.unwrap().to_owned();

        let src_node = graph.get_node(src_index);
        let tgt_node = graph.get_node(tgt_index);
        assert!(src_node.is_some());
        assert!(tgt_node.is_some());
        let src_node = src_node.unwrap();
        let tgt_node = tgt_node.unwrap();

        assert_eq!(graph.edge_count(), 0);
        let edge = TestEdge::new(src_index, tgt_index, src_node, tgt_node);
        let edge_value = edge.value().clone();
        graph.add_edge(src_index, tgt_index, Box::new(edge));
        assert_eq!(graph.edge_count(), 1);

        let edge_index = graph.get_edge_index(src_index, tgt_index);
        assert!(edge_index.is_some());

        let edge = graph.get_edge_by_pair(src_index, tgt_index);
        assert!(edge.is_some());

        let edge = edge.unwrap();
        assert_eq!(edge.value(), edge_value);
    }

    #[test]
    fn test_remove_edge_by_node() {
        let mut graph = HeteroGraph::new();

        let src_name = "src_node".to_owned();
        let tgt_name = "tgt_node".to_owned();
        let src_value = 1.0;
        let tgt_value = 2.0;
        let src_node = TestNode::new(src_name.clone(), src_value);
        let tgt_node = TestNode::new(tgt_name.clone(), tgt_value);

        let src_index = graph.add_node(Box::new(src_node));
        let tgt_index = graph.add_node(Box::new(tgt_node));
        let src_node = graph.get_node(src_index).unwrap();
        let tgt_node = graph.get_node(tgt_index).unwrap();

        let edge = TestEdge::new(src_index, tgt_index, src_node, tgt_node);
        graph.add_edge(src_index, tgt_index, Box::new(edge));
        assert_eq!(graph.edge_count(), 1);

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
        let src_node = TestNode::new(src_name.clone(), src_value);
        let tgt_node = TestNode::new(tgt_name.clone(), tgt_value);

        let src_index = graph.add_node(Box::new(src_node));
        let tgt_index = graph.add_node(Box::new(tgt_node));
        let src_node = graph.get_node(src_index).unwrap();
        let tgt_node = graph.get_node(tgt_index).unwrap();

        let edge = TestEdge::new(src_index, tgt_index, src_node, tgt_node);
        graph.add_edge(src_index, tgt_index, Box::new(edge));
        assert_eq!(graph.edge_count(), 1);

        graph.remove_edge_by_pair(src_index, tgt_index);
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0);
    }
}

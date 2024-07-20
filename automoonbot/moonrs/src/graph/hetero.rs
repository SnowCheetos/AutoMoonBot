use crate::{
    edges::{EdgeType, StaticEdge},
    nodes::{NodeType, StaticNode},
    *,
};

#[derive(Default)]
pub struct HeteroGraph {
    graph: StableDiGraph<NodeType, EdgeType>,
    node_memo: HashMap<&'static str, HashSet<NodeIndex>>,
    edge_memo: HashMap<&'static str, HashSet<EdgeIndex>>,
}

impl HeteroGraph {
    pub fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            node_memo: HashMap::new(),
            edge_memo: HashMap::new(),
        }
    }

    pub fn get_node(&self, index: NodeIndex) -> Option<&NodeType> {
        self.graph.node_weight(index)
    }

    pub fn get_edge(&self, index: EdgeIndex) -> Option<&EdgeType> {
        self.graph.edge_weight(index)
    }

    pub fn add_node(&mut self, node: NodeType) {
        let cls = node.cls();
        let index = self.graph.add_node(node);
        self.node_memo.entry(cls).or_default().insert(index);
    }

    pub fn add_edge(&mut self, src: NodeIndex, tgt: NodeIndex, edge: EdgeType) {
        let cls = edge.cls();
        let index = self.graph.add_edge(src, tgt, edge);
        self.edge_memo.entry(cls).or_default().insert(index);
    }

    pub fn rem_node(&mut self, index: NodeIndex) {
        if let Some(node) = self.graph.remove_node(index) {
            if let Some(indices) = self.node_memo.get_mut(&node.cls()) {
                indices.remove(&index);
                if indices.is_empty() {
                    self.node_memo.remove(node.cls());
                }
            }
        }
    }

    pub fn rem_edge(&mut self, index: EdgeIndex) {
        if let Some(edge) = self.graph.remove_edge(index) {
            if let Some(indices) = self.edge_memo.get_mut(&edge.cls()) {
                indices.remove(&index);
                if indices.is_empty() {
                    self.edge_memo.remove(&edge.cls());
                }
            }
        }
    }
}

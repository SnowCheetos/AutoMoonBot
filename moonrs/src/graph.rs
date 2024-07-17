use petgraph::stable_graph::{EdgeIndex, NodeIndex, StableDiGraph};
use std::collections::{HashMap, HashSet};

use crate::{edges::statics::StaticEdge, nodes::statics::StaticNode};

#[derive(Default)]
pub struct HeteroGraph<N, E> {
    graph: StableDiGraph<N, E>,
    node_memo: HashMap<&'static str, HashSet<NodeIndex>>,
    edge_memo: HashMap<&'static str, HashSet<EdgeIndex>>,
}

impl<N, E> HeteroGraph<N, E>
where
    N: StaticNode,
    E: StaticEdge,
{
    pub fn new() -> Self {
        Self {
            graph: StableDiGraph::new(),
            node_memo: HashMap::new(),
            edge_memo: HashMap::new(),
        }
    }

    pub fn get_node(&self, index: NodeIndex) -> Option<&N> {
        self.graph.node_weight(index)
    }

    pub fn get_edge(&self, index: EdgeIndex) -> Option<&E> {
        self.graph.edge_weight(index)
    }

    pub fn add_node(&mut self, node: N) {
        let name = node.id();
        let index = self.graph.add_node(node);
        self.node_memo.entry(name).or_default().insert(index);
    }

    pub fn add_edge(&mut self, src: NodeIndex, tgt: NodeIndex, edge: E) {
        let name = edge.name();
        let index = self.graph.add_edge(src, tgt, edge);
        self.edge_memo.entry(name).or_default().insert(index);
    }

    pub fn rem_node(&mut self, index: NodeIndex) {
        if let Some(node) = self.graph.remove_node(index) {
            if let Some(indices) = self.node_memo.get_mut(&node.id()) {
                indices.remove(&index);
                if indices.is_empty() {
                    self.node_memo.remove(&node.id());
                }
            }
        }
    }

    pub fn rem_edge(&mut self, index: EdgeIndex) {
        if let Some(edge) = self.graph.remove_edge(index) {
            if let Some(indices) = self.edge_memo.get_mut(&edge.name()) {
                indices.remove(&index);
                if indices.is_empty() {
                    self.edge_memo.remove(&edge.name());
                }
            }
        }
    }
}

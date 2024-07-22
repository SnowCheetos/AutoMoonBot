use crate::graph::*;

impl HeteroGraph {
    fn compute_all_edges(&mut self, src: NodeIndex) {
        let indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        for tgt in indices.into_iter() {
            self.try_add_edge(src, tgt);
            self.try_add_edge(tgt, src);
        }
    }

    fn try_add_edge(&mut self, src: NodeIndex, tgt: NodeIndex) {
        if let Some(edge) = self.compute_dir_edge(src, tgt) {
            self.add_edge(src, tgt, edge);
        }
    }

    fn compute_dir_edge(&self, src: NodeIndex, tgt: NodeIndex) -> Option<Box<dyn StaticEdge>> {
        if let (Some(source), Some(target)) = (self.get_node(src), self.get_node(tgt)) {
            return match (source.cls(), target.cls()) {
                ("TestNode", "TestNode") => TestEdge::try_new(src, tgt, source, target)
                    .map(|edge| Box::new(edge) as Box<dyn StaticEdge>),
                _ => None,
            };
        }
        None
    }

    pub fn add_test_node(&mut self, name: String, value: f64, capacity: usize) {
        let node = TestNode::new(name, value, capacity);
        let index = self.add_node(Box::new(node));
        self.compute_all_edges(index);
    }
}

#[cfg(feature = "python")]
#[pymethods]
impl HeteroGraph {
    #[new]
    pub fn init() -> Self {
        Self::new()
    }

    #[staticmethod]
    pub fn hello_python() -> &'static str {
        "Hello From HeteroGraph"
    }

    #[pyo3(name = "node_count")]
    pub fn node_count_py(&self) -> usize {
        self.node_count()
    }

    #[pyo3(name = "edge_count")]
    pub fn edge_count_py(&self) -> usize {
        self.edge_count()
    }

    #[pyo3(name = "remove_node")]
    pub fn remove_node_py(&mut self, name: String) {
        self.remove_node_by_name(name);
    }

    #[pyo3(name = "add_test_node")]
    pub fn add_test_node_py(&mut self, name: String, value: f64, capacity: usize) {
        self.add_test_node(name, value, capacity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_edges() {
        let mut graph = HeteroGraph::new();
        let name_1 = "node_1".to_owned();
        let name_2 = "node_2".to_owned();
        let name_3 = "node_3".to_owned();
        let value_1 = 1.0;
        let value_2 = 2.0;
        let value_3 = 2.0;

        graph.add_test_node(name_1.clone(), value_1, 1);
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.edge_count(), 0);

        graph.add_test_node(name_2.clone(), value_2, 1);
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 2);

        graph.add_test_node(name_3.clone(), value_3, 1);
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 4);

        graph.remove_node_by_name(name_1.clone());
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0);
    }
}

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

    fn compute_dir_edge(&self, src: NodeIndex, tgt: NodeIndex) -> Option<EdgeType> {
        if let (Some(source), Some(target)) = (self.get_node(src), self.get_node(tgt)) {
            return match (source, target) {
                (NodeType::TestNode(source), NodeType::TestNode(target)) => {
                    TestEdge::try_new(src, tgt, source, target).map(|edge| edge.into())
                }
                (NodeType::Publisher(source), NodeType::Article(target)) => {
                    Published::try_new(src, tgt, source, target).map(|edge| edge.into())
                }
                (NodeType::Article(source), NodeType::Company(target)) => {
                    Mentioned::try_new(src, tgt, source, target).map(|edge| edge.into())
                }
                (NodeType::Article(source), NodeType::Equity(target)) => {
                    Referenced::try_new(src, tgt, source, target).map(|edge| edge.into())
                }
                (NodeType::Company(source), NodeType::Equity(target)) => {
                    Issues::try_new(src, tgt, source, target).map(|edge| edge.into())
                }
                (NodeType::ETFs(source), NodeType::Indices(target)) => {
                    Mirrors::try_new(src, tgt, source, target).map(|edge| edge.into())
                }
                (NodeType::Equity(source), NodeType::Equity(target)) => {
                    Influences::try_new(src, tgt, source, target).map(|edge| edge.into())
                }
                (NodeType::Equity(source), NodeType::Options(target)) => {
                    Derives::try_new(src, tgt, source, target).map(|edge| edge.into())
                }
                _ => None,
            };
        }
        None
    }

    pub fn add_test_node(&mut self, name: String, value: f64, capacity: usize) {
        let node = TestNode::new(name, value, capacity);
        let index = self.add_node(node.into());
        self.compute_all_edges(index);
    }

    pub fn add_article(
        &mut self,
        title: String,
        summary: String,
        sentiment: f64,
        publisher: String,
        tickers: Option<HashMap<String, f64>>,
    ) {
        let node = Article::new(title, summary, sentiment, publisher.clone(), tickers);
        let index = self.add_node(node.into());
        self.compute_all_edges(index);
        if self.get_node_index(publisher.clone()).is_none() {
            self.add_publisher(publisher, 100);
        }
    }

    pub fn add_publisher(&mut self, name: String, capacity: usize) {
        let node = Publisher::new(name, capacity);
        let index = self.add_node(node.into());
        self.compute_all_edges(index);
    }

    pub fn add_company(&mut self, name: String, symbols: HashSet<String>, capacity: usize) {
        let node = Company::new(name, symbols, capacity);
        let index = self.add_node(node.into());
        self.compute_all_edges(index);
    }

    pub fn add_equity(&mut self, symbol: String, capacity: usize) {
        let node = Equity::new(symbol, capacity);
        let index = self.add_node(node.into());
        self.compute_all_edges(index);
    }

    pub fn add_currency(&mut self, symbol: String, capacity: usize) {
        let node = Currency::new(symbol, capacity);
        let index = self.add_node(node.into());
        self.compute_all_edges(index);
    }

    pub fn add_etf(&mut self, symbol: String, indice: String, capacity: usize) {
        let node = ETFs::new(symbol, indice.clone(), capacity);
        let index = self.add_node(node.into());
        self.compute_all_edges(index);
        if self.get_node_index(indice.clone()).is_none() {
            self.add_indice(indice, capacity);
        }
    }

    pub fn add_indice(&mut self, symbol: String, capacity: usize) {
        let node = Indices::new(symbol, capacity);
        let index = self.add_node(node.into());
        self.compute_all_edges(index);
    }

    pub fn add_bond(
        &mut self,
        symbol: String,
        interest_rate: f64,
        maturity: Instant,
        capacity: usize,
    ) {
        let node = Bonds::new(symbol, interest_rate, maturity, capacity);
        let index = self.add_node(node.into());
        self.compute_all_edges(index);
    }

    pub fn add_option(
        &mut self,
        contract_id: String,
        direction: String,
        underlying: String,
        strike: f64,
        expiration: Instant,
        capacity: usize,
    ) {
        let node = Options::new(
            contract_id,
            direction,
            underlying,
            strike,
            expiration,
            capacity,
        );
        let index = self.add_node(node.into());
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

    pub fn clear(&mut self) {
        self.graph.clear();
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

    #[pyo3(name = "add_article")]
    pub fn add_article_py(
        &mut self,
        title: String,
        summary: String,
        sentiment: f64,
        publisher: String,
        tickers: Option<HashMap<String, f64>>,
    ) {
        self.add_article(title, summary, sentiment, publisher, tickers);
    }

    #[pyo3(name = "add_publisher")]
    pub fn add_publisher_py(&mut self, name: String, capacity: usize) {
        self.add_publisher(name, capacity);
    }

    #[pyo3(name = "add_company")]
    pub fn add_company_py(&mut self, name: String, symbols: HashSet<String>, capacity: usize) {
        self.add_company(name, symbols, capacity);
    }

    #[pyo3(name = "add_equity")]
    pub fn add_equity_py(&mut self, symbol: String, capacity: usize) {
        self.add_equity(symbol, capacity);
    }

    #[pyo3(name = "add_currency")]
    pub fn add_currency_py(&mut self, symbol: String, capacity: usize) {
        self.add_currency(symbol, capacity);
    }

    #[pyo3(name = "add_etf")]
    pub fn add_etf_py(&mut self, symbol: String, indice: String, capacity: usize) {
        self.add_etf(symbol, indice, capacity);
    }

    #[pyo3(name = "add_indice")]
    pub fn add_indice_py(&mut self, symbol: String, capacity: usize) {
        self.add_indice(symbol, capacity);
    }

    #[pyo3(name = "add_bond")]
    pub fn add_bond_py(
        &mut self,
        symbol: String,
        interest_rate: f64,
        maturity: Instant,
        capacity: usize,
    ) {
        self.add_bond(symbol, interest_rate, maturity, capacity);
    }

    #[pyo3(name = "add_option")]
    pub fn add_option_py(
        &mut self,
        contract_id: String,
        direction: String,
        underlying: String,
        strike: f64,
        expiration: Instant,
        capacity: usize,
    ) {
        self.add_option(
            contract_id,
            direction,
            underlying,
            strike,
            expiration,
            capacity,
        );
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

    #[test]
    fn test_combination_1() {
        let mut graph = HeteroGraph::new();
        graph.add_article(
            "test_article".to_owned(),
            "test_summary".to_owned(),
            0.5,
            "test_publisher".to_owned(),
            None,
        );

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);

        let article = graph.get_node_by_name("test_article".to_owned());
        assert!(article.is_some_and(|a| a.name() == "test_article"));
        let article_index = graph.get_node_index("test_article".to_owned()).unwrap();

        let publisher = graph.get_node_by_name("test_publisher".to_owned());
        assert!(publisher.is_some_and(|p| p.name() == "test_publisher"));
        let publisher_index = graph.get_node_index("test_publisher".to_owned()).unwrap();

        let edge = graph.get_edge_by_names("test_publisher".to_owned(), "test_article".to_owned());
        assert!(edge.is_some_and(|e| e.cls() == "Published"
            && e.src_index() == publisher_index
            && e.tgt_index() == article_index));
    }
}

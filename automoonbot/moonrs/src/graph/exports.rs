use petgraph::graph::edge_index;

use crate::graph::*;

impl HeteroGraph {
    fn to_pyg(
        &self,
    ) -> (
        HashMap<String, na::DMatrix<f64>>,
        HashMap<String, na::DMatrix<i64>>,
        HashMap<String, na::DMatrix<f64>>,
    ) {
        let mut x: HashMap<String, na::DMatrix<f64>> = HashMap::new();
        let mut edge_index: HashMap<String, na::DMatrix<i64>> = HashMap::new();
        let mut edge_attr: HashMap<String, na::DMatrix<f64>> = HashMap::new();
        let mut temp: HashMap<NodeIndex, (String, usize)> = HashMap::new();
        for (cls, indices) in self.node_cls_memo() {
            if indices.is_empty() {
                continue;
            }

            if let Some(&sample_index) = indices.iter().next() {
                if let Some(sample_node) = self.get_node(sample_index) {
                    if let Some(feature) = sample_node.feature() {
                        x.insert(
                            cls.clone(),
                            na::DMatrix::zeros(indices.len(), feature.len()),
                        );
                        if let Some(matrix) = x.get_mut(cls) {
                            for (i, &index) in indices.iter().enumerate() {
                                if let Some(_) = self.get_node(index) {
                                    matrix.row_mut(i).copy_from(&feature);
                                    temp.insert(index, (cls.clone(), i));
                                }
                            }
                        }
                    }
                }
            }
        }
        for (cls, edges) in self.edge_cls_memo() {
            if edges.is_empty() {
                continue;
            }

            let edge_index_matrix = edge_index.get_mut(cls).unwrap();
            let edge_attr_matrix = edge_attr.get_mut(cls).unwrap();

            for (i, edge_index) in edges.iter().enumerate() {
                let edge = self.get_edge(*edge_index).unwrap();
                let src = edge.src_index();
                let tgt = edge.tgt_index();

                if let (Some(&(_, src_index)), Some(&(_, tgt_index))) =
                    (temp.get(&src), temp.get(&tgt))
                {
                    edge_index_matrix[(0, i)] = src_index as i64;
                    edge_index_matrix[(1, i)] = tgt_index as i64;
                    edge_attr_matrix.row_mut(i).copy_from(&edge.feature());
                }
            }
        }
        (x, edge_index, edge_attr)
    }

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
        capacity: usize,
        tickers: Option<HashMap<String, f64>>,
    ) {
        let node = Article::new(title, summary, sentiment, publisher.clone(), tickers);
        let index = self.add_node(node.into());
        self.compute_all_edges(index);
        if self.get_node_index(publisher.clone()).is_none() {
            self.add_publisher(publisher, capacity);
        }
    }

    pub fn add_publisher(&mut self, name: String, capacity: usize) {
        let node = Publisher::new(name, capacity);
        let index = self.add_node(node.into());
        self.compute_all_edges(index);
    }

    pub fn add_company(&mut self, name: String, symbols: Vec<String>, capacity: usize) {
        let node = Company::new(name, symbols, capacity);
        let index = self.add_node(node.into());
        self.compute_all_edges(index);
    }

    pub fn add_equity(&mut self, symbol: String, company: Option<String>, capacity: usize) {
        let node = Equity::new(symbol, company, capacity);
        let company = node.company();
        let index = self.add_node(node.into());
        self.compute_all_edges(index);
        if let Some(company) = company {
            if self.get_node_index(company.clone()).is_none() {
                self.add_company(company.to_owned(), Vec::new(), capacity);
            }
        }
    }

    pub fn add_currency(&mut self, symbol: String, capacity: usize) {
        let node = Currency::new(symbol, capacity);
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
        capacity: usize,
        tickers: HashMap<String, f64>,
    ) {
        if tickers.is_empty() {
            self.add_article(title, summary, sentiment, publisher, capacity, None);
        } else {
            self.add_article(
                title,
                summary,
                sentiment,
                publisher,
                capacity,
                Some(tickers),
            );
        }
    }

    #[pyo3(name = "add_publisher")]
    pub fn add_publisher_py(&mut self, name: String, capacity: usize) {
        self.add_publisher(name, capacity);
    }

    #[pyo3(name = "add_company")]
    pub fn add_company_py(&mut self, name: String, capacity: usize) {
        self.add_company(name, capacity);
    }

    #[pyo3(name = "add_equity")]
    pub fn add_equity_py(&mut self, symbol: String, capacity: usize) {
        self.add_equity(symbol, capacity);
    }

    #[pyo3(name = "add_currency")]
    pub fn add_currency_py(&mut self, symbol: String, capacity: usize) {
        self.add_currency(symbol, capacity);
    }

    #[pyo3(name = "add_bond")]
    pub fn add_bond_py(
        &mut self,
        symbol: String,
        interest_rate: f64,
        maturity: f64,
        capacity: usize,
    ) {
        let maturity_instant = UNIX_EPOCH + Duration::from_secs_f64(maturity);
        let maturity =
            Instant::now() + (maturity_instant.duration_since(SystemTime::now()).unwrap());
        self.add_bond(symbol, interest_rate, maturity, capacity);
    }

    #[pyo3(name = "add_option")]
    pub fn add_option_py(
        &mut self,
        contract_id: String,
        direction: String,
        underlying: String,
        strike: f64,
        expiration: f64,
        capacity: usize,
    ) {
        let expiration_instant = UNIX_EPOCH + Duration::from_secs_f64(expiration);
        let expiration = Instant::now()
            + (expiration_instant
                .duration_since(SystemTime::now())
                .unwrap());
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
    fn test_to_pyg() {
        // Initialize the graph
        let mut graph = HeteroGraph::new();

        // Add nodes and edges
        graph.add_test_node("TestNode1".to_string(), 1.0, 3);
        graph.add_test_node("TestNode2".to_string(), 2.0, 3);
        graph.add_article(
            "Article1".to_string(),
            "Summary1".to_string(),
            0.5,
            "Publisher1".to_string(),
            3,
            None,
        );
        graph.add_company("Company1".to_string(), vec!["Equity1".to_string()], 3);
        graph.add_equity("Equity1".to_string(), Some("Company1".to_string()), 3);

        // Convert graph to pyg format
        let (x_py, edge_index_py, edge_attr_py) = graph.to_pyg();

        // Define expected results
        let mut expected_x: HashMap<String, na::DMatrix<f64>> = HashMap::new();
        expected_x.insert(
            "TestNode".to_string(),
            na::DMatrix::from_row_slice(2, 3, &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]),
        );

        let mut expected_edge_index: HashMap<String, na::DMatrix<i64>> = HashMap::new();
        // Assume edges between nodes are added in some order
        expected_edge_index.insert(
            "TestEdge".to_string(),
            na::DMatrix::from_row_slice(2, 2, &[0, 1, 1, 0]),
        );

        let mut expected_edge_attr: HashMap<String, na::DMatrix<f64>> = HashMap::new();
        expected_edge_attr.insert(
            "TestEdge".to_string(),
            na::DMatrix::from_row_slice(2, 3, &[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        );

        // Check if the result matches the expected values
        assert_eq!(x_py, expected_x);
        assert_eq!(edge_index_py, expected_edge_index);
        assert_eq!(edge_attr_py, expected_edge_attr);
    }

    #[test]
    fn test_combination_1() {
        let mut graph = HeteroGraph::new();
        graph.add_article(
            "test_article".to_owned(),
            "test_summary".to_owned(),
            0.5,
            "test_publisher".to_owned(),
            10,
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

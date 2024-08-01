use crate::nodes::*;

pub trait StaticNode: Send + Sync {
    fn cls(&self) -> &'static str;
    fn name(&self) -> &String;
    fn value(&self) -> Option<f64>;
    fn dim(&self) -> usize;
    fn feature(&self) -> Option<na::RowDVector<f64>>;
}

impl StaticNode for NodeType {
    fn cls(&self) -> &'static str {
        match self {
            NodeType::Article(article) => article.cls(),
            NodeType::Publisher(publisher) => publisher.cls(),
            NodeType::Company(company) => company.cls(),
            NodeType::Currency(currency) => currency.cls(),
            NodeType::Equity(equity) => equity.cls(),
            NodeType::Bonds(bonds) => bonds.cls(),
            NodeType::Options(options) => options.cls(),
            NodeType::TestNode(test_node) => test_node.cls(),
        }
    }

    fn name(&self) -> &String {
        match self {
            NodeType::Article(article) => &article.title,
            NodeType::Publisher(publisher) => &publisher.name,
            NodeType::Company(company) => &company.name,
            NodeType::Currency(currency) => &currency.symbol,
            NodeType::Equity(equity) => &equity.symbol,
            NodeType::Bonds(bonds) => &bonds.symbol,
            NodeType::Options(options) => &options.contract_id,
            NodeType::TestNode(test_node) => &test_node.name,
        }
    }

    fn value(&self) -> Option<f64> {
        None
    }

    fn dim(&self) -> usize {
        match self {
            NodeType::Article(node) => node.dim(),
            NodeType::Publisher(node) => node.dim(),
            NodeType::Company(node) => node.dim(),
            NodeType::Currency(node) => node.dim(),
            NodeType::Equity(node) => node.dim(),
            NodeType::Bonds(node) => node.dim(),
            NodeType::Options(node) => node.dim(),
            NodeType::TestNode(node) => node.dim(),
        }
    }

    fn feature(&self) -> Option<na::RowDVector<f64>> {
        match self {
            NodeType::Article(node) => {
                node.feature()
            },
            NodeType::Publisher(node) => {
                node.feature()
            },
            NodeType::Company(node) => {
                node.feature()
            },
            NodeType::Currency(node) => {
                node.feature()
            },
            NodeType::Equity(node) => {
                node.feature()
            },
            NodeType::Bonds(node) => {
                node.feature()
            },
            NodeType::Options(node) => {
                node.feature()
            },
            NodeType::TestNode(node) => {
                node.feature()
            },
        }
    }
}

impl StaticNode for Article {
    fn cls(&self) -> &'static str {
        "Article"
    }

    fn name(&self) -> &String {
        &self.title
    }

    fn value(&self) -> Option<f64> {
        None
    }

    fn feature(&self) -> Option<na::RowDVector<f64>> {
        Some(na::RowDVector::from_vec(vec![self.sentiment()]))
    }

    fn dim(&self) -> usize {
        1
    }
}

impl StaticNode for Publisher {
    fn cls(&self) -> &'static str {
        "Publisher"
    }

    fn name(&self) -> &String {
        &self.name
    }

    fn value(&self) -> Option<f64> {
        None
    }

    fn feature(&self) -> Option<na::RowDVector<f64>> {
        if self.sentiments().empty() {
            return None;
        }
        Some(na::RowDVector::from_vec(vec![*self.sentiments().last().unwrap()]))
    }

    fn dim(&self) -> usize {
        1
    }
}

impl StaticNode for Company {
    fn cls(&self) -> &'static str {
        "Company"
    }

    fn name(&self) -> &String {
        &self.name
    }

    fn value(&self) -> Option<f64> {
        None
    }

    fn feature(&self) -> Option<na::RowDVector<f64>> {
        Some(self.mat()?.row(0).into_owned())
    }

    fn dim(&self) -> usize {
        self.income_statement.cols()
            + self.balance_sheet.cols()
            + self.cash_flow.cols()
            + self.earnings.cols()
    }
}

impl StaticNode for Currency {
    fn cls(&self) -> &'static str {
        "Currency"
    }

    fn name(&self) -> &String {
        &self.symbol
    }

    fn value(&self) -> Option<f64> {
        None
    }

    fn feature(&self) -> Option<na::RowDVector<f64>> {
        Some(self.mat()?.row(0).into_owned())
    }

    fn dim(&self) -> usize {
        self.history.cols()
    }
}

impl StaticNode for Equity {
    fn cls(&self) -> &'static str {
        "Equity"
    }

    fn name(&self) -> &String {
        &self.symbol
    }

    fn value(&self) -> Option<f64> {
        None
    }

    fn feature(&self) -> Option<na::RowDVector<f64>> {
        Some(self.mat()?.row(0).into_owned())
    }

    fn dim(&self) -> usize {
        self.history.cols()
    }
}

impl StaticNode for Bonds {
    fn cls(&self) -> &'static str {
        "Bonds"
    }

    fn name(&self) -> &String {
        &self.symbol
    }

    fn value(&self) -> Option<f64> {
        None
    }

    fn feature(&self) -> Option<na::RowDVector<f64>> {
        Some(self.mat()?.row(0).into_owned())
    }

    fn dim(&self) -> usize {
        self.history.cols()
    }
}

impl StaticNode for Options {
    fn cls(&self) -> &'static str {
        "Options"
    }

    fn name(&self) -> &String {
        &self.contract_id
    }

    fn value(&self) -> Option<f64> {
        None
    }

    fn feature(&self) -> Option<na::RowDVector<f64>> {
        Some(self.mat()?.row(0).into_owned())
    }

    fn dim(&self) -> usize {
        self.history.cols()
    }
}

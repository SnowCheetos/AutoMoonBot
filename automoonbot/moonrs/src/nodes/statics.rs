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
            NodeType::Article(_) => "Article",
            NodeType::Publisher(_) => "Publisher",
            NodeType::Company(_) => "Company",
            NodeType::Currency(_) => "Currency",
            NodeType::Equity(_) => "Equity",
            NodeType::Bonds(_) => "Bonds",
            NodeType::Options(_) => "Options",
            NodeType::TestNode(_) => "TestNode",
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
                Some(na::RowDVector::from_vec(vec![node.sentiment()]))
            },
            NodeType::Publisher(node) => {
                Some(na::RowDVector::from_vec(vec![*node.sentiments().last().unwrap()]))
            },
            NodeType::Company(node) => {
                Some(node.mat()?.row(0).into_owned())
            },
            NodeType::Currency(node) => {
                Some(node.mat()?.row(0).into_owned())
            },
            NodeType::Equity(node) => {
                Some(node.mat()?.row(0).into_owned())
            },
            NodeType::Bonds(node) => {
                Some(node.mat()?.row(0).into_owned())
            },
            NodeType::Options(node) => {
                Some(node.mat()?.row(0).into_owned())
            },
            NodeType::TestNode(_) => {
                Some(na::RowDVector::from_vec(vec![0.0]))
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
        todo!()
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
        todo!()
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
        todo!()
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
        todo!()
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
        todo!()
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
        todo!()
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
        todo!()
    }

    fn dim(&self) -> usize {
        self.history.cols()
    }
}

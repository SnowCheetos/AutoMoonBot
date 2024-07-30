use crate::nodes::*;

pub trait StaticNode: Send + Sync {
    fn cls(&self) -> &'static str;
    fn name(&self) -> &String;
    fn value(&self) -> Option<f64>;
    fn as_any(&self) -> &dyn Any;
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
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

    fn as_any(&self) -> &dyn Any {
        self
    }
}

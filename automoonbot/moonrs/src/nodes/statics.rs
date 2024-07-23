use crate::nodes::*;

pub trait StaticNode: Send + Sync {
    fn cls(&self) -> &'static str;
    fn name(&self) -> &String;
    fn value(&self) -> Option<f64>;
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
}

impl StaticNode for Indices {
    fn cls(&self) -> &'static str {
        "Indices"
    }

    fn name(&self) -> &String {
        &self.symbol
    }

    fn value(&self) -> Option<f64> {
        None
    }
}

impl StaticNode for ETFs {
    fn cls(&self) -> &'static str {
        "ETFs"
    }

    fn name(&self) -> &String {
        &self.symbol
    }

    fn value(&self) -> Option<f64> {
        None
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
}

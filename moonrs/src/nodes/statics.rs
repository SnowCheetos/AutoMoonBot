use super::*;

pub trait StaticNode: Eq + Hash + Clone + Send + Sync + Copy {
    fn params(&self);
}

impl StaticNode for Author {
    fn params(&self) {}
}

impl StaticNode for Publisher {
    fn params(&self) {}
}

impl StaticNode for Article {
    fn params(&self) {}
}

impl StaticNode for Company {
    fn params(&self) {}
}

impl StaticNode for Sector {
    fn params(&self) {}
}

impl StaticNode for Industry {
    fn params(&self) {}
}

impl StaticNode for Exchange {
    fn params(&self) {}
}

impl StaticNode for Currency {
    fn params(&self) {}
}

impl StaticNode for Bond {
    fn params(&self) {}
}

impl StaticNode for ETFund {
    fn params(&self) {}
}

impl StaticNode for Commodity {
    fn params(&self) {}
}

impl StaticNode for CryptoCurrency {
    fn params(&self) {}
}

impl StaticNode for Options {
    fn params(&self) {}
}

impl StaticNode for Futures {
    fn params(&self) {}
}

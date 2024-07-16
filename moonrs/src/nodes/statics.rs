use crate::nodes::*;

pub trait StaticNode: Clone + Send + Sync {
    fn name(&self) -> &'static str;
}

impl StaticNode for Author {
    fn name(&self) -> &'static str {
        self.name
    }
}

impl StaticNode for Publisher {
    fn name(&self) -> &'static str {
        self.name
    }
}

impl StaticNode for Article {
    fn name(&self) -> &'static str {
        self.title
    }
}

impl StaticNode for Company {
    fn name(&self) -> &'static str {
        self.name
    }
}

impl StaticNode for Sector {
    fn name(&self) -> &'static str {
        self.name
    }
}

impl StaticNode for Industry {
    fn name(&self) -> &'static str {
        self.name
    }
}

impl StaticNode for Exchange {
    fn name(&self) -> &'static str {
        self.name
    }
}

impl StaticNode for Currency {
    fn name(&self) -> &'static str {
        self.symbol
    }
}

impl StaticNode for Bond {
    fn name(&self) -> &'static str {
        self.symbol
    }
}

impl StaticNode for TreasuryBill {
    fn name(&self) -> &'static str {
        self.symbol
    }
}

impl StaticNode for Index {
    fn name(&self) -> &'static str {
        self.symbol
    }
}

impl StaticNode for MutualFund {
    fn name(&self) -> &'static str {
        self.symbol
    }
}

impl StaticNode for ETFund {
    fn name(&self) -> &'static str {
        self.symbol
    }
}

impl StaticNode for Equity {
    fn name(&self) -> &'static str {
        self.symbol
    }
}

impl StaticNode for Commodity {
    fn name(&self) -> &'static str {
        self.symbol
    }
}

impl StaticNode for CryptoCurrency {
    fn name(&self) -> &'static str {
        self.symbol
    }
}

impl StaticNode for Options {
    fn name(&self) -> &'static str {
        self.symbol
    }
}

impl StaticNode for Futures {
    fn name(&self) -> &'static str {
        self.symbol
    }
}

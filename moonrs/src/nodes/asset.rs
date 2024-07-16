use crate::nodes::{dynamic::DynamicNode, *};

pub trait Asset: DynamicNode {
    fn symbol(&self) -> &'static str;
    fn region(&self) -> &'static str;
    //fn latest(&self) -> Vec<f64>;
    //fn quote(&self, aggr: Option<String>) -> f64;
}

impl Asset for Currency {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
    fn region(&self) -> &'static str {
        self.region
    }
}

impl Asset for Bond {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
    fn region(&self) -> &'static str {
        self.region
    }
}

impl Asset for TreasuryBill {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
    fn region(&self) -> &'static str {
        self.region
    }
}

impl Asset for Index {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
    fn region(&self) -> &'static str {
        self.region
    }
}

impl Asset for MutualFund {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
    fn region(&self) -> &'static str {
        self.region
    }
}

impl Asset for ETFund {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
    fn region(&self) -> &'static str {
        self.region
    }
}

impl Asset for Equity {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
    fn region(&self) -> &'static str {
        self.region
    }
}

impl Asset for Commodity {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
    fn region(&self) -> &'static str {
        self.region
    }
}

impl Asset for CryptoCurrency {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
    fn region(&self) -> &'static str {
        self.region
    }
}

impl Asset for Options {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
    fn region(&self) -> &'static str {
        self.region
    }
}

impl Asset for Futures {
    fn symbol(&self) -> &'static str {
        self.symbol
    }
    fn region(&self) -> &'static str {
        self.region
    }
}

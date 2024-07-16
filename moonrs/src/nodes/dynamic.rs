use crate::nodes::{statics::StaticNode, *};

pub trait DynamicNode: StaticNode {
    fn update(&mut self);
}

impl DynamicNode for Company {
    fn update(&mut self) {}
}

impl DynamicNode for Exchange {
    fn update(&mut self) {}
}

impl DynamicNode for Currency {
    fn update(&mut self) {}
}

impl DynamicNode for Bond {
    fn update(&mut self) {}
}

impl DynamicNode for TreasuryBill {
    fn update(&mut self) {}
}

impl DynamicNode for Index {
    fn update(&mut self) {}
}

impl DynamicNode for MutualFund {
    fn update(&mut self) {}
}

impl DynamicNode for ETFund {
    fn update(&mut self) {}
}

impl DynamicNode for Equity {
    fn update(&mut self) {}
}

impl DynamicNode for Commodity {
    fn update(&mut self) {}
}

impl DynamicNode for CryptoCurrency {
    fn update(&mut self) {}
}

impl DynamicNode for Options {
    fn update(&mut self) {}
}

impl DynamicNode for Futures {
    fn update(&mut self) {}
}

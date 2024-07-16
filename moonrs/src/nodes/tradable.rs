use crate::nodes::{asset::Asset, *};

pub trait Tradable: Asset {
    fn exchanges(&self) -> Vec<Exchange>;
}

impl Tradable for Currency {
    fn exchanges(&self) -> Vec<Exchange> {
        self.exchanges.clone()
    }
}

impl Tradable for Bond {
    fn exchanges(&self) -> Vec<Exchange> {
        self.exchanges.clone()
    }
}

impl Tradable for TreasuryBill {
    fn exchanges(&self) -> Vec<Exchange> {
        self.exchanges.clone()
    }
}

impl Tradable for MutualFund {
    fn exchanges(&self) -> Vec<Exchange> {
        self.exchanges.clone()
    }
}

impl Tradable for ETFund {
    fn exchanges(&self) -> Vec<Exchange> {
        self.exchanges.clone()
    }
}

impl Tradable for Equity {
    fn exchanges(&self) -> Vec<Exchange> {
        self.exchanges.clone()
    }
}

impl Tradable for CryptoCurrency {
    fn exchanges(&self) -> Vec<Exchange> {
        self.exchanges.clone()
    }
}

impl Tradable for Options {
    fn exchanges(&self) -> Vec<Exchange> {
        self.exchanges.clone()
    }
}

impl Tradable for Futures {
    fn exchanges(&self) -> Vec<Exchange> {
        self.exchanges.clone()
    }
}

use crate::*;

#[derive(Debug, Clone)]
pub struct Rate {
    rate: f64,
    unit: Duration,
}

#[derive(Debug, Clone)]
pub enum Opt {
    Call,
    Put,
}

lazy_static! {
    /// A quite subjective list of non-meme cryptos
    pub static ref SERIOUS_CRYPTOS: HashSet<&'static str> = {
        let mut m = HashSet::new();
        m.insert("BTC");
        m.insert("ETH");
        m.insert("ADA");
        m.insert("DOT");
        m.insert("SOL");
        m.insert("LINK");
        m.insert("BNB");
        m.insert("LTC");
        m.insert("XRP");
        m.insert("XMR");
        m.insert("XLM");
        m
    };
}
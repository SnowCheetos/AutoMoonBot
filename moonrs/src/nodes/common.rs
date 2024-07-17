use crate::nodes::*;

#[derive(Debug, Clone)]
pub struct Exchange {}

#[derive(Debug, Clone)]
pub struct Equity {
    pub symbol: String,
    pub region: String,
    pub exchanges: HashSet<String>,
    pub buffer: TemporalDeque<Aggregate>,
}

impl Exchange {
    pub fn new() {}
}

impl Equity {
    pub fn new(capacity: usize, symbol: String, region: String, exchanges: Vec<String>) -> Self {
        Equity {
            symbol,
            region,
            exchanges: HashSet::from_iter(exchanges.into_iter()),
            buffer: TemporalDeque::new(capacity),
        }
    }
}

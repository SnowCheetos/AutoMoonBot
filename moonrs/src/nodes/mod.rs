pub mod asset;
pub mod coupons;
pub mod derivative;
pub mod dynamic;
pub mod entity;
pub mod ephemeral;
pub mod marketable;
pub mod statics;
use crate::data::{aggregate::*, buffer::*, queue::*};
use std::{
    collections::HashSet,
    hash::Hash,
    time::{Duration, Instant},
};

#[derive(Debug, Clone)]
pub struct Equity {
    symbol: String,
    region: String,
    exchanges: HashSet<String>,
    buffer: TemporalDeque<Aggregate>,
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

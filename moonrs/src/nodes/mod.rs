pub mod asset;
pub mod derivative;
pub mod dynamic;
pub mod entity;
pub mod predictable;
pub mod statics;
pub mod ephemeral;
pub mod tradable;
use crate::data::{aggregate::*, buffer::*, queue::*};
use std::{
    hash::Hash,
    time::{Duration, Instant},
};

#[derive(Debug, Clone)]
pub struct Equity {
    symbol: &'static str,
    region: &'static str,
    buffer: TemporalDeque<Aggregate>,
}
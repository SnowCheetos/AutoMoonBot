pub mod asset;
pub mod derivative;
pub mod dynamic;
pub mod entity;
pub mod fixed;
pub mod statics;
pub mod temporal;
pub mod tradable;

use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
struct Rate {
    value: f64,
    duration: Duration,
}

#[derive(Debug, Clone, Copy)]
struct Quote {
    timestamp: Instant,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
    adjusted: bool,
}

/*
News and sentiments
*/
#[derive(Debug, Clone)]
pub struct Author {
    name: &'static str,
}

#[derive(Debug, Clone)]
pub struct Publisher {
    name: &'static str,
}

#[derive(Debug, Clone)]
pub struct Article {
    title: &'static str,
    summary: &'static str,
    link: &'static str,
    sentiment: f64,
    published: Instant,
    authors: Vec<Author>,
}

/*
Commercial entities
*/
#[derive(Debug, Clone)]
pub struct Company {
    name: &'static str,
    founded: Instant,
}

#[derive(Debug, Clone)]
pub struct Industry {
    name: &'static str,
}

#[derive(Debug, Clone)]
pub struct Sector {
    name: &'static str,
    industries: Vec<Industry>,
}

#[derive(Debug, Clone)]
pub struct Exchange {
    name: &'static str,
    region: &'static str,
}

/*
Financial assets
*/
#[derive(Debug, Clone)]
pub struct Currency {
    symbol: &'static str,
    region: &'static str,
    prices: Vec<Vec<f64>>,
    exchanges: Vec<Exchange>,
}

#[derive(Debug, Clone)]
pub struct Bond {
    symbol: &'static str,
    region: &'static str,
    coupon: Rate,
    exchanges: Vec<Exchange>,
}

#[derive(Debug, Clone)]
pub struct TreasuryBill {
    symbol: &'static str,
    region: &'static str,
    coupon: Rate,
    exchanges: Vec<Exchange>,
}

#[derive(Debug, Clone)]
pub struct MutualFund {
    symbol: &'static str,
    region: &'static str,
    sectors: Vec<Sector>,
    exchanges: Vec<Exchange>,
}

#[derive(Debug, Clone)]
pub struct Index {
    symbol: &'static str,
    region: &'static str,
    sectors: Vec<Sector>,
}

#[derive(Debug, Clone)]
pub struct Equity {
    symbol: &'static str,
    region: &'static str,
    exchanges: Vec<Exchange>,
}

#[derive(Debug, Clone)]
pub struct ETFund {
    symbol: &'static str,
    region: &'static str,
    exchanges: Vec<Exchange>,
}

#[derive(Debug, Clone)]
pub struct Commodity {
    symbol: &'static str,
    region: &'static str,
}

#[derive(Debug, Clone)]
pub struct CryptoCurrency {
    symbol: &'static str,
    region: &'static str,
    exchanges: Vec<Exchange>,
}

#[derive(Debug, Clone)]
pub struct Options {
    symbol: &'static str,
    region: &'static str,
    exchanges: Vec<Exchange>,
}

#[derive(Debug, Clone)]
pub struct Futures {
    symbol: &'static str,
    region: &'static str,
    exchanges: Vec<Exchange>,
}

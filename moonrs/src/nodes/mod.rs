pub mod asset;
pub mod derivative;
pub mod entity;
pub mod fixed;
pub mod temporal;
pub mod tradable;
pub mod dynamic;
pub mod statics;

use std::hash::Hash;

/*
News and sentiments
*/
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Author {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Publisher {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Article {}

/*
Commercial entities
*/
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Company {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Sector {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Industry {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Exchange {}

/*
Financial assets
*/
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Currency {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Bond {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct MutualFund {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Index {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Equity {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct ETFund {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Commodity {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct CryptoCurrency {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Options {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Futures {}

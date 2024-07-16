pub mod asset;
pub mod derivative;
pub mod entity;
pub mod fixed;
pub mod node;
pub mod temporal;
pub mod tradable;

/*
News and sentiments
*/
#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Article {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Publisher {}

/*
Economic entities
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
pub struct ETF {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Commodity {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct CryptoCurrency {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Options {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Futures {}
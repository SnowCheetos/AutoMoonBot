use super::node::DynamicNode;

pub trait Asset: DynamicNode {
    fn symbol(&self) -> String;
    fn region(&self) -> String;
    fn latest(&self) -> Vec<f64>;
    fn quote(&self, aggr: Option<String>) -> f64;
}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Equity {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Bond {}
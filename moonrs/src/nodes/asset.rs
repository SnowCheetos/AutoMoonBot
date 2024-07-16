use crate::nodes::{dynamic::DynamicNode, *};

pub trait Asset: DynamicNode {
    fn symbol(&self) -> &'static str;
    fn region(&self) -> &'static str;
    fn latest(&self) -> Vec<f64>;
    fn quote(&self, aggr: Option<String>) -> f64;
}
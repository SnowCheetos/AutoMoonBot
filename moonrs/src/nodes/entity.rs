use std::time::Instant;

use super::node::StaticNode;

pub trait Entity: StaticNode {
    fn name(&self) -> String;
    fn creation(&self) -> Instant;
}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Exchange {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Company {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Article {}

#[derive(Debug, Eq, PartialEq, Hash, Clone, Copy)]
pub struct Publisher {}

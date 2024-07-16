use std::time::Instant;

use super::node::StaticNode;

pub trait Entity: StaticNode {
    fn name(&self) -> String;
    fn creation(&self) -> Instant;
}
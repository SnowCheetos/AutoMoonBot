use std::time::Instant;

use super::statics::StaticNode;

pub trait Entity: StaticNode {
    fn name(&self) -> String;
    fn creation(&self) -> Instant;
}
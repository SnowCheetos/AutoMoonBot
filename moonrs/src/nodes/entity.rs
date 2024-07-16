use crate::nodes::{statics::StaticNode, *};

pub trait Entity: StaticNode {
    fn creation(&self) -> Instant;
}

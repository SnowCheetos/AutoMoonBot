use crate::nodes::{statics::StaticNode, *};

pub trait Entity: StaticNode {
    fn name(&self) -> String;
    fn creation(&self) -> Instant;
}

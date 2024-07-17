use crate::nodes::{statics::StaticNode, *};

pub trait Entity<T>: StaticNode {
    fn creation(&self) -> Instant;
}

use crate::edges::*;
use crate::nodes::StaticNode;

pub trait StaticEdge: Clone + Send + Sync {
    type Source: StaticNode;
    type Target: StaticNode;

    fn cls(&self) -> &'static str;
}

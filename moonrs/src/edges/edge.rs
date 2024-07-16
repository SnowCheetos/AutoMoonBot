use std::hash::Hash;

use crate::nodes::statics::StaticNode;

pub trait StaticEdge: Eq + Hash + Clone + Send + Sync + Copy {
    type Source: StaticNode;
    type Target: StaticNode;

    fn new(source: Self::Source, target: Self::Target) -> Self;
}

pub trait DynamicEdge: StaticEdge {
    fn update(&mut self);
}

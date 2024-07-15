use std::hash::Hash;

use crate::node::StaticNode;

pub trait StaticEdge: Eq + Hash + Clone + Send + Sync + Copy {
    type Source: StaticNode;
    type Target: StaticNode;
}

pub trait DynamicEdge: StaticEdge {
    fn update(&mut self);
}

enum Edges {}

use std::hash::Hash;

pub trait StaticNode: Eq + Hash + Clone + Send + Sync + Copy {
    fn params(&self);
}

pub trait DynamicNode: StaticNode {
    fn update(&mut self);
}

enum Nodes {}

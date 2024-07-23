use crate::nodes::*;

pub trait DynamicNode<Ix, T>: StaticNode
where
    Ix: Clone + Hash + Eq + PartialOrd,
    T: Clone,
{
    fn update(&mut self, index: Ix, item: T) -> bool;
    fn empty(&self) -> bool;
    fn to_vec(&self) -> Vec<&T>;
    fn first(&self) -> Option<&T>;
    fn last(&self) -> Option<&T>;
    fn between(&self, start: Ix, end: Ix) -> Option<Vec<&T>>;
}

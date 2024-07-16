use crate::nodes::{statics::StaticNode, *};

pub trait DynamicNode<Ix, T>: StaticNode + RingIndexBuffer<Ix, T> 
where
    Ix: Clone + Hash + Eq + PartialOrd,
{
    fn update(&mut self);
}
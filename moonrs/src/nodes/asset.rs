use crate::nodes::{dynamic::DynamicNode, *};

pub trait Asset<T>: DynamicNode<Instant, T> + BlockRingIndexBuffer<Instant, T, f64>
where
    T: Copy + IntoIterator<Item = f64>,
{
    fn symbol(&self) -> &'static str;
    fn region(&self) -> &'static str;
    fn latest(&self) -> Vec<f64>;
    fn quote(&self, aggr: Option<String>) -> f64;
}

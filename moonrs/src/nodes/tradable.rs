use crate::nodes::{asset::Asset, *};

pub trait Tradable<T>: Asset<T>
where
    T: Clone + IntoIterator<Item = f64>,
{
    fn exchanges(&self);
    fn latest(&self) -> Vec<f64>;
}

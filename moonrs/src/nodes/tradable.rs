use super::{asset::Asset, *};

pub trait Tradable: Asset {
    fn exchanges(&self) -> Vec<Exchange>;
}

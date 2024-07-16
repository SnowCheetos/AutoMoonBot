use super::{asset::Asset, Exchange};

pub trait Tradable: Asset {
    fn exchanges(&self) -> Vec<Exchange>;
}

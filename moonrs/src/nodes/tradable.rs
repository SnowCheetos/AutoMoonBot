use super::{asset::Asset, entity::Exchange};

pub trait Tradable: Asset {
    fn exchanges(&self) -> Vec<Exchange>;
}

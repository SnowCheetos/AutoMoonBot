use crate::nodes::{asset::Asset, *};

pub trait Tradable: Asset {
    fn exchanges(&self);
}
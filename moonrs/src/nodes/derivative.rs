use super::{asset::Asset, temporal::Temporal};

pub trait Derivative: Temporal {
    type Underlying: Asset;
}


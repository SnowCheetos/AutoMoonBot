use crate::nodes::{asset::Asset, temporal::Temporal};

pub trait Derivative: Temporal {
    fn spot(&self);
}

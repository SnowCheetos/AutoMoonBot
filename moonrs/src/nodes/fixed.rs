use crate::nodes::{temporal::Temporal, *};

pub trait Fixed: Temporal {
    fn coupon(&self, timespan: Option<Duration>) -> f64;
}
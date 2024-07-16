use std::time::Duration;

use super::temporal::Temporal;

pub trait Fixed: Temporal {
    fn rate(&self, timespan: Duration) -> f64;
}

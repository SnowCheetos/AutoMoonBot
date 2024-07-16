use std::time::Instant;

use super::tradable::Tradable;

pub trait Temporal: Tradable {
    fn expiration(&self) -> Instant;
}


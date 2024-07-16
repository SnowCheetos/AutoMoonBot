use crate::nodes::{tradable::Tradable, *};

pub trait Temporal: Tradable {
    fn expiration(&self) -> Instant;
}


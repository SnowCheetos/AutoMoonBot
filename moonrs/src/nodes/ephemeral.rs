use crate::nodes::{marketable::Marketable, *};

/// ...
pub trait Ephemeral: Marketable {
    fn expiration(&self) -> &Instant;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {}
}

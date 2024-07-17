use crate::nodes::{marketable::Marketable, *};

/// ...
pub trait Ephemeral<T>: Marketable<T>
where
    T: Clone + IntoIterator<Item = f64>,
{
    fn expiration(&self) -> &Instant;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {}
}

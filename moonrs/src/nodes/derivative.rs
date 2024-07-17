use crate::nodes::{ephemeral::Ephemeral, *};

pub trait Derivative<T>: Ephemeral<T>
where
    T: Clone + IntoIterator<Item = f64>,
{
    fn spot(&self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {}
}

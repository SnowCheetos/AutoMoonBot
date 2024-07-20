use crate::nodes::*;

/// ...
pub trait Derivative: Ephemeral {
    fn spot(&self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {}
}

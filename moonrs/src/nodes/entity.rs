use crate::nodes::{statics::StaticNode, *};

#[doc = r#"..."#]
pub trait Entity<K, V>: StaticNode where
K: Clone + Hash + Eq,
V: Clone,
{
    type Buffer: MapBuffer<K, V>;
    fn creation(&self) -> &Instant;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {}
}

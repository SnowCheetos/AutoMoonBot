use crate::nodes::*;

/// ...
pub trait Entity<K, V>: StaticNode
where
    K: Clone + Hash + Eq,
    V: Clone,
{
    type MetaData: MapBuffer<K, V>;

    fn metadata(&self) -> &Self::MetaData;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {}
}

use crate::nodes::*;

/// An asset
pub trait Asset<T>: DynamicNode<T>
where
    T: Clone + IntoIterator<Item = f64>,
{
    type Buffer: BlockRingIndexBuffer<Instant, T, f64>;

    fn symbol(&self) -> &String;
    fn region(&self) -> &String;
    fn quote(&self) -> Option<f64>;
}

impl Asset<Aggregate> for Equity {
    type Buffer = TemporalDeque<Aggregate>;

    fn symbol(&self) -> &String {
        &self.symbol
    }

    fn region(&self) -> &String {
        &self.region
    }

    fn quote(&self) -> Option<f64> {
        Some(self.last()?.close())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equity() {
        let symbol = "EQU".to_string();
        let region = "MOON".to_string();
        let equity = Equity::new(10, symbol, region, Vec::new());

        assert_eq!(equity.symbol(), "EQU");
        assert_eq!(equity.region(), "MOON");
    }
}

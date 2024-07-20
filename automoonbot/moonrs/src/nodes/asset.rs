use crate::nodes::*;

/// An asset is anything of **value** that can be **owned** or **controlled**.
/// Note the emphasis of ownership here, something that cannot be owned, even
/// if it carries value, cannot be considered an asset. For instance, the **S&P 500**
/// itself is NOT considered an asset, althought it behaves much like a regular
/// stock, one cannot go ahead and purchase a share of the **S&P 500** itself
/// (not including ETFs such as **VOO**, which are considered an independent node).
/// On the other hand, **Crude Oil** IS considered an asset (*but not marketable*), 
/// although it's pretty hard to imagine someone obtaining barrels of actual crude oil, 
/// you **can** technically purchase it. 
/// 
/// All assets are required to have 
pub trait Asset: DynamicNode<Aggregate> {
    type Buffer: BlockRingIndexBuffer<Instant, Aggregate, f64>;

    fn symbol(&self) -> &String;
    fn region(&self) -> &String;
    fn quote(&self) -> Option<f64>;
}

impl Asset for Equity {
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

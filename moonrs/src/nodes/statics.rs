use crate::nodes::*;

/// The base trait that all nodes must implement.
pub trait StaticNode: Clone + Send + Sync {
    /// Returns the class name of the node. 
    /// 
    /// Returned string must be unique and match that of the struct name itself.
    /// 
    /// # Examples
    /// ```rust
    /// impl StaticNode for SomeNode {
    /// fn cls(&self) -> &'static str {
    /// "SomeNode"
    /// }
    /// }
    /// let node = SomeNode::new(...);
    /// assert_eq!(node.cls(), "Equity");
    /// ```
    fn cls(&self) -> &'static str;
}

impl StaticNode for Equity {
    fn cls(&self) -> &'static str {
        "Equity"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equity() {
        let symbol = "EQU".to_owned();
        let region = "MOON".to_owned();
        let equity = Equity::new(10, symbol, region, Vec::new());

        assert_eq!(equity.cls(), "Equity");
    }
}

use crate::nodes::*;

/// Static node
pub trait StaticNode: Clone + Send + Sync {
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

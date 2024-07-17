use crate::nodes::*;

#[doc = r#"..."#]
pub trait StaticNode: Clone + Send + Sync {
    type Buffer: DataBuffer;

    fn id(&self) -> &'static str;
    fn buffer(&self) -> &Self::Buffer;
}

impl StaticNode for Equity {
    type Buffer = TemporalDeque<Aggregate>;

    fn id(&self) -> &'static str {
        "Equity"
    }

    fn buffer(&self) -> &Self::Buffer {
        &self.buffer
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

        assert_eq!(equity.id(), "Equity");
        assert!(equity.buffer().empty());
    }
}

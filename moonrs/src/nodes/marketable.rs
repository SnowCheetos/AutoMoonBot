use crate::nodes::*;

/// ...
pub trait Marketable<T>: Asset<T>
where
    T: Clone + IntoIterator<Item = f64>,
{
    fn exchanges(&self) -> Vec<String>;
}

impl Marketable<Aggregate> for Equity {
    fn exchanges(&self) -> Vec<String> {
        self.exchanges.clone().into_iter().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equity() {
        let symbol = "EQU".to_string();
        let region = "MOON".to_string();
        let exchanges = vec!["NYSE".to_string(), "NASDAQ".to_string()];
        let equity = Equity::new(10, symbol, region, exchanges.clone());

        assert_eq!(equity.exchanges().len(), exchanges.len());
        assert!(equity.exchanges().contains(&exchanges[0]));
        assert!(equity.exchanges().contains(&exchanges[1]));
    }
}

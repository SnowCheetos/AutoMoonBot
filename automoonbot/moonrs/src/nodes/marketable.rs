use crate::nodes::*;

/// ...
pub trait Marketable: Asset {
    fn exchanges(&self) -> Vec<String>;
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_equity() {
//         let symbol = "EQU".to_string();
//         let region = "MOON".to_string();
//         let exchanges = vec!["NYSE".to_string(), "NASDAQ".to_string()];
//         let equity = Equity::new(10, symbol, region, exchanges.clone());

//         assert_eq!(equity.exchanges().len(), exchanges.len());
//         assert!(equity.exchanges().contains(&exchanges[0]));
//         assert!(equity.exchanges().contains(&exchanges[1]));
//     }
// }

use crate::nodes::*;

/// ...
pub trait FixedIncome: Marketable {
    fn interest(&self, timespan: Option<Duration>) -> f64;
    fn yields(&self, timespan: Option<Duration>) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {}
}

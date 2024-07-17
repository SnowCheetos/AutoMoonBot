use crate::nodes::{marketable::Marketable, *};

#[doc = r#"A fixed-income (coupon) asset is one with consistent `interest` payments at a fixed interval.
However, it\'s still considered a dynamic asset since the market prices for the asset is not fixed, 
which is what determines the `yield` of the asset, which is defined as
```math
yield (pct%/period) = interest_rate / market_price
```
"#]
pub trait FixedIncome<T>: Marketable<T>
where
    T: Clone + IntoIterator<Item = f64>,
{
    fn interest(&self, timespan: Option<Duration>) -> f64;
    fn yields(&self, timespan: Option<Duration>) -> f64;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {}
}

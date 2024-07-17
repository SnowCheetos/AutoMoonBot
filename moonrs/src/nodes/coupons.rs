use crate::nodes::*;

/// ...
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

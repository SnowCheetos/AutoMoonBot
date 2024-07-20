

pub trait Aggregates: Iterator + IntoIterator {
    fn to_vec(&self) -> Vec<f64>;
}
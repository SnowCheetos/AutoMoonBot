use crate::data::*;
use statrs::statistics::Statistics;

#[derive(Debug, Clone, Copy)]
pub struct Aggregate {
    timestamp: Instant,
    duration: Duration,
    adjusted: bool,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    volume: f64,
}

pub struct AggregateIterator {
    aggregate: Aggregate,
    index: usize,
}

impl Iterator for AggregateIterator {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        let result = match self.index {
            0 => Some(self.aggregate.open),
            1 => Some(self.aggregate.high),
            2 => Some(self.aggregate.low),
            3 => Some(self.aggregate.close),
            4 => Some(self.aggregate.volume),
            _ => None,
        };
        self.index += 1;
        result
    }
}

impl IntoIterator for Aggregate {
    type Item = f64;
    type IntoIter = AggregateIterator;

    fn into_iter(self) -> Self::IntoIter {
        AggregateIterator {
            aggregate: self,
            index: 0,
        }
    }
}

impl Aggregate {
    pub fn new(
        timestamp: Instant,
        duration: Duration,
        adjusted: bool,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
    ) -> Self {
        Aggregate {
            timestamp,
            duration,
            adjusted,
            open,
            high,
            low,
            close,
            volume,
        }
    }

    pub fn timestamp(&self) -> Instant {
        self.timestamp
    }

    pub fn duration(&self) -> Duration {
        self.duration
    }

    pub fn adjusted(&self) -> bool {
        self.adjusted
    }

    pub fn open(&self) -> f64 {
        self.open
    }

    pub fn high(&self) -> f64 {
        self.high
    }

    pub fn low(&self) -> f64 {
        self.low
    }

    pub fn close(&self) -> f64 {
        self.close
    }

    pub fn volume(&self) -> f64 {
        self.volume
    }

    pub fn to_vec(&self) -> Vec<f64> {
        self.into_iter().collect()
    }

    pub fn amean(&self) -> f64 {
        self.to_vec().as_slice()[0..4].mean()
    }

    pub fn gmean(&self) -> f64 {
        self.to_vec().as_slice()[0..4].geometric_mean()
    }

    pub fn hmean(&self) -> f64 {
        self.to_vec().as_slice()[0..4].harmonic_mean()
    }

    pub fn std(&self) -> f64 {
        self.to_vec().as_slice()[0..4].std_dev()
    }

    pub fn var(&self) -> f64 {
        self.to_vec().as_slice()[0..4].variance()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn test_aggregate_new() {
        let timestamp = Instant::now();
        let duration = Duration::new(60, 0);
        let aggregate =
            Aggregate::new(timestamp, duration, true, 100.0, 110.0, 90.0, 105.0, 1000.0);

        assert_eq!(aggregate.timestamp(), timestamp);
        assert_eq!(aggregate.duration(), duration);
        assert_eq!(aggregate.adjusted(), true);
        assert_eq!(aggregate.open(), 100.0);
        assert_eq!(aggregate.high(), 110.0);
        assert_eq!(aggregate.low(), 90.0);
        assert_eq!(aggregate.close(), 105.0);
        assert_eq!(aggregate.volume(), 1000.0);
    }

    #[test]
    fn test_aggregate_to_vec() {
        let aggregate = Aggregate::new(
            Instant::now(),
            Duration::new(60, 0),
            true,
            100.0,
            110.0,
            90.0,
            105.0,
            1000.0,
        );
        let vec = aggregate.to_vec();
        assert_eq!(vec, vec![100.0, 110.0, 90.0, 105.0, 1000.0]);
    }

    #[test]
    fn test_aggregate_amean() {
        let aggregate = Aggregate::new(
            Instant::now(),
            Duration::new(60, 0),
            true,
            100.0,
            110.0,
            90.0,
            105.0,
            1000.0,
        );
        let mean = aggregate.amean();
        assert_eq!(mean, 101.25f64);
    }

    #[test]
    fn test_aggregate_gmean() {
        let aggregate = Aggregate::new(
            Instant::now(),
            Duration::new(60, 0),
            true,
            1.0,
            3.0,
            9.0,
            27.0,
            1000.0,
        );
        let gmean = aggregate.gmean();
        assert!((gmean - 5.196152422706632f64).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_hmean() {
        let aggregate = Aggregate::new(
            Instant::now(),
            Duration::new(60, 0),
            true,
            1.0,
            2.0,
            4.0,
            4.0,
            1000.0,
        );
        let hmean = aggregate.hmean();
        assert!((hmean - 2.0f64).abs() < 1e-10);
    }

    #[test]
    fn test_aggregate_std() {
        let aggregate = Aggregate::new(
            Instant::now(),
            Duration::new(60, 0),
            true,
            100.0,
            110.0,
            90.0,
            105.0,
            1000.0,
        );
        let cmp_std = aggregate.std();
        let mnl_std = ((f64::powi(100.0 - 101.25, 2)
            + f64::powi(110.0 - 101.25, 2)
            + f64::powi(90.0 - 101.25, 2)
            + f64::powi(105.0 - 101.25, 2))
            / 4f64)
            .sqrt();

        assert!((cmp_std - mnl_std).abs() < 1e-4);
    }

    #[test]
    fn test_aggregate_var() {
        let aggregate = Aggregate::new(
            Instant::now(),
            Duration::new(60, 0),
            true,
            100.0,
            110.0,
            90.0,
            105.0,
            1000.0,
        );
        let var = aggregate.var();
        assert!((var - 72.9167).abs() < 1e-4);
    }
}

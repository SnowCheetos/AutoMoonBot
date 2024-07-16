use nalgebra::Vector4;
use statrs::statistics::Statistics;

use crate::utils::*;

#[derive(Debug, Clone)]
pub struct Aggregate {
    timestamp: Instant,
    period: Duration,
    volume: f64,
    adjusted: bool,
    prices: Vector4<f64>,
}

impl Aggregate {
    pub fn new(
        timestamp: Instant,
        period: Duration,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        adjusted: bool,
    ) -> Self {
        Aggregate {
            timestamp,
            period,
            volume,
            adjusted,
            prices: Vector4::new(open, high, low, close),
        }
    }

    pub fn timestamp(&self) -> Instant {
        self.timestamp
    }

    pub fn period(&self) -> Duration {
        self.period
    }

    pub fn adjusted(&self) -> bool {
        self.adjusted
    }

    pub fn open(&self) -> f64 {
        self.prices[0]
    }

    pub fn high(&self) -> f64 {
        self.prices[1]
    }

    pub fn low(&self) -> f64 {
        self.prices[2]
    }

    pub fn close(&self) -> f64 {
        self.prices[3]
    }

    pub fn volume(&self) -> f64 {
        self.volume
    }

    pub fn min(&self) -> f64 {
        self.low()
    }

    pub fn max(&self) -> f64 {
        self.high()
    }

    pub fn mean(&self) -> f64 {
        self.prices.mean()
    }

    pub fn gmean(&self) -> f64 {
        self.prices.geometric_mean()
    }

    pub fn hmean(&self) -> f64 {
        self.prices.harmonic_mean()
    }

    pub fn qmean(&self) -> f64 {
        self.prices.quadratic_mean()
    }

    pub fn pmean(&self, p: f64) -> f64 {
        if p == 0.0 {
            self.gmean()
        } else if p == 1.0 {
            self.mean()
        } else {
            let sum_pth_power: f64 = self.prices.iter().map(|&x| x.powf(p)).sum();
            (sum_pth_power / self.prices.len() as f64).powf(1.0 / p)
        }
    }

    pub fn std(&self) -> f64 {
        self.prices.std_dev()
    }

    pub fn var(&self) -> f64 {
        self.prices.variance()
    }
}

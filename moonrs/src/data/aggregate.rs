use std::time::Instant;

pub struct Aggregate {
    timestamp: Instant,
    open: f64,
    high: f64,
    low: f64,
    close: f64,
    adjusted: f64,
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
            4 => Some(self.aggregate.adjusted),
            5 => Some(self.aggregate.volume),
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
        &self,
        timestamp: Instant,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        adjusted: f64,
        volume: f64,
    ) -> Self {
        Aggregate {
            timestamp,
            open,
            high,
            low,
            close,
            adjusted,
            volume,
        }
    }

    pub fn index(&self) -> Instant {
        self.timestamp
    }
}

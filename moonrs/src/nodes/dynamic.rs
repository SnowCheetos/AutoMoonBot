use crate::nodes::*;

/// ...
pub trait DynamicNode<T>: StaticNode
where
    T: Clone,
{
    type Buffer: RingIndexBuffer<Instant, T>;

    fn update(&mut self, item: T) -> bool;
    fn first(&self) -> Option<&T>;
    fn last(&self) -> Option<&T>;
}

impl DynamicNode<Aggregate> for Equity {
    type Buffer = TemporalDeque<Aggregate>;

    fn update(&mut self, item: Aggregate) -> bool {
        self.buffer.push(item.timestamp(), item)
    }

    fn first(&self) -> Option<&Aggregate> {
        self.buffer.first()
    }

    fn last(&self) -> Option<&Aggregate> {
        self.buffer.last()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equity() {
        let symbol = "EQU".to_string();
        let region = "MOON".to_string();
        let mut equity = Equity::new(2, symbol, region, Vec::new());
        let now = Instant::now();
        let span = Duration::new(60, 0);
        let next = now + span;
        let aggregate1 = Aggregate::new(now, span, true, 1.0, 2.0, 0.5, 1.5, 100.0);
        let aggregate2 = Aggregate::new(next, span, false, 1.1, 2.1, 0.6, 1.6, 200.0);
        let aggregate3 = Aggregate::new(next + span, span, false, 1.3, 2.2, 0.7, 1.5, 150.0);

        let front = equity.first();
        let back = equity.last();
        assert!(front.is_none());
        assert!(back.is_none());

        let success = equity.update(aggregate1);
        assert!(success);
        let front = equity.first();
        let back = equity.last();
        assert!(front.is_some_and(|item| item.timestamp() == now));
        assert!(back.is_some_and(|item| item.timestamp() == now));

        let success = equity.update(aggregate2);
        assert!(success);
        let front = equity.first();
        let back = equity.last();
        assert!(front.is_some_and(|item| item.timestamp() == now));
        assert!(back.is_some_and(|item| item.timestamp() == next));

        let success = equity.update(aggregate2);
        assert!(!success);
        let front = equity.first();
        let back = equity.last();
        assert!(front.is_some_and(|item| item.timestamp() == now));
        assert!(back.is_some_and(|item| item.timestamp() == next));

        let success = equity.update(aggregate3);
        assert!(success);
        let front = equity.first();
        let back = equity.last();
        assert!(front.is_some_and(|item| item.timestamp() == next));
        assert!(back.is_some_and(|item| item.timestamp() == next + span));
    }
}

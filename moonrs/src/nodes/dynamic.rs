use crate::nodes::{statics::StaticNode, *};

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

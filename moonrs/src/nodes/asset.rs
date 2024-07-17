use crate::nodes::{dynamic::DynamicNode, *};

pub trait Asset<T>: DynamicNode<T>
where
    T: Clone + IntoIterator<Item = f64>,
{
    type Buffer: BlockRingIndexBuffer<Instant, T, f64>;

    fn symbol(&self) -> &'static str;
    fn region(&self) -> &'static str;
    fn quote(&self) -> Option<f64>;
}

impl Asset<Aggregate> for Equity {
    type Buffer = TemporalDeque<Aggregate>;

    fn symbol(&self) -> &'static str {
        self.symbol
    }

    fn region(&self) -> &'static str {
        self.region
    }

    fn quote(&self) -> Option<f64> {
        Some(self.buffer.last()?.close())
    }
}

use crate::nodes::*;

pub trait StaticNode: Clone + Send + Sync {
    type Buffer: DataBuffer;

    fn id(&self) -> &'static str;
    fn buffer(&self) -> &Self::Buffer;
}

impl StaticNode for Equity {
    type Buffer = TemporalDeque<Aggregate>;

    fn id(&self) -> &'static str {
        "Equity"
    }

    fn buffer(&self) -> &Self::Buffer {
        &self.buffer
    }
}

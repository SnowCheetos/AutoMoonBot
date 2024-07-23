use crate::edges::*;

pub trait StaticEdge: Send + Sync {
    fn value(&self) -> f64;
    fn src_index(&self) -> &NodeIndex;
    fn tgt_index(&self) -> &NodeIndex;
}
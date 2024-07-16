use crate::nodes::*;

pub trait StaticNode: Clone + Send + Sync {
    fn name(&self) -> &'static str;
}
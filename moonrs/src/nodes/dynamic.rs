use crate::nodes::{statics::StaticNode, *};

pub trait DynamicNode: StaticNode {
    fn update(&mut self);
}
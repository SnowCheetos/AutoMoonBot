use crate::edges::statics::StaticEdge;

pub trait DynamicEdge: StaticEdge {
    fn update(&mut self);
}

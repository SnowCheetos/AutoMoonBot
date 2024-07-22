use crate::edges::*;

pub trait ForwardDynEdge<S, T, X>: StaticEdge
where
    S: StaticNode,
    T: DynamicNode<X>,
    X: Clone,
{
    fn fowrard_corr(&self, src: &S, tgt: &T);
}

mod tests {
    use super::*;

    #[test]
    fn test() {}
}

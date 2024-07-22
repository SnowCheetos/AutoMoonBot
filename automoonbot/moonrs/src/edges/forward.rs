use crate::edges::*;

pub trait ForwardDynEdge<S, T, Ix, X>: StaticEdge
where
    S: StaticNode,
    T: DynamicNode<Ix, X>,
    Ix: Clone + Hash + Eq + PartialOrd,
    X: Clone,
{
    fn fowrard_corr(&self, src: &S, tgt: &T);
}

mod tests {
    use super::*;

    #[test]
    fn test() {}
}

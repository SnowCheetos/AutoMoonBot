use crate::edges::*;

pub trait BackwardDynEdge<S, T, Ix, X>: StaticEdge
where
    S: DynamicNode<Ix, X>,
    T: StaticNode,
    Ix: Clone + Hash + Eq + PartialOrd,
    X: Clone,
{
    fn backward_corr(&self, src: &S, tgt: &T);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {}
}

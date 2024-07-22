use crate::edges::*;

pub trait BackwardDynEdge<S, T, X>: StaticEdge
where
    S: DynamicNode<X>,
    T: StaticNode,
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

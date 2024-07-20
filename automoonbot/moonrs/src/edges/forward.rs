use crate::edges::*;

/// Forward dynamic edge, where a source node affects
/// the dynamic target node. For example, a news post
/// caused the stock prices to rise.
///
/// ```math
/// S_t \rightarrow {\delta T \over \delta t}
/// ```
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

use crate::edges::*;

/// Mutual dynamic edge, where a dynamic source node
/// affects the dynamic target node. For example, an
/// upward momentum in ^VIX is observed with an down
/// trend in the S&P 500 in the same period of time.
///
/// ```math
/// {\delta S \over \delta t} \leftrightarrow {\delta T \over \delta t}
/// ```
pub trait MutualDynEdge<S, T, X, Y>: ForwardDynEdge<S, T, X> + BackwardDynEdge<S, T, Y>
where
    S: DynamicNode<Y>,
    T: DynamicNode<X>,
    X: Clone,
    Y: Clone,
{
    fn mutual_corr(&self, src: &S, tgt: &T);
}

mod tests {
    use super::*;

    #[test]
    fn test() {}
}

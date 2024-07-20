use crate::edges::*;

/// Backward dynamic edge, where a dnyamic source node
/// affects the target node. For example, good performance
/// in a company resulted in the announcement of dividends.
///
/// ```math
/// {\delta S \over \delta t} \rightarrow T_t
/// ```
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

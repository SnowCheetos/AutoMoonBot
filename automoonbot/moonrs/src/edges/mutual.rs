use crate::edges::*;

pub trait MutualDynEdge<S, T, Ix, X, Y>: StaticEdge
where
    S: DynamicNode<Ix, X>,
    T: DynamicNode<Ix, Y>,
    Ix: Clone + Hash + Eq + PartialOrd,
    X: Clone,
    Y: Clone,
{
    fn covariance(&self) -> Option<&na::DMatrix<f64>>;
    fn correlation(&self) -> Option<&na::DMatrix<f64>>;
    fn compute_covariance(&self, src: &S, tgt: &T) -> Option<na::DMatrix<f64>>;
    fn compute_correlation(&self, src: &S, tgt: &T) -> Option<na::DMatrix<f64>>;
    fn update(&mut self, src: &S, tgt: &T);
}

impl MutualDynEdge<ETFs, Indices, Instant, PriceAggregate, PriceAggregate> for Mirrors {
    fn covariance(&self) -> Option<&na::DMatrix<f64>> {
        self.covariance.as_ref()
    }

    fn correlation(&self) -> Option<&na::DMatrix<f64>> {
        self.correlation.as_ref()
    }

    fn compute_covariance(&self, src: &ETFs, tgt: &Indices) -> Option<na::DMatrix<f64>> {
        let src_mat = src.mat()?;
        let tgt_mat = tgt.mat()?;
        compute_covariance(src_mat, tgt_mat)
    }

    fn compute_correlation(&self, src: &ETFs, tgt: &Indices) -> Option<na::DMatrix<f64>> {
        let src_mat = src.mat()?;
        let tgt_mat = tgt.mat()?;
        compute_correlation(src_mat, tgt_mat)
    }

    fn update(&mut self, src: &ETFs, tgt: &Indices) {
        if let (Some(covariance), Some(correlation)) = (
            self.compute_covariance(src, tgt),
            self.compute_correlation(src, tgt),
        ) {
            self.covariance = Some(covariance);
            self.correlation = Some(correlation);
        }
    }
}

impl MutualDynEdge<Equity, Equity, Instant, PriceAggregate, PriceAggregate> for Influences {
    fn covariance(&self) -> Option<&na::DMatrix<f64>> {
        self.covariance.as_ref()
    }

    fn correlation(&self) -> Option<&na::DMatrix<f64>> {
        self.correlation.as_ref()
    }

    fn compute_covariance(&self, src: &Equity, tgt: &Equity) -> Option<na::DMatrix<f64>> {
        let src_mat = src.mat()?;
        let tgt_mat = tgt.mat()?;
        compute_covariance(src_mat, tgt_mat)
    }

    fn compute_correlation(&self, src: &Equity, tgt: &Equity) -> Option<na::DMatrix<f64>> {
        let src_mat = src.mat()?;
        let tgt_mat = tgt.mat()?;
        compute_correlation(src_mat, tgt_mat)
    }

    fn update(&mut self, src: &Equity, tgt: &Equity) {
        if let (Some(covariance), Some(correlation)) = (
            self.compute_covariance(src, tgt),
            self.compute_correlation(src, tgt),
        ) {
            self.covariance = Some(covariance);
            self.correlation = Some(correlation);
        }
    }
}

impl MutualDynEdge<Equity, Options, Instant, PriceAggregate, OptionsAggregate> for Derives {
    fn covariance(&self) -> Option<&na::DMatrix<f64>> {
        self.covariance.as_ref()
    }

    fn correlation(&self) -> Option<&na::DMatrix<f64>> {
        self.correlation.as_ref()
    }

    fn compute_covariance(&self, src: &Equity, tgt: &Options) -> Option<na::DMatrix<f64>> {
        let src_mat = src.mat()?;
        let tgt_mat = tgt.mat()?;
        compute_covariance(src_mat, tgt_mat)
    }

    fn compute_correlation(&self, src: &Equity, tgt: &Options) -> Option<na::DMatrix<f64>> {
        let src_mat = src.mat()?;
        let tgt_mat = tgt.mat()?;
        compute_correlation(src_mat, tgt_mat)
    }

    fn update(&mut self, src: &Equity, tgt: &Options) {
        if let (Some(covariance), Some(correlation)) = (
            self.compute_covariance(src, tgt),
            self.compute_correlation(src, tgt),
        ) {
            self.covariance = Some(covariance);
            self.correlation = Some(correlation);
        }
    }
}

mod tests {
    use super::*;

    #[test]
    fn test() {}
}
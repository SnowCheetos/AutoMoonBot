use crate::*;

#[derive(Debug, Clone)]
pub struct Rate {
    rate: f64,
    unit: Duration,
}

#[derive(Debug, Clone)]
pub enum Opt {
    Call,
    Put,
}

pub fn compute_covariance(src_mat: na::DMatrix<f64>, tgt_mat: na::DMatrix<f64>) -> Option<na::DMatrix<f64>> {
    if src_mat.nrows() != tgt_mat.nrows() {
        return None;
    }
    let n = src_mat.nrows() as f64;
    let src_mean = src_mat.column_mean();
    let tgt_mean = tgt_mat.column_mean();
    let centered_src =
        src_mat.clone() - na::DMatrix::from_columns(&vec![src_mean.clone(); src_mat.nrows()]);
    let centered_tgt =
        tgt_mat.clone() - na::DMatrix::from_columns(&vec![tgt_mean.clone(); tgt_mat.nrows()]);
    let covariance = (centered_src.transpose() * centered_tgt) / (n - 1.0);
    Some(covariance)
}

pub fn compute_correlation(src_mat: na::DMatrix<f64>, tgt_mat: na::DMatrix<f64>) -> Option<na::DMatrix<f64>> {
    if src_mat.nrows() != tgt_mat.nrows() {
        return None;
    }
    let n = src_mat.nrows() as f64;
    let src_mean = src_mat.column_mean();
    let tgt_mean = tgt_mat.column_mean();
    let centered_src =
        src_mat.clone() - na::DMatrix::from_columns(&vec![src_mean.clone(); src_mat.nrows()]);
    let centered_tgt =
        tgt_mat.clone() - na::DMatrix::from_columns(&vec![tgt_mean.clone(); tgt_mat.nrows()]);
    let src_std = centered_src.column_variance().map(|var| var.sqrt());
    let tgt_std = centered_tgt.column_variance().map(|var| var.sqrt());
    let standardized_src =
        centered_src.component_div(&na::DMatrix::from_columns(&vec![
            src_std.clone();
            src_mat.nrows()
        ]));
    let standardized_tgt =
        centered_tgt.component_div(&na::DMatrix::from_columns(&vec![
            tgt_std.clone();
            tgt_mat.nrows()
        ]));
    let correlation = (standardized_src.transpose() * standardized_tgt) / (n - 1.0);
    Some(correlation)
}

pub fn get_company(symbol: String) -> Option<String> {
    todo!()
}

pub fn get_symbols(company: String) -> HashSet<String> {
    todo!()
}

lazy_static! {
    /// A quite subjective list of non-meme cryptos
    pub static ref SERIOUS_CRYPTOS: HashSet<&'static str> = {
        let mut m = HashSet::new();
        m.insert("BTC");
        m.insert("ETH");
        m.insert("ADA");
        m.insert("DOT");
        m.insert("SOL");
        m.insert("LINK");
        m.insert("BNB");
        m.insert("LTC");
        m.insert("XRP");
        m.insert("XMR");
        m.insert("XLM");
        m
    };
}
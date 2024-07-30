use crate::edges::*;

#[derive(Debug)]
pub enum EdgeType {
    TestEdge(TestEdge),
    Published(Published),
    Mentioned(Mentioned),
    Referenced(Referenced),
    Issues(Issues),
    Influences(Influences),
    Derives(Derives),
}

impl From<TestEdge> for EdgeType {
    fn from(edge: TestEdge) -> Self {
        EdgeType::TestEdge(edge)
    }
}

impl From<Published> for EdgeType {
    fn from(edge: Published) -> Self {
        EdgeType::Published(edge)
    }
}

impl From<Mentioned> for EdgeType {
    fn from(edge: Mentioned) -> Self {
        EdgeType::Mentioned(edge)
    }
}

impl From<Referenced> for EdgeType {
    fn from(edge: Referenced) -> Self {
        EdgeType::Referenced(edge)
    }
}

impl From<Issues> for EdgeType {
    fn from(edge: Issues) -> Self {
        EdgeType::Issues(edge)
    }
}

impl From<Influences> for EdgeType {
    fn from(edge: Influences) -> Self {
        EdgeType::Influences(edge)
    }
}

impl From<Derives> for EdgeType {
    fn from(edge: Derives) -> Self {
        EdgeType::Derives(edge)
    }
}

#[derive(Debug)]
pub struct Published {
    pub(super) src_index: NodeIndex,
    pub(super) tgt_index: NodeIndex,
}

#[derive(Debug)]
pub struct Mentioned {
    pub(super) src_index: NodeIndex,
    pub(super) tgt_index: NodeIndex,
}

#[derive(Debug)]
pub struct Referenced {
    pub(super) src_index: NodeIndex,
    pub(super) tgt_index: NodeIndex,
    pub(super) sentiment: f64,
}

#[derive(Debug)]
pub struct Issues {
    pub(super) src_index: NodeIndex,
    pub(super) tgt_index: NodeIndex,
    pub(super) covariance: Option<na::DMatrix<f64>>,
    pub(super) correlation: Option<na::DMatrix<f64>>,
}

#[derive(Debug)]
pub struct Mirrors {
    pub(super) src_index: NodeIndex,
    pub(super) tgt_index: NodeIndex,
    pub(super) covariance: Option<na::DMatrix<f64>>,
    pub(super) correlation: Option<na::DMatrix<f64>>,
}

#[derive(Debug)]
pub struct Influences {
    pub(super) src_index: NodeIndex,
    pub(super) tgt_index: NodeIndex,
    pub(super) covariance: Option<na::DMatrix<f64>>,
    pub(super) correlation: Option<na::DMatrix<f64>>,
}

#[derive(Debug)]
pub struct Derives {
    pub(super) src_index: NodeIndex,
    pub(super) tgt_index: NodeIndex,
    pub(super) covariance: Option<na::DMatrix<f64>>,
    pub(super) correlation: Option<na::DMatrix<f64>>,
}

impl Published {
    pub fn try_new(
        src_index: NodeIndex,
        tgt_index: NodeIndex,
        publisher: &Publisher,
        article: &Article,
    ) -> Option<Self> {
        if src_index == tgt_index {
            return None;
        }

        if publisher.name() == article.publisher() {
            Some(Published {
                src_index,
                tgt_index,
            })
        } else {
            None
        }
    }
}

impl Mentioned {
    pub fn try_new(
        src_index: NodeIndex,
        tgt_index: NodeIndex,
        article: &Article,
        company: &Company,
    ) -> Option<Self> {
        if src_index == tgt_index {
            return None;
        }

        if let Some(_) = article.ticker_intersect(company.symbols()) {
            Some(Mentioned {
                src_index,
                tgt_index,
            })
        } else {
            None
        }
    }
}

impl Referenced {
    pub fn try_new(
        src_index: NodeIndex,
        tgt_index: NodeIndex,
        article: &Article,
        equity: &Equity,
    ) -> Option<Self> {
        if src_index == tgt_index {
            return None;
        }

        if let Some(sentiment) = article.ticker_sentiment(equity.name().clone()) {
            Some(Referenced {
                src_index,
                tgt_index,
                sentiment,
            })
        } else {
            None
        }
    }

    pub fn sentiment(&self) -> f64 {
        self.sentiment
    }
}

impl Issues {
    pub fn try_new(
        src_index: NodeIndex,
        tgt_index: NodeIndex,
        company: &Company,
        equity: &Equity,
    ) -> Option<Self> {
        if src_index == tgt_index {
            return None;
        }

        if company.symbols().contains(equity.name()) {
            let covariance = compute_covariance(company.mat()?, equity.mat()?);
            let correlation = compute_correlation(company.mat()?, equity.mat()?);
            Some(Issues {
                src_index,
                tgt_index,
                covariance,
                correlation,
            })
        } else {
            None
        }
    }
}

impl Influences {
    pub fn try_new(
        src_index: NodeIndex,
        tgt_index: NodeIndex,
        src_node: &Equity,
        tgt_node: &Equity,
    ) -> Option<Self> {
        if src_index == tgt_index {
            return None;
        }

        let covariance = compute_covariance(src_node.mat()?, tgt_node.mat()?);
        let correlation = compute_correlation(src_node.mat()?, tgt_node.mat()?);
        Some(Influences {
            src_index,
            tgt_index,
            covariance,
            correlation,
        })
    }
}

impl Derives {
    pub fn try_new(
        src_index: NodeIndex,
        tgt_index: NodeIndex,
        equity: &Equity,
        option: &Options,
    ) -> Option<Self> {
        if src_index == tgt_index {
            return None;
        }

        if option.underlying() == equity.name() {
            let covariance = compute_covariance(equity.mat()?, option.mat()?);
            let correlation = compute_correlation(equity.mat()?, option.mat()?);
            Some(Derives {
                src_index,
                tgt_index,
                covariance,
                correlation,
            })
        } else {
            None
        }
    }
}

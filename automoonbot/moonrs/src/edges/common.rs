use crate::edges::*;

#[derive(Debug)]
pub enum EdgeType {
    TestEdge(TestEdge),
    Published(Published),
    Mentioned(Mentioned),
    Referenced(Referenced),
    Issues(Issues),
    Mirrors(Mirrors),
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

impl From<Mirrors> for EdgeType {
    fn from(edge: Mirrors) -> Self {
        EdgeType::Mirrors(edge)
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
        src_node: &dyn StaticNode,
        tgt_node: &dyn StaticNode,
    ) -> Option<Self> {
        if src_index == tgt_index {
            return None;
        }

        let publisher = src_node.as_any().downcast_ref::<Publisher>()?;
        let article = tgt_node.as_any().downcast_ref::<Article>()?;

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
        src_node: &dyn StaticNode,
        tgt_node: &dyn StaticNode,
    ) -> Option<Self> {
        if src_index == tgt_index {
            return None;
        }

        let article = src_node.as_any().downcast_ref::<Article>()?;
        let company = tgt_node.as_any().downcast_ref::<Company>()?;

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
        src_node: &dyn StaticNode,
        tgt_node: &dyn StaticNode,
    ) -> Option<Self> {
        if src_index == tgt_index {
            return None;
        }

        let article = src_node.as_any().downcast_ref::<Article>()?;
        let equity = tgt_node.as_any().downcast_ref::<Equity>()?;

        if let Some(sentiment) = article.ticker_sentiment(equity.name().clone()) {
            Some(Referenced {
                src_index,
                tgt_index,
            })
        } else {
            None
        }
    }
}

impl Issues {
    pub fn try_new(
        src_index: NodeIndex,
        tgt_index: NodeIndex,
        src_node: &dyn StaticNode,
        tgt_node: &dyn StaticNode,
    ) -> Option<Self> {
        if src_index == tgt_index {
            return None;
        }

        let company = src_node.as_any().downcast_ref::<Company>()?;
        let equity = tgt_node.as_any().downcast_ref::<Equity>()?;

        if company.symbols().contains(equity.name()) {
            todo!()
        } else {
            None
        }
    }
}

impl Mirrors {
    pub fn try_new(
        src_index: NodeIndex,
        tgt_index: NodeIndex,
        src_node: &dyn StaticNode,
        tgt_node: &dyn StaticNode,
    ) -> Option<Self> {
        if src_index == tgt_index {
            return None;
        }

        let etf = src_node.as_any().downcast_ref::<ETFs>()?;
        let index = tgt_node.as_any().downcast_ref::<Indices>()?;

        if etf.indice() == index.name() {
            todo!()
        } else {
            None
        }
    }
}

impl Influences {
    pub fn try_new(
        src_index: NodeIndex,
        tgt_index: NodeIndex,
        src_node: &dyn StaticNode,
        tgt_node: &dyn StaticNode,
    ) -> Option<Self> {
        if src_index == tgt_index {
            return None;
        }

        let source = src_node.as_any().downcast_ref::<Equity>()?;
        let target = tgt_node.as_any().downcast_ref::<Equity>()?;

        todo!()
    }
}

impl Derives {
    pub fn try_new(
        src_index: NodeIndex,
        tgt_index: NodeIndex,
        src_node: &dyn StaticNode,
        tgt_node: &dyn StaticNode,
    ) -> Option<Self> {
        if src_index == tgt_index {
            return None;
        }

        let equity = src_node.as_any().downcast_ref::<Equity>()?;
        let option = tgt_node.as_any().downcast_ref::<Options>()?;

        if option.underlying() == equity.name() {
            todo!()
        } else {
            None
        }
    }
}

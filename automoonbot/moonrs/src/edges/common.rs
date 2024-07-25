use crate::edges::*;

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
}

#[derive(Debug)]
pub struct Mirrors {
    pub(super) src_index: NodeIndex,
    pub(super) tgt_index: NodeIndex,
}

#[derive(Debug)]
pub struct Derives {
    pub(super) src_index: NodeIndex,
    pub(super) tgt_index: NodeIndex,
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
            todo!();
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

        if let Some(intersection) = article.ticker_intersect(company.symbols()) {
            todo!()
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
            todo!()
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

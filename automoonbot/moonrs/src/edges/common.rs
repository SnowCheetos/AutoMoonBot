use crate::edges::*;

#[derive(Debug, Clone)]
pub enum EdgeType {
    Mentioned(Mentioned),
    Composed(Composed),
}

#[derive(Debug, Clone)]
pub struct Mentioned {
    pub(in crate::edges) src_ty: String,
    pub(in crate::edges) tgt_ty: String,
    pub(in crate::edges) src_nm: String,
    pub(in crate::edges) tgt_nm: String,
    pub(in crate::edges) relevance: f64,
    pub(in crate::edges) sentiment: f64,
}

#[derive(Debug, Clone)]
pub struct Composed {}

impl Mentioned {
    pub fn new(src_ty: &Article, tgt_ty: &StaticEvent, relevance: f64, sentiment: f64) -> Self {
        Mentioned {
            src_ty: src_ty.cls().to_owned(),
            tgt_ty: tgt_ty.cls().to_owned(),
            src_nm: src_ty.name().to_owned(),
            tgt_nm: tgt_ty.name().to_owned(),
            relevance,
            sentiment,
        }
    }
}

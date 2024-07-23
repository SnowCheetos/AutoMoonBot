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
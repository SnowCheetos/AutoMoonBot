use crate::edges::*;

pub trait StaticEdge: Send + Sync {
    fn value(&self) -> f64;
    fn src_index(&self) -> &NodeIndex;
    fn tgt_index(&self) -> &NodeIndex;
}

impl StaticEdge for Published {
    fn value(&self) -> f64 {
        0.0
    }

    fn src_index(&self) -> &NodeIndex {
        &self.src_index
    }

    fn tgt_index(&self) -> &NodeIndex {
        &self.tgt_index
    }
}

impl StaticEdge for Mentioned {
    fn value(&self) -> f64 {
        0.0
    }

    fn src_index(&self) -> &NodeIndex {
        &self.src_index
    }

    fn tgt_index(&self) -> &NodeIndex {
        &self.tgt_index
    }
}

impl StaticEdge for Referenced {
    fn value(&self) -> f64 {
        0.0
    }

    fn src_index(&self) -> &NodeIndex {
        &self.src_index
    }

    fn tgt_index(&self) -> &NodeIndex {
        &self.tgt_index
    }
}

impl StaticEdge for Issues {
    fn value(&self) -> f64 {
        0.0
    }

    fn src_index(&self) -> &NodeIndex {
        &self.src_index
    }

    fn tgt_index(&self) -> &NodeIndex {
        &self.tgt_index
    }
}

impl StaticEdge for Mirrors {
    fn value(&self) -> f64 {
        0.0
    }

    fn src_index(&self) -> &NodeIndex {
        &self.src_index
    }

    fn tgt_index(&self) -> &NodeIndex {
        &self.tgt_index
    }
}

impl StaticEdge for Derives {
    fn value(&self) -> f64 {
        0.0
    }

    fn src_index(&self) -> &NodeIndex {
        &self.src_index
    }

    fn tgt_index(&self) -> &NodeIndex {
        &self.tgt_index
    }
}
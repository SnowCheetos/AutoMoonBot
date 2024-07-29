use crate::edges::*;

pub trait StaticEdge: Send + Sync {
    fn cls(&self) -> &'static str;
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
    
    fn cls(&self) -> &'static str {
        "Published"
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
    
    fn cls(&self) -> &'static str {
        "Mentioned"
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
    
    fn cls(&self) -> &'static str {
        "Referenced"
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
    
    fn cls(&self) -> &'static str {
        "Issues"
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
    
    fn cls(&self) -> &'static str {
        "Mirrors"
    }
}

impl StaticEdge for Influences {
    fn value(&self) -> f64 {
        0.0
    }

    fn src_index(&self) -> &NodeIndex {
        &self.src_index
    }

    fn tgt_index(&self) -> &NodeIndex {
        &self.tgt_index
    }
    
    fn cls(&self) -> &'static str {
        "Influences"
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
    
    fn cls(&self) -> &'static str {
        "Derives"
    }
}
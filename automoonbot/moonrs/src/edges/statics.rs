use crate::edges::*;

pub trait StaticEdge: Send + Sync {
    fn cls(&self) -> &'static str;
    fn value(&self) -> f64;
    fn src_index(&self) -> &NodeIndex;
    fn tgt_index(&self) -> &NodeIndex;
    fn feature(&self) -> na::RowDVector<f64>;
}

impl StaticEdge for EdgeType {
    fn value(&self) -> f64 {
        0.0
    }

    fn src_index(&self) -> &NodeIndex {
        match self {
            EdgeType::Published(p) => &p.src_index,
            EdgeType::Mentioned(m) => &m.src_index,
            EdgeType::Referenced(r) => &r.src_index,
            EdgeType::Issues(i) => &i.src_index,
            EdgeType::Influences(i) => &i.src_index,
            EdgeType::Derives(d) => &d.src_index,
            EdgeType::TestEdge(t) => &t.src_index,
        }
    }

    fn tgt_index(&self) -> &NodeIndex {
        match self {
            EdgeType::Published(p) => &p.tgt_index,
            EdgeType::Mentioned(m) => &m.tgt_index,
            EdgeType::Referenced(r) => &r.tgt_index,
            EdgeType::Issues(i) => &i.tgt_index,
            EdgeType::Influences(i) => &i.tgt_index,
            EdgeType::Derives(d) => &d.tgt_index,
            EdgeType::TestEdge(t) => &t.tgt_index,
        }
    }
    
    fn cls(&self) -> &'static str {
        match self {
            EdgeType::Published(_) => "Published",
            EdgeType::Mentioned(_) => "Mentioned",
            EdgeType::Referenced(_) => "Referenced",
            EdgeType::Issues(_) => "Issues",
            EdgeType::Influences(_) => "Influences",
            EdgeType::Derives(_) => "Derives",
            EdgeType::TestEdge(_) => "TestEdge",
        }
    }

    fn feature(&self) -> na::RowDVector<f64> {
        todo!()
    }
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

    fn feature(&self) -> na::RowDVector<f64> {
        todo!()
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

    fn feature(&self) -> na::RowDVector<f64> {
        todo!()
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

    fn feature(&self) -> na::RowDVector<f64> {
        todo!()
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

    fn feature(&self) -> na::RowDVector<f64> {
        todo!()
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

    fn feature(&self) -> na::RowDVector<f64> {
        todo!()
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

    fn feature(&self) -> na::RowDVector<f64> {
        todo!()
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

    fn feature(&self) -> na::RowDVector<f64> {
        todo!()
    }
}
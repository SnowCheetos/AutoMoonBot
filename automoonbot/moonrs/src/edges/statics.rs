use crate::edges::*;

pub trait StaticEdge: Send + Sync {
    fn cls(&self) -> &'static str;
    fn value(&self) -> f64;
    fn src_index(&self) -> &NodeIndex;
    fn tgt_index(&self) -> &NodeIndex;
    fn feature(&self) -> Option<na::RowDVector<f64>>;
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
            EdgeType::Published(p) => p.cls(),
            EdgeType::Mentioned(m) => m.cls(),
            EdgeType::Referenced(r) => r.cls(),
            EdgeType::Issues(i) => i.cls(),
            EdgeType::Influences(i) => i.cls(),
            EdgeType::Derives(d) => d.cls(),
            EdgeType::TestEdge(t) => t.cls(),
        }
    }

    fn feature(&self) -> Option<na::RowDVector<f64>> {
        match self {
            EdgeType::Published(p) => p.feature(),
            EdgeType::Mentioned(m) => m.feature(),
            EdgeType::Referenced(r) => r.feature(),
            EdgeType::Issues(i) => i.feature(),
            EdgeType::Influences(i) => i.feature(),
            EdgeType::Derives(d) => d.feature(),
            EdgeType::TestEdge(t) => t.feature(),
        }
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

    fn feature(&self) -> Option<na::RowDVector<f64>> {
        Some(na::RowDVector::from_vec(vec![0.0]))
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

    fn feature(&self) -> Option<na::RowDVector<f64>> {
        Some(na::RowDVector::from_vec(vec![0.0]))
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

    fn feature(&self) -> Option<na::RowDVector<f64>> {
        Some(na::RowDVector::from_vec(vec![self.sentiment]))
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

    fn feature(&self) -> Option<na::RowDVector<f64>> {
        self.covariance.as_ref().map(|cov| cov.row(0).into_owned())
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

    fn feature(&self) -> Option<na::RowDVector<f64>> {
        self.covariance.as_ref().map(|cov| cov.row(0).into_owned())
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

    fn feature(&self) -> Option<na::RowDVector<f64>> {
        self.covariance.as_ref().map(|cov| cov.row(0).into_owned())
    }
}

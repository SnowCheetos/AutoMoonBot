use crate::edges::*;

/// The base trait that all edges must implement.
/// Static edge implies (*not exclusively*) that
/// the source and target node shares some static
/// relationship. For example, an author published
/// an article on some date.
///
/// ```math
/// S_t \leftrightarrow T_t
/// ```
pub trait StaticEdge: Clone + Send + Sync {
    type Source: StaticNode;
    type Target: StaticNode;
    /// Returns the class name of the edge, must be
    /// unique and match that of the struct name itself.
    ///
    /// # Examples
    /// ```rust
    /// impl StaticEdge for SomeEdge {
    ///     type Source: SomeNodeType;
    ///     type Target: SomeNodeType;
    ///     fn cls(&self) -> &'static str {
    ///         "SomeEdge"
    ///     }
    /// }
    /// let edge = SomeEdge::new(...);
    /// assert_eq!(edge.cls(), "SomeEdge");
    /// ```
    fn cls(&self) -> &'static str;
    fn src_type(&self) -> &String;
    fn tgt_type(&self) -> &String;
    fn src_name(&self) -> &String;
    fn tgt_name(&self) -> &String;
}

impl StaticEdge for EdgeType {
    type Source = NodeType;
    type Target = NodeType;

    fn cls(&self) -> &'static str {
        match self {
            EdgeType::Mentioned(edge) => edge.cls(),
            EdgeType::Composed(edge) => todo!(),
        }
    }

    fn src_type(&self) -> &String {
        match self {
            EdgeType::Mentioned(edge) => edge.src_type(),
            EdgeType::Composed(edge) => todo!(),
        }
    }

    fn tgt_type(&self) -> &String {
        match self {
            EdgeType::Mentioned(edge) => edge.tgt_type(),
            EdgeType::Composed(edge) => todo!(),
        }
    }

    fn src_name(&self) -> &String {
        match self {
            EdgeType::Mentioned(edge) => edge.src_name(),
            EdgeType::Composed(edge) => todo!(),
        }
    }

    fn tgt_name(&self) -> &String {
        match self {
            EdgeType::Mentioned(edge) => edge.tgt_name(),
            EdgeType::Composed(edge) => todo!(),
        }
    }
}

impl StaticEdge for Mentioned {
    type Source = Article;
    type Target = StaticEvent;

    fn cls(&self) -> &'static str {
        "Mentioned"
    }

    fn src_type(&self) -> &String {
        &self.src_ty
    }

    fn tgt_type(&self) -> &String {
        &self.tgt_ty
    }

    fn src_name(&self) -> &String {
        &self.src_nm
    }

    fn tgt_name(&self) -> &String {
        &self.tgt_nm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mentioned() {
        let now = Instant::now();
        let later = now + Duration::new(60, 0);
        let source = Article::new(
            now,
            "news".to_owned(),
            0.5,
            "publisher".to_owned(),
            None,
            None,
            None,
        );
        let target = StaticEvent::new(later, 0.5, "".to_owned());
        let edge = Mentioned::new(&source, &target, 0.5, 0.5);

        assert_eq!(edge.cls(), "Mentioned", "Got {} instead", edge.cls());
        assert_eq!(
            edge.src_type(),
            "Article",
            "Got {} instead",
            edge.src_type()
        );
        assert_eq!(
            edge.tgt_type(),
            "StaticEvent",
            "Got {} instead",
            edge.tgt_type()
        );
        assert_eq!(
            edge.src_name(),
            source.name(),
            "Got {} instead",
            edge.src_type()
        );
        assert_eq!(
            edge.tgt_name(),
            target.name(),
            "Got {} instead",
            edge.tgt_type()
        );
    }
}

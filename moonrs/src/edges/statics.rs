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
    fn src(&self) -> &String;
    fn tgt(&self) -> &String;
    fn sid(&self) -> &String;
    fn tid(&self) -> &String;
}

pub mod static_relations {
    use super::*;

    impl StaticEdge for Mentioned {
        type Source = Article;
        type Target = StaticEvent;

        fn cls(&self) -> &'static str {
            "Mentioned"
        }

        fn src(&self) -> &String {
            &self.source
        }

        fn tgt(&self) -> &String {
            &self.target
        }
        
        fn sid(&self) -> &String {
            &self.src_uuid
        }
        
        fn tid(&self) -> &String {
            &self.tgt_uuid
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mentioned() {
        let now = Instant::now();
        let later = now + Duration::new(60, 0);
        let source = Article::new(now, "news".to_owned(), 0.5, None);
        let target = StaticEvent::new(later, 0.5, "".to_owned());
        let edge = Mentioned::new(&source, &target, 0.5, 0.5);

        assert_eq!(edge.cls(), "Mentioned", "Got {} instead", edge.cls());
        assert_eq!(edge.src(), "Article", "Got {} instead", edge.src());
        assert_eq!(edge.tgt(), "StaticEvent", "Got {} instead", edge.tgt());
        assert_eq!(edge.cls(), "Mentioned", "Got {} instead", edge.cls());
        assert_eq!(edge.sid(), source.uuid(), "Got {} instead", edge.src());
        assert_eq!(edge.tid(), target.uuid(), "Got {} instead", edge.tgt());
    }
}

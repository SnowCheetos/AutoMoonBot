use crate::nodes::*;

/// The base trait that all nodes must implement.
pub trait StaticNode: Clone + Send + Sync {
    /// Returns the class name of the node, must be
    /// unique and match that of the struct name itself.
    ///
    /// # Examples
    /// ```rust
    /// impl StaticNode for SomeNode {
    ///     fn cls(&self) -> &'static str {
    ///         "SomeNode"
    ///     }
    /// }
    /// let node = SomeNode::new(...);
    /// assert_eq!(node.cls(), "SomeNode");
    /// ```
    fn cls(&self) -> &'static str;
    fn uuid(&self) -> &String;
}

pub mod static_entities {
    use super::*;

    impl StaticNode for StaticEvent {
        fn cls(&self) -> &'static str {
            "StaticEvent"
        }

        fn uuid(&self) -> &String {
            &self.uuid
        }
    }

    impl StaticNode for Article {
        fn cls(&self) -> &'static str {
            "Article"
        }

        fn uuid(&self) -> &String {
            &self.uuid
        }
    }
}

pub mod dynamic_entities {
    use super::*;

    impl StaticNode for Equity {
        fn cls(&self) -> &'static str {
            "Equity"
        }

        fn uuid(&self) -> &String {
            &self.uuid
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_equity() {
        let symbol1 = "EQU".to_owned();
        let symbol2 = "LOL".to_owned();
        let region = "MOON".to_owned();
        let equity1 = Equity::new(10, symbol1, region.clone(), Vec::new());
        let equity2 = Equity::new(10, symbol2, region, Vec::new());

        assert_eq!(equity1.cls(), "Equity");
        assert_eq!(equity2.cls(), "Equity");
        assert_ne!(equity1.uuid(), equity2.uuid());
    }
}

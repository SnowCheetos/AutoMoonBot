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

    /// Returns an unique identifer string for a specific 
    /// instance. This is to avoid using references which 
    /// eliminates the need to manage lifetimes.
    ///
    /// # Examples
    /// ```rust
    /// 
    /// struct SomeNode {
    ///     name: String,
    /// }
    /// 
    /// impl SomeNode {
    ///     fn new(name: String) -> Self {
    ///         SomeNode {
    ///             name,
    ///         }
    ///     }
    /// }
    /// 
    /// impl StaticNode for SomeNode {
    ///     fn name(&self) -> &String {
    ///         &self.name
    ///     }
    /// }
    /// let node = SomeNode::new("my_name_is_Jeff".to_owned());
    /// assert_eq!(node.name(), "my_name_is_Jeff");
    /// ```
    fn name(&self) -> &String;
}

pub mod static_entities {
    use super::*;

    impl StaticNode for StaticEvent {
        fn cls(&self) -> &'static str {
            "StaticEvent"
        }
        
        fn name(&self) -> &String {
            &self.description
        }
    }

    impl StaticNode for Article {
        fn cls(&self) -> &'static str {
            "Article"
        }
        
        fn name(&self) -> &String {
            &self.title
        }
    }
}

pub mod dynamic_entities {
    use super::*;

    impl StaticNode for Equity {
        fn cls(&self) -> &'static str {
            "Equity"
        }
        
        fn name(&self) -> &String {
            &self.symbol
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
        assert_ne!(equity1.name(), equity2.name());
    }
}

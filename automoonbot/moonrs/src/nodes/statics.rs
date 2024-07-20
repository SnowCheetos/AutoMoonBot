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
    /// eliminates the need to manage lifetimes, especially
    /// since the nodes are stored in a graph.
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

impl StaticNode for NodeType {
    fn cls(&self) -> &'static str {
        match self {
            NodeType::StaticEvent(node) => node.cls(),
            NodeType::Article(node) => node.cls(),
            NodeType::TemporalEvent(node) => todo!(),
            NodeType::Author(node) => todo!(),
            NodeType::Publisher(node) => todo!(),
            NodeType::Region(node) => todo!(),
            NodeType::Exchange(node) => todo!(),
            NodeType::Company(node) => todo!(),
            NodeType::Sector(node) => todo!(),
            NodeType::Indices(node) => todo!(),
            NodeType::Commodity(node) => todo!(),
            NodeType::Currency(node) => todo!(),
            NodeType::Bonds(node) => todo!(),
            NodeType::Equity(node) => node.cls(),
            NodeType::Crypto(node) => todo!(),
            NodeType::ETFs(node) => todo!(),
            NodeType::Futures(node) => todo!(),
            NodeType::Shorts(node) => todo!(),
            NodeType::Options(node) => todo!(),
        }
    }

    fn name(&self) -> &String {
        match self {
            NodeType::StaticEvent(node) => node.name(),
            NodeType::Article(node) => node.name(),
            NodeType::TemporalEvent(_) => todo!(),
            NodeType::Author(_) => todo!(),
            NodeType::Publisher(_) => todo!(),
            NodeType::Region(_) => todo!(),
            NodeType::Exchange(_) => todo!(),
            NodeType::Company(_) => todo!(),
            NodeType::Sector(_) => todo!(),
            NodeType::Indices(_) => todo!(),
            NodeType::Commodity(_) => todo!(),
            NodeType::Currency(_) => todo!(),
            NodeType::Bonds(_) => todo!(),
            NodeType::Equity(_) => todo!(),
            NodeType::Crypto(_) => todo!(),
            NodeType::ETFs(_) => todo!(),
            NodeType::Futures(_) => todo!(),
            NodeType::Shorts(_) => todo!(),
            NodeType::Options(_) => todo!(),
        }
    }
}

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

impl StaticNode for Equity {
    fn cls(&self) -> &'static str {
        "Equity"
    }

    fn name(&self) -> &String {
        &self.symbol
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

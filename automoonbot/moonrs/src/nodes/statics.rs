use crate::nodes::*;

pub trait StaticNode: Send + Sync {
    fn cls(&self) -> &'static str;
    fn name(&self) -> &String;
    fn value(&self) -> f64;
}

impl StaticNode for TestNode {
    fn cls(&self) -> &'static str {
        "TestNode"
    }

    fn name(&self) -> &String {
        &self.name
    }
    
    fn value(&self) -> f64 {
        self.value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_node() {
        let name1 = "Node1".to_owned();
        let name2 = "Node2".to_owned();
        let node1 = TestNode::new(name1.clone(), 0.0);
        let node2 = TestNode::new(name2.clone(), 0.0);

        assert_eq!(node1.cls(), "TestNode");
        assert_eq!(node2.cls(), "TestNode");
        assert_eq!(node1.cls(), node2.cls());
        assert_ne!(node1.name(), node2.name());
    }
}

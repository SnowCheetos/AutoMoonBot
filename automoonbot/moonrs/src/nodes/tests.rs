use crate::nodes::*;

#[derive(Debug)]
pub struct TestNode {
    pub(super) name: String,
    pub(super) value: f64,
    pub(super) buffer: TimeSeries<f64>,
}

impl TestNode {
    pub fn new(name: String, value: f64, capacity: usize) -> Self {
        TestNode {
            name,
            value,
            buffer: TimeSeries::new(capacity),
        }
    }

    pub fn value(&self) -> f64 {
        self.value
    }
}

impl StaticNode for TestNode {
    fn cls(&self) -> &'static str {
        "TestNode"
    }

    fn name(&self) -> &String {
        &self.name
    }
    
    fn value(&self) -> Option<f64> {
        Some(self.value)
    }
}

impl DynamicNode<Instant, f64> for TestNode {
    fn update(&mut self, index: Instant, item: f64) -> bool {
        self.buffer.push(index, item)
    }

    fn empty(&self) -> bool {
        self.buffer.empty()
    }

    fn to_vec(&self) -> Vec<&f64> {
        self.buffer.to_vec()
    }

    fn first(&self) -> Option<&f64> {
        self.buffer.first()
    }

    fn last(&self) -> Option<&f64> {
        self.buffer.last()
    }

    fn between(&self, start: Instant, end: Instant) -> Option<Vec<&f64>> {
        self.buffer.between(&start, &end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_static_node() {
        let name1 = "Node1".to_owned();
        let name2 = "Node2".to_owned();
        let node1 = TestNode::new(name1.clone(), 0.0, 1);
        let node2 = TestNode::new(name2.clone(), 0.0, 1);

        assert_eq!(node1.cls(), "TestNode");
        assert_eq!(node2.cls(), "TestNode");
        assert_eq!(node1.cls(), node2.cls());
        assert_ne!(node1.name(), node2.name());
    }

    #[test]
    fn test_dynamic_node() {
        let name1 = "Node1".to_owned();
        let name2 = "Node2".to_owned();
        let mut node1 = TestNode::new(name1.clone(), 0.0, 5);
        let mut node2 = TestNode::new(name2.clone(), 0.0, 5);
        assert!(node1.empty() && node2.empty());

        let now = Instant::now();
        let value1 = 1.0;
        let value2 = 2.0;
        let success1 = node1.update(now, value1);
        let success2 = node2.update(now, value2);
        assert!(success1 && success2);

        let success = node1.update(now, value1);
        assert!(!success);

        assert!(node1.first().is_some_and(|value| *value == value1));
        assert!(node2.first().is_some_and(|value| *value == value2));

        let later = Instant::now();
        node1.update(later, value2);
        node2.update(later, value1);

        assert!(node1.last().is_some_and(|value| *value == value2));
        assert!(node2.last().is_some_and(|value| *value == value1));

        let vec1 = node1.between(now, later);
        let vec2 = node2.between(now, later);

        assert!(vec1.is_some_and(|v| v == vec![&value1, &value2]));
        assert!(vec2.is_some_and(|v| v == vec![&value2, &value1]));
    }
}
use std::cell::RefCell;
use std::rc::Rc;

/// Shared, mutable tensor storage backed by a flat Vec<f32>.
/// Multiple tensors (views) can share the same Storage via Rc<RefCell<...>>.
#[derive(Debug, Clone)]
pub struct Storage {
    pub data: Vec<f32>,
}

impl Storage {
    pub fn new(data: Vec<f32>) -> Rc<RefCell<Self>> {
        Rc::new(RefCell::new(Storage { data }))
    }

    pub fn zeros(size: usize) -> Rc<RefCell<Self>> {
        Self::new(vec![0.0; size])
    }

    pub fn ones(size: usize) -> Rc<RefCell<Self>> {
        Self::new(vec![1.0; size])
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use crate::storage::Storage;

/// A multi-dimensional tensor with f32 storage.
///
/// Uses a flat Vec<f32> for storage with shape and stride metadata.
/// Element at index (i0, i1, ..., in) is at:
///   storage[storage_offset + i0*stride[0] + i1*stride[1] + ... + in*stride[n]]
///
/// Views (transpose, reshape) share storage and just change strides/offset.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub(crate) storage: Rc<RefCell<Storage>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) storage_offset: usize,
}

impl Tensor {
    // ---- Constructors ----

    /// Create a tensor from raw data and shape.
    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "Data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            expected
        );
        let strides = Self::compute_contiguous_strides(&shape);
        Tensor {
            storage: Storage::new(data),
            shape,
            strides,
            storage_offset: 0,
        }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let strides = Self::compute_contiguous_strides(&shape);
        Tensor {
            storage: Storage::zeros(size),
            shape,
            strides,
            storage_offset: 0,
        }
    }

    /// Create a tensor filled with ones.
    pub fn ones(shape: Vec<usize>) -> Self {
        let size: usize = shape.iter().product();
        let strides = Self::compute_contiguous_strides(&shape);
        Tensor {
            storage: Storage::ones(size),
            shape,
            strides,
            storage_offset: 0,
        }
    }

    /// Create a tensor filled with a specific value.
    pub fn full(shape: Vec<usize>, value: f32) -> Self {
        let size: usize = shape.iter().product();
        let strides = Self::compute_contiguous_strides(&shape);
        Tensor {
            storage: Storage::new(vec![value; size]),
            shape,
            strides,
            storage_offset: 0,
        }
    }

    /// Create a tensor with random values in [0, 1).
    pub fn rand(shape: Vec<usize>) -> Self {
        use rand::Rng;
        let size: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..size).map(|_| rng.gen::<f32>()).collect();
        let strides = Self::compute_contiguous_strides(&shape);
        Tensor {
            storage: Storage::new(data),
            shape,
            strides,
            storage_offset: 0,
        }
    }

    /// Create a tensor with random values from a normal distribution N(0, 1).
    pub fn randn(shape: Vec<usize>) -> Self {
        use rand::Rng;
        let size: usize = shape.iter().product();
        let mut rng = rand::thread_rng();
        // Box-Muller transform
        let data: Vec<f32> = (0..size)
            .map(|_| {
                let u1: f64 = rng.gen::<f64>().max(1e-10);
                let u2: f64 = rng.gen::<f64>();
                ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
            })
            .collect();
        let strides = Self::compute_contiguous_strides(&shape);
        Tensor {
            storage: Storage::new(data),
            shape,
            strides,
            storage_offset: 0,
        }
    }

    /// Create a scalar tensor.
    pub fn scalar(value: f32) -> Self {
        Tensor {
            storage: Storage::new(vec![value]),
            shape: vec![],
            strides: vec![],
            storage_offset: 0,
        }
    }

    /// Create a 1-D tensor with values from start to end (exclusive), step 1.
    pub fn arange(start: f32, end: f32, step: f32) -> Self {
        let mut data = Vec::new();
        let mut val = start;
        if step > 0.0 {
            while val < end {
                data.push(val);
                val += step;
            }
        } else if step < 0.0 {
            while val > end {
                data.push(val);
                val += step;
            }
        }
        let len = data.len();
        Self::from_data(data, vec![len])
    }

    // ---- Properties ----

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn storage_offset(&self) -> usize {
        self.storage_offset
    }

    /// Whether this tensor is contiguous in memory (C-order).
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }
        let expected = Self::compute_contiguous_strides(&self.shape);
        self.strides == expected
    }

    // ---- Data Access ----

    /// Get the flat index into storage for given multi-dimensional indices.
    pub fn flat_index(&self, indices: &[usize]) -> usize {
        assert_eq!(indices.len(), self.ndim(), "Wrong number of indices");
        let mut idx = self.storage_offset;
        for (i, &dim_idx) in indices.iter().enumerate() {
            assert!(dim_idx < self.shape[i], "Index out of bounds");
            idx += dim_idx * self.strides[i];
        }
        idx
    }

    /// Get an element by multi-dimensional index.
    pub fn get(&self, indices: &[usize]) -> f32 {
        let idx = self.flat_index(indices);
        self.storage.borrow().data[idx]
    }

    /// Set an element by multi-dimensional index.
    pub fn set(&self, indices: &[usize], value: f32) {
        let idx = self.flat_index(indices);
        self.storage.borrow_mut().data[idx] = value;
    }

    /// Get scalar value (for 0-dim tensors).
    pub fn item(&self) -> f32 {
        assert!(self.numel() == 1, "item() requires a single-element tensor");
        self.storage.borrow().data[self.storage_offset]
    }

    /// Return a contiguous copy of data in row-major order.
    pub fn to_vec(&self) -> Vec<f32> {
        let n = self.numel();
        if n == 0 {
            return vec![];
        }
        let mut result = Vec::with_capacity(n);
        let storage = self.storage.borrow();
        self.collect_data(&storage.data, &self.shape, &self.strides, self.storage_offset, &mut result);
        result
    }

    fn collect_data(
        &self,
        data: &[f32],
        shape: &[usize],
        strides: &[usize],
        offset: usize,
        result: &mut Vec<f32>,
    ) {
        if shape.is_empty() {
            result.push(data[offset]);
            return;
        }
        if shape.len() == 1 {
            for i in 0..shape[0] {
                result.push(data[offset + i * strides[0]]);
            }
            return;
        }
        for i in 0..shape[0] {
            self.collect_data(data, &shape[1..], &strides[1..], offset + i * strides[0], result);
        }
    }

    /// Return a new contiguous tensor with the same data.
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return self.clone();
        }
        Tensor::from_data(self.to_vec(), self.shape.clone())
    }

    // ---- Views (share storage) ----

    /// Reshape the tensor (must be contiguous). Returns a view sharing storage.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let mut resolved = new_shape.clone();
        let mut infer_idx = None;
        let mut known_product: usize = 1;

        for (i, &dim) in resolved.iter().enumerate() {
            if dim == usize::MAX {
                // Use usize::MAX as -1 sentinel
                assert!(infer_idx.is_none(), "Can only infer one dimension");
                infer_idx = Some(i);
            } else {
                known_product *= dim;
            }
        }

        if let Some(idx) = infer_idx {
            assert!(
                self.numel() % known_product == 0,
                "Cannot reshape tensor of size {} into {:?}",
                self.numel(),
                new_shape
            );
            resolved[idx] = self.numel() / known_product;
        }

        let new_numel: usize = resolved.iter().product();
        assert_eq!(
            new_numel,
            self.numel(),
            "Cannot reshape tensor of size {} into {:?}",
            self.numel(),
            resolved
        );

        // Must be contiguous for simple reshape view
        if self.is_contiguous() {
            let strides = Self::compute_contiguous_strides(&resolved);
            Tensor {
                storage: Rc::clone(&self.storage),
                shape: resolved,
                strides,
                storage_offset: self.storage_offset,
            }
        } else {
            // Make contiguous copy, then reshape
            let t = self.contiguous();
            let strides = Self::compute_contiguous_strides(&resolved);
            Tensor {
                storage: t.storage,
                shape: resolved,
                strides,
                storage_offset: 0,
            }
        }
    }

    /// Transpose two dimensions. Returns a view sharing storage.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        assert!(dim0 < self.ndim() && dim1 < self.ndim(), "Dimension out of range");
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);
        Tensor {
            storage: Rc::clone(&self.storage),
            shape: new_shape,
            strides: new_strides,
            storage_offset: self.storage_offset,
        }
    }

    /// Shorthand for transpose(0, 1) on a 2D tensor. Commonly `.t()`.
    pub fn t(&self) -> Tensor {
        assert!(self.ndim() == 2, "t() requires a 2D tensor");
        self.transpose(0, 1)
    }

    /// Unsqueeze: add a dimension of size 1 at the given position.
    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        assert!(dim <= self.ndim(), "Dimension out of range for unsqueeze");
        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();
        // The stride for a size-1 dim doesn't matter (could be anything), use 1
        let stride_val = if dim < self.ndim() { self.strides[dim] } else { 1 };
        new_shape.insert(dim, 1);
        new_strides.insert(dim, stride_val);
        Tensor {
            storage: Rc::clone(&self.storage),
            shape: new_shape,
            strides: new_strides,
            storage_offset: self.storage_offset,
        }
    }

    /// Squeeze: remove all dimensions of size 1 (or a specific dimension).
    pub fn squeeze(&self, dim: Option<usize>) -> Tensor {
        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();
        for (i, (&s, &st)) in self.shape.iter().zip(self.strides.iter()).enumerate() {
            let should_squeeze = match dim {
                Some(d) => i == d && s == 1,
                None => s == 1,
            };
            if !should_squeeze {
                new_shape.push(s);
                new_strides.push(st);
            }
        }
        Tensor {
            storage: Rc::clone(&self.storage),
            shape: new_shape,
            strides: new_strides,
            storage_offset: self.storage_offset,
        }
    }

    /// Expand tensor to a larger size (broadcasting). Expanded dims must be size 1.
    pub fn expand(&self, new_shape: &[usize]) -> Tensor {
        assert!(
            new_shape.len() >= self.ndim(),
            "Expanded shape must have at least as many dimensions"
        );
        let offset = new_shape.len() - self.ndim();
        let mut result_strides = vec![0usize; new_shape.len()];

        for i in 0..new_shape.len() {
            if i < offset {
                result_strides[i] = 0;
            } else {
                let orig = i - offset;
                if self.shape[orig] == new_shape[i] {
                    result_strides[i] = self.strides[orig];
                } else if self.shape[orig] == 1 {
                    result_strides[i] = 0;
                } else {
                    panic!(
                        "Cannot expand dim {} from {} to {}",
                        orig, self.shape[orig], new_shape[i]
                    );
                }
            }
        }

        Tensor {
            storage: Rc::clone(&self.storage),
            shape: new_shape.to_vec(),
            strides: result_strides,
            storage_offset: self.storage_offset,
        }
    }

    // ---- In-place ops ----

    /// In-place addition: self += other (element-wise with broadcasting).
    pub fn add_(&self, other: &Tensor) {
        let data = self.to_vec();
        let other_data = other.to_vec();
        // Simple case: same numel
        assert_eq!(data.len(), other_data.len(), "add_ requires same size tensors (use broadcast first)");
        let mut storage = self.storage.borrow_mut();
        for (i, &v) in other_data.iter().enumerate() {
            storage.data[self.storage_offset + i] += v;
        }
    }

    /// In-place scalar multiplication: self *= scalar.
    pub fn mul_scalar_(&self, scalar: f32) {
        if self.is_contiguous() {
            let mut storage = self.storage.borrow_mut();
            let start = self.storage_offset;
            let end = start + self.numel();
            for v in &mut storage.data[start..end] {
                *v *= scalar;
            }
        } else {
            let n = self.numel();
            let indices = self.all_indices();
            let mut storage = self.storage.borrow_mut();
            for idx in indices.iter().take(n) {
                storage.data[*idx] *= scalar;
            }
        }
    }

    /// In-place fill with scalar.
    pub fn fill_(&self, value: f32) {
        if self.is_contiguous() {
            let mut storage = self.storage.borrow_mut();
            let start = self.storage_offset;
            let end = start + self.numel();
            for v in &mut storage.data[start..end] {
                *v = value;
            }
        } else {
            let indices = self.all_indices();
            let mut storage = self.storage.borrow_mut();
            for idx in &indices {
                storage.data[*idx] = value;
            }
        }
    }

    /// In-place subtraction of scaled tensor: self -= alpha * other
    pub fn sub_alpha_(&self, other: &Tensor, alpha: f32) {
        let self_data = self.to_vec();
        let other_data = other.to_vec();
        assert_eq!(self_data.len(), other_data.len());
        let mut storage = self.storage.borrow_mut();
        for (i, &v) in other_data.iter().enumerate() {
            storage.data[self.storage_offset + i] -= alpha * v;
        }
    }

    /// Copy data from another tensor into self.
    pub fn copy_from(&self, src: &Tensor) {
        let data = src.to_vec();
        assert_eq!(self.numel(), data.len());
        let mut storage = self.storage.borrow_mut();
        if self.is_contiguous() {
            let start = self.storage_offset;
            storage.data[start..start + data.len()].copy_from_slice(&data);
        } else {
            let indices = self.all_indices();
            for (i, &idx) in indices.iter().enumerate() {
                storage.data[idx] = data[i];
            }
        }
    }

    // ---- Helpers ----

    /// Compute contiguous (row-major) strides from shape.
    pub fn compute_contiguous_strides(shape: &[usize]) -> Vec<usize> {
        if shape.is_empty() {
            return vec![];
        }
        let mut strides = vec![0usize; shape.len()];
        strides[shape.len() - 1] = 1;
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    /// Get all flat storage indices in row-major order.
    fn all_indices(&self) -> Vec<usize> {
        let n = self.numel();
        let mut indices = Vec::with_capacity(n);
        if n == 0 {
            return indices;
        }
        let ndim = self.ndim();
        if ndim == 0 {
            indices.push(self.storage_offset);
            return indices;
        }
        let mut coord = vec![0usize; ndim];
        for _ in 0..n {
            let mut idx = self.storage_offset;
            for d in 0..ndim {
                idx += coord[d] * self.strides[d];
            }
            indices.push(idx);
            // increment coord
            for d in (0..ndim).rev() {
                coord[d] += 1;
                if coord[d] < self.shape[d] {
                    break;
                }
                coord[d] = 0;
            }
        }
        indices
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let data = self.to_vec();
        write!(f, "Tensor(shape={:?}, data={:?})", self.shape, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_data() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 6);
        assert!(t.is_contiguous());
    }

    #[test]
    fn test_get_set() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.get(&[0, 0]), 1.0);
        assert_eq!(t.get(&[0, 1]), 2.0);
        assert_eq!(t.get(&[1, 0]), 3.0);
        assert_eq!(t.get(&[1, 1]), 4.0);

        t.set(&[1, 1], 42.0);
        assert_eq!(t.get(&[1, 1]), 42.0);
    }

    #[test]
    fn test_zeros_ones() {
        let z = Tensor::zeros(vec![3, 4]);
        assert_eq!(z.numel(), 12);
        assert_eq!(z.get(&[0, 0]), 0.0);

        let o = Tensor::ones(vec![2, 2]);
        assert_eq!(o.get(&[1, 1]), 1.0);
    }

    #[test]
    fn test_transpose() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tt = t.t();
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.get(&[0, 0]), 1.0);
        assert_eq!(tt.get(&[0, 1]), 4.0);
        assert_eq!(tt.get(&[1, 0]), 2.0);
        assert_eq!(tt.get(&[2, 1]), 6.0);
        assert!(!tt.is_contiguous());
    }

    #[test]
    fn test_reshape() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = t.reshape(vec![3, 2]);
        assert_eq!(r.shape(), &[3, 2]);
        assert_eq!(r.get(&[0, 0]), 1.0);
        assert_eq!(r.get(&[2, 1]), 6.0);
    }

    #[test]
    fn test_to_vec() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let tt = t.t();
        let data = tt.to_vec();
        assert_eq!(data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_scalar() {
        let s = Tensor::scalar(3.14);
        assert_eq!(s.ndim(), 0);
        assert_eq!(s.numel(), 1);
        assert_eq!(s.item(), 3.14);
    }

    #[test]
    fn test_unsqueeze_squeeze() {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let u = t.unsqueeze(0);
        assert_eq!(u.shape(), &[1, 3]);
        let s = u.squeeze(None);
        assert_eq!(s.shape(), &[3]);
    }

    #[test]
    fn test_arange() {
        let t = Tensor::arange(0.0, 5.0, 1.0);
        assert_eq!(t.shape(), &[5]);
        assert_eq!(t.to_vec(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    }
}

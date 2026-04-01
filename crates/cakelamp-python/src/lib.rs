use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use cakelamp_core::tensor::Tensor as RustTensor;
use cakelamp_core::ops;

/// Python wrapper around the Rust Tensor.
/// Marked unsendable because Rust Tensor uses Rc<RefCell<>> internally.
#[pyclass(name = "Tensor", unsendable)]
#[derive(Clone)]
pub struct PyTensor {
    pub(crate) inner: RustTensor,
}

impl PyTensor {
    pub fn new(inner: RustTensor) -> Self {
        PyTensor { inner }
    }
}

#[pymethods]
impl PyTensor {
    /// Create a tensor from a flat list of floats and a shape.
    #[new]
    #[pyo3(signature = (data, shape=None))]
    fn py_new(data: Vec<f32>, shape: Option<Vec<usize>>) -> PyResult<Self> {
        let shape = shape.unwrap_or_else(|| vec![data.len()]);
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(PyValueError::new_err(format!(
                "data length {} does not match shape {:?} (expected {})",
                data.len(), shape, expected
            )));
        }
        Ok(PyTensor::new(RustTensor::from_data(data, shape)))
    }

    /// Get the shape as a tuple.
    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    /// Get strides as a tuple.
    #[getter]
    fn strides(&self) -> Vec<usize> {
        self.inner.strides().to_vec()
    }

    /// Number of dimensions.
    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    /// Number of elements.
    fn numel(&self) -> usize {
        self.inner.numel()
    }

    /// Get scalar value from a single-element tensor.
    fn item(&self) -> PyResult<f32> {
        if self.inner.numel() != 1 {
            return Err(PyValueError::new_err("item() requires a single-element tensor"));
        }
        Ok(self.inner.item())
    }

    /// Return the data as a flat list.
    fn tolist(&self) -> Vec<f32> {
        self.inner.to_vec()
    }

    /// Return a contiguous copy.
    fn contiguous(&self) -> Self {
        PyTensor::new(self.inner.contiguous())
    }

    /// Whether the tensor is contiguous in memory.
    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    // ---- View operations ----

    /// Reshape the tensor.
    fn reshape(&self, shape: Vec<usize>) -> Self {
        PyTensor::new(self.inner.reshape(shape))
    }

    /// Transpose two dimensions.
    #[pyo3(signature = (dim0=0, dim1=1))]
    fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        PyTensor::new(self.inner.transpose(dim0, dim1))
    }

    /// Transpose a 2-D tensor (shorthand).
    fn t(&self) -> PyResult<Self> {
        if self.inner.ndim() != 2 {
            return Err(PyValueError::new_err("t() requires a 2-D tensor"));
        }
        Ok(PyTensor::new(self.inner.t()))
    }

    /// Insert a dimension of size 1.
    fn unsqueeze(&self, dim: usize) -> Self {
        PyTensor::new(self.inner.unsqueeze(dim))
    }

    /// Remove dimensions of size 1.
    #[pyo3(signature = (dim=None))]
    fn squeeze(&self, dim: Option<usize>) -> Self {
        PyTensor::new(self.inner.squeeze(dim))
    }

    /// Expand to a larger size (broadcasting).
    fn expand(&self, shape: Vec<usize>) -> Self {
        PyTensor::new(self.inner.expand(&shape))
    }

    /// Get element by index.
    fn get(&self, indices: Vec<usize>) -> PyResult<f32> {
        Ok(self.inner.get(&indices))
    }

    // ---- Unary operations ----

    fn neg(&self) -> Self {
        PyTensor::new(ops::neg(&self.inner))
    }

    fn exp(&self) -> Self {
        PyTensor::new(ops::exp(&self.inner))
    }

    fn log(&self) -> Self {
        PyTensor::new(ops::log(&self.inner))
    }

    fn relu(&self) -> Self {
        PyTensor::new(ops::relu(&self.inner))
    }

    fn sigmoid(&self) -> Self {
        PyTensor::new(ops::sigmoid(&self.inner))
    }

    fn tanh(&self) -> Self {
        PyTensor::new(ops::tanh(&self.inner))
    }

    fn abs(&self) -> Self {
        PyTensor::new(ops::abs(&self.inner))
    }

    fn sqrt(&self) -> Self {
        PyTensor::new(ops::sqrt(&self.inner))
    }

    fn clamp(&self, min: f32, max: f32) -> Self {
        PyTensor::new(ops::clamp(&self.inner, min, max))
    }

    // ---- Reduction operations ----

    /// Sum all elements.
    #[pyo3(name = "sum")]
    #[pyo3(signature = (dim=None, keepdim=false))]
    fn py_sum(&self, dim: Option<usize>, keepdim: bool) -> Self {
        match dim {
            Some(d) => PyTensor::new(ops::sum_dim(&self.inner, d, keepdim)),
            None => PyTensor::new(ops::sum(&self.inner)),
        }
    }

    /// Mean of all elements.
    #[pyo3(signature = (dim=None, keepdim=false))]
    fn mean(&self, dim: Option<usize>, keepdim: bool) -> Self {
        match dim {
            Some(d) => PyTensor::new(ops::mean_dim(&self.inner, d, keepdim)),
            None => PyTensor::new(ops::mean(&self.inner)),
        }
    }

    /// Max of all elements.
    #[pyo3(name = "max")]
    #[pyo3(signature = (dim=None, keepdim=false))]
    fn py_max(&self, dim: Option<usize>, keepdim: bool) -> Self {
        match dim {
            Some(d) => {
                PyTensor::new(reduce_max_dim(&self.inner, d, keepdim))
            }
            None => PyTensor::new(ops::max(&self.inner)),
        }
    }

    /// Argmax along a dimension.
    fn argmax(&self, dim: usize) -> Self {
        PyTensor::new(ops::argmax(&self.inner, dim))
    }

    // ---- Matrix operations ----

    /// Matrix multiply.
    fn matmul(&self, other: &PyTensor) -> Self {
        PyTensor::new(ops::matmul(&self.inner, &other.inner))
    }

    /// Softmax along a dimension.
    fn softmax(&self, dim: usize) -> Self {
        PyTensor::new(ops::softmax(&self.inner, dim))
    }

    /// Log-softmax along a dimension.
    fn log_softmax(&self, dim: usize) -> Self {
        PyTensor::new(ops::log_softmax(&self.inner, dim))
    }

    // ---- Operator overloading ----

    fn __add__(&self, other: &PyTensor) -> Self {
        PyTensor::new(ops::add(&self.inner, &other.inner))
    }

    fn __radd__(&self, other: &PyTensor) -> Self {
        PyTensor::new(ops::add(&other.inner, &self.inner))
    }

    fn __sub__(&self, other: &PyTensor) -> Self {
        PyTensor::new(ops::sub(&self.inner, &other.inner))
    }

    fn __rsub__(&self, other: &PyTensor) -> Self {
        PyTensor::new(ops::sub(&other.inner, &self.inner))
    }

    fn __mul__(&self, other: &PyTensor) -> Self {
        PyTensor::new(ops::mul(&self.inner, &other.inner))
    }

    fn __rmul__(&self, other: &PyTensor) -> Self {
        PyTensor::new(ops::mul(&other.inner, &self.inner))
    }

    fn __truediv__(&self, other: &PyTensor) -> Self {
        PyTensor::new(ops::div(&self.inner, &other.inner))
    }

    fn __neg__(&self) -> Self {
        PyTensor::new(ops::neg(&self.inner))
    }

    fn __matmul__(&self, other: &PyTensor) -> Self {
        PyTensor::new(ops::matmul(&self.inner, &other.inner))
    }

    fn __repr__(&self) -> String {
        let data = self.inner.to_vec();
        let shape = self.inner.shape();
        if data.len() <= 20 {
            format!("Tensor(data={:?}, shape={:?})", data, shape)
        } else {
            format!(
                "Tensor(data=[{}, {}, ... {} more], shape={:?})",
                data[0], data[1],
                data.len() - 2,
                shape
            )
        }
    }

    fn __str__(&self) -> String {
        self.__repr__()
    }

    fn __len__(&self) -> usize {
        if self.inner.ndim() == 0 {
            0
        } else {
            self.inner.shape()[0]
        }
    }

    // ---- In-place operations (for optimizer use) ----

    /// In-place addition.
    fn add_(&mut self, other: &PyTensor) {
        self.inner.add_(&other.inner);
    }

    /// In-place scalar multiplication.
    fn mul_scalar_(&mut self, scalar: f32) {
        self.inner.mul_scalar_(scalar);
    }

    /// In-place fill with a value.
    fn fill_(&mut self, value: f32) {
        self.inner.fill_(value);
    }

    /// In-place sub with alpha: self -= alpha * other
    fn sub_alpha_(&mut self, other: &PyTensor, alpha: f32) {
        self.inner.sub_alpha_(&other.inner, alpha);
    }

    /// Copy data from another tensor.
    fn copy_from(&mut self, src: &PyTensor) {
        self.inner.copy_from(&src.inner);
    }

    /// Return a deep copy.
    fn clone(&self) -> Self {
        // Force contiguous copy to break storage sharing
        let data = self.inner.to_vec();
        let shape = self.inner.shape().to_vec();
        PyTensor::new(RustTensor::from_data(data, shape))
    }
}

/// Helper for max_dim reduction
fn reduce_max_dim(t: &RustTensor, dim: usize, keepdim: bool) -> RustTensor {
    // Use the softmax's internal max-subtraction approach
    let shape = t.shape();
    let ndim = t.ndim();
    let dim_size = shape[dim];

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;
    let out_numel: usize = out_shape.iter().product();
    let mut out_data = vec![f32::NEG_INFINITY; out_numel];
    let out_strides = RustTensor::compute_contiguous_strides(&out_shape);

    let mut indices = vec![0usize; ndim];
    let total = t.numel();
    for _ in 0..total {
        let val = t.get(&indices);
        let mut out_flat = 0;
        for d in 0..ndim {
            let idx = if d == dim { 0 } else { indices[d] };
            out_flat += idx * out_strides[d];
        }
        if val > out_data[out_flat] {
            out_data[out_flat] = val;
        }
        // increment indices
        for d in (0..ndim).rev() {
            indices[d] += 1;
            if indices[d] < shape[d] {
                break;
            }
            indices[d] = 0;
        }
    }

    if keepdim {
        RustTensor::from_data(out_data, out_shape)
    } else {
        let squeezed: Vec<usize> = shape.iter().enumerate()
            .filter(|&(i, _)| i != dim)
            .map(|(_, &s)| s)
            .collect();
        if squeezed.is_empty() {
            RustTensor::scalar(out_data[0])
        } else {
            RustTensor::from_data(out_data, squeezed)
        }
    }
}

// ---- Module-level functions ----

/// Create a tensor filled with zeros.
#[pyfunction]
fn zeros(shape: Vec<usize>) -> PyTensor {
    PyTensor::new(RustTensor::zeros(shape))
}

/// Create a tensor filled with ones.
#[pyfunction]
fn ones(shape: Vec<usize>) -> PyTensor {
    PyTensor::new(RustTensor::ones(shape))
}

/// Create a tensor filled with a value.
#[pyfunction]
fn full(shape: Vec<usize>, value: f32) -> PyTensor {
    PyTensor::new(RustTensor::full(shape, value))
}

/// Create a tensor with uniform random values in [0, 1).
#[pyfunction]
fn rand(shape: Vec<usize>) -> PyTensor {
    PyTensor::new(RustTensor::rand(shape))
}

/// Create a tensor with normal random values (mean=0, std=1).
#[pyfunction]
fn randn(shape: Vec<usize>) -> PyTensor {
    PyTensor::new(RustTensor::randn(shape))
}

/// Create a scalar tensor.
#[pyfunction]
fn scalar(value: f32) -> PyTensor {
    PyTensor::new(RustTensor::scalar(value))
}

/// Create a 1-D tensor with evenly spaced values.
#[pyfunction]
#[pyo3(signature = (start, end, step=1.0))]
fn arange(start: f32, end: f32, step: f32) -> PyTensor {
    PyTensor::new(RustTensor::arange(start, end, step))
}

/// Create a tensor from a flat list and shape.
#[pyfunction]
fn tensor(data: Vec<f32>, shape: Vec<usize>) -> PyResult<PyTensor> {
    let expected: usize = shape.iter().product();
    if data.len() != expected {
        return Err(PyValueError::new_err(format!(
            "data length {} does not match shape {:?}",
            data.len(), shape
        )));
    }
    Ok(PyTensor::new(RustTensor::from_data(data, shape)))
}

/// Matrix multiply.
#[pyfunction]
fn matmul(a: &PyTensor, b: &PyTensor) -> PyTensor {
    PyTensor::new(ops::matmul(&a.inner, &b.inner))
}

/// ReLU activation.
#[pyfunction]
fn relu(a: &PyTensor) -> PyTensor {
    PyTensor::new(ops::relu(&a.inner))
}

/// Sigmoid activation.
#[pyfunction]
fn sigmoid(a: &PyTensor) -> PyTensor {
    PyTensor::new(ops::sigmoid(&a.inner))
}

/// Softmax along a dimension.
#[pyfunction]
fn softmax(a: &PyTensor, dim: usize) -> PyTensor {
    PyTensor::new(ops::softmax(&a.inner, dim))
}

/// Log-softmax along a dimension.
#[pyfunction]
fn log_softmax(a: &PyTensor, dim: usize) -> PyTensor {
    PyTensor::new(ops::log_softmax(&a.inner, dim))
}

/// One-hot encoding.
#[pyfunction]
fn one_hot(indices: &PyTensor, num_classes: usize) -> PyTensor {
    PyTensor::new(ops::one_hot(&indices.inner, num_classes))
}

/// The main Python module.
#[pymodule]
fn cakelamp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(rand, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(scalar, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(log_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(one_hot, m)?)?;
    Ok(())
}

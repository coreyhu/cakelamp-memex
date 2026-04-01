use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use crate::tensor::Tensor as RustTensor;
use crate::ops;

/// Python-visible Tensor class wrapping the Rust Tensor.
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
    /// Create a tensor from a flat list and shape.
    #[new]
    #[pyo3(signature = (data, shape))]
    fn py_new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(PyValueError::new_err(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(), shape, expected
            )));
        }
        Ok(PyTensor::new(RustTensor::from_data(data, shape)))
    }

    /// Create a zeros tensor.
    #[staticmethod]
    fn zeros(shape: Vec<usize>) -> Self {
        PyTensor::new(RustTensor::zeros(shape))
    }

    /// Create a ones tensor.
    #[staticmethod]
    fn ones(shape: Vec<usize>) -> Self {
        PyTensor::new(RustTensor::ones(shape))
    }

    /// Create a tensor filled with a value.
    #[staticmethod]
    fn full(shape: Vec<usize>, value: f32) -> Self {
        PyTensor::new(RustTensor::full(shape, value))
    }

    /// Create a tensor with random values in [0, 1).
    #[staticmethod]
    fn rand(shape: Vec<usize>) -> Self {
        PyTensor::new(RustTensor::rand(shape))
    }

    /// Create a tensor with random normal values.
    #[staticmethod]
    fn randn(shape: Vec<usize>) -> Self {
        PyTensor::new(RustTensor::randn(shape))
    }

    /// Create a scalar tensor.
    #[staticmethod]
    fn scalar(value: f32) -> Self {
        PyTensor::new(RustTensor::scalar(value))
    }

    /// Create an arange tensor.
    #[staticmethod]
    #[pyo3(signature = (start, end, step=1.0))]
    fn arange(start: f32, end: f32, step: f32) -> Self {
        PyTensor::new(RustTensor::arange(start, end, step))
    }

    // ---- Properties ----

    #[getter]
    fn shape(&self) -> Vec<usize> {
        self.inner.shape().to_vec()
    }

    #[getter]
    fn strides(&self) -> Vec<usize> {
        self.inner.strides().to_vec()
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.ndim()
    }

    fn numel(&self) -> usize {
        self.inner.numel()
    }

    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
    }

    fn item(&self) -> PyResult<f32> {
        if self.inner.numel() != 1 {
            return Err(PyValueError::new_err("item() requires a single-element tensor"));
        }
        Ok(self.inner.item())
    }

    fn tolist(&self) -> Vec<f32> {
        self.inner.to_vec()
    }

    // ---- View operations ----

    fn reshape(&self, shape: Vec<usize>) -> Self {
        PyTensor::new(self.inner.reshape(shape))
    }

    fn transpose(&self, dim0: usize, dim1: usize) -> Self {
        PyTensor::new(self.inner.transpose(dim0, dim1))
    }

    fn t(&self) -> PyResult<Self> {
        if self.inner.ndim() != 2 {
            return Err(PyValueError::new_err("t() requires a 2D tensor"));
        }
        Ok(PyTensor::new(self.inner.t()))
    }

    fn unsqueeze(&self, dim: usize) -> Self {
        PyTensor::new(self.inner.unsqueeze(dim))
    }

    #[pyo3(signature = (dim=None))]
    fn squeeze(&self, dim: Option<usize>) -> Self {
        PyTensor::new(self.inner.squeeze(dim))
    }

    fn expand(&self, shape: Vec<usize>) -> Self {
        PyTensor::new(self.inner.expand(&shape))
    }

    fn contiguous(&self) -> Self {
        PyTensor::new(self.inner.contiguous())
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

    fn __repr__(&self) -> String {
        format!("{}", self.inner)
    }

    fn __str__(&self) -> String {
        format!("{}", self.inner)
    }

    // ---- Unary ops as methods ----

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

    // ---- Reduction ops ----

    fn sum(&self) -> Self {
        PyTensor::new(ops::sum(&self.inner))
    }

    #[pyo3(signature = (dim, keepdim=false))]
    fn sum_dim(&self, dim: usize, keepdim: bool) -> Self {
        PyTensor::new(ops::sum_dim(&self.inner, dim, keepdim))
    }

    fn mean(&self) -> Self {
        PyTensor::new(ops::mean(&self.inner))
    }

    #[pyo3(signature = (dim, keepdim=false))]
    fn mean_dim(&self, dim: usize, keepdim: bool) -> Self {
        PyTensor::new(ops::mean_dim(&self.inner, dim, keepdim))
    }

    fn max(&self) -> Self {
        PyTensor::new(ops::max(&self.inner))
    }

    fn min(&self) -> Self {
        PyTensor::new(ops::min(&self.inner))
    }

    fn argmax(&self, dim: usize) -> Self {
        PyTensor::new(ops::argmax(&self.inner, dim))
    }

    // ---- Matrix ops ----

    fn matmul(&self, other: &PyTensor) -> Self {
        PyTensor::new(ops::matmul(&self.inner, &other.inner))
    }

    fn mm(&self, other: &PyTensor) -> Self {
        self.matmul(other)
    }

    fn __matmul__(&self, other: &PyTensor) -> Self {
        self.matmul(other)
    }

    // ---- Softmax ----

    #[pyo3(signature = (dim=1))]
    fn softmax(&self, dim: usize) -> Self {
        PyTensor::new(ops::softmax(&self.inner, dim))
    }

    #[pyo3(signature = (dim=1))]
    fn log_softmax(&self, dim: usize) -> Self {
        PyTensor::new(ops::log_softmax(&self.inner, dim))
    }

    // ---- In-place ops ----

    fn add_(&self, other: &PyTensor) {
        self.inner.add_(&other.inner);
    }

    fn mul_scalar_(&self, scalar: f32) {
        self.inner.mul_scalar_(scalar);
    }

    fn fill_(&self, value: f32) {
        self.inner.fill_(value);
    }

    fn sub_alpha_(&self, other: &PyTensor, alpha: f32) {
        self.inner.sub_alpha_(&other.inner, alpha);
    }

    fn copy_from(&self, src: &PyTensor) {
        self.inner.copy_from(&src.inner);
    }

    // ---- Scalar arithmetic (for operator overloading with scalars) ----

    fn add_scalar(&self, s: f32) -> Self {
        PyTensor::new(ops::add_scalar(&self.inner, s))
    }

    fn mul_scalar(&self, s: f32) -> Self {
        PyTensor::new(ops::mul_scalar(&self.inner, s))
    }

    // ---- Comparison ----

    fn eq(&self, other: &PyTensor) -> Self {
        PyTensor::new(ops::eq(&self.inner, &other.inner))
    }

    fn gt(&self, other: &PyTensor) -> Self {
        PyTensor::new(ops::gt(&self.inner, &other.inner))
    }

    // ---- Data access ----

    fn get(&self, indices: Vec<usize>) -> f32 {
        self.inner.get(&indices)
    }

    fn set(&self, indices: Vec<usize>, value: f32) {
        self.inner.set(&indices, value);
    }
}

// ---- Module-level functions ----

#[pyfunction]
fn tensor(data: Vec<f32>, shape: Vec<usize>) -> PyResult<PyTensor> {
    PyTensor::py_new(data, shape)
}

#[pyfunction]
fn zeros(shape: Vec<usize>) -> PyTensor {
    PyTensor::zeros(shape)
}

#[pyfunction]
fn ones(shape: Vec<usize>) -> PyTensor {
    PyTensor::ones(shape)
}

#[pyfunction]
fn full(shape: Vec<usize>, value: f32) -> PyTensor {
    PyTensor::full(shape, value)
}

#[pyfunction]
#[pyo3(name = "rand")]
fn py_rand(shape: Vec<usize>) -> PyTensor {
    PyTensor::rand(shape)
}

#[pyfunction]
fn randn(shape: Vec<usize>) -> PyTensor {
    PyTensor::randn(shape)
}

#[pyfunction]
fn matmul(a: &PyTensor, b: &PyTensor) -> PyTensor {
    a.matmul(b)
}

#[pyfunction]
fn relu(a: &PyTensor) -> PyTensor {
    a.relu()
}

#[pyfunction]
fn sigmoid(a: &PyTensor) -> PyTensor {
    a.sigmoid()
}

#[pyfunction]
#[pyo3(name = "tanh")]
fn py_tanh(a: &PyTensor) -> PyTensor {
    a.tanh()
}

#[pyfunction]
#[pyo3(signature = (a, dim=1))]
fn softmax(a: &PyTensor, dim: usize) -> PyTensor {
    a.softmax(dim)
}

#[pyfunction]
#[pyo3(signature = (a, dim=1))]
fn log_softmax(a: &PyTensor, dim: usize) -> PyTensor {
    a.log_softmax(dim)
}

#[pyfunction]
fn one_hot(indices: &PyTensor, num_classes: usize) -> PyTensor {
    PyTensor::new(ops::one_hot(&indices.inner, num_classes))
}

#[pyfunction]
#[pyo3(signature = (start, end, step=1.0))]
fn arange(start: f32, end: f32, step: f32) -> PyTensor {
    PyTensor::arange(start, end, step)
}

/// The cakelamp._core Python module.
#[pymodule]
#[pyo3(name = "_core")]
pub fn cakelamp_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_function(wrap_pyfunction!(tensor, m)?)?;
    m.add_function(wrap_pyfunction!(zeros, m)?)?;
    m.add_function(wrap_pyfunction!(ones, m)?)?;
    m.add_function(wrap_pyfunction!(full, m)?)?;
    m.add_function(wrap_pyfunction!(py_rand, m)?)?;
    m.add_function(wrap_pyfunction!(randn, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(relu, m)?)?;
    m.add_function(wrap_pyfunction!(sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(py_tanh, m)?)?;
    m.add_function(wrap_pyfunction!(softmax, m)?)?;
    m.add_function(wrap_pyfunction!(log_softmax, m)?)?;
    m.add_function(wrap_pyfunction!(one_hot, m)?)?;
    m.add_function(wrap_pyfunction!(arange, m)?)?;
    Ok(())
}

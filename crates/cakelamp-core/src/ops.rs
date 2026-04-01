//! Tensor operations: element-wise, reductions, matmul, activations.

use crate::broadcast::{broadcast_shape, broadcast_strides};
use crate::tensor::Tensor;

// ---- Element-wise binary ops with broadcasting ----

fn binary_op_broadcast(a: &Tensor, b: &Tensor, f: impl Fn(f32, f32) -> f32) -> Tensor {
    let out_shape = broadcast_shape(a.shape(), b.shape())
        .expect("Shapes are not broadcastable");

    let a_strides = broadcast_strides(a.shape(), a.strides(), &out_shape);
    let b_strides = broadcast_strides(b.shape(), b.strides(), &out_shape);

    let numel: usize = out_shape.iter().product();
    let ndim = out_shape.len();
    let mut result = Vec::with_capacity(numel);

    let a_storage = a.storage.borrow();
    let b_storage = b.storage.borrow();

    let mut coord = vec![0usize; ndim];
    for _ in 0..numel {
        let mut a_idx = a.storage_offset;
        let mut b_idx = b.storage_offset;
        for d in 0..ndim {
            a_idx += coord[d] * a_strides[d];
            b_idx += coord[d] * b_strides[d];
        }
        result.push(f(a_storage.data[a_idx], b_storage.data[b_idx]));

        // Increment coordinate
        for d in (0..ndim).rev() {
            coord[d] += 1;
            if coord[d] < out_shape[d] {
                break;
            }
            coord[d] = 0;
        }
    }

    Tensor::from_data(result, out_shape)
}

fn unary_op(a: &Tensor, f: impl Fn(f32) -> f32) -> Tensor {
    let data = a.to_vec();
    let result: Vec<f32> = data.iter().map(|&x| f(x)).collect();
    Tensor::from_data(result, a.shape().to_vec())
}

// ---- Element-wise arithmetic ----

/// Element-wise addition with broadcasting.
pub fn add(a: &Tensor, b: &Tensor) -> Tensor {
    binary_op_broadcast(a, b, |x, y| x + y)
}

/// Element-wise subtraction with broadcasting.
pub fn sub(a: &Tensor, b: &Tensor) -> Tensor {
    binary_op_broadcast(a, b, |x, y| x - y)
}

/// Element-wise multiplication with broadcasting.
pub fn mul(a: &Tensor, b: &Tensor) -> Tensor {
    binary_op_broadcast(a, b, |x, y| x * y)
}

/// Element-wise division with broadcasting.
pub fn div(a: &Tensor, b: &Tensor) -> Tensor {
    binary_op_broadcast(a, b, |x, y| x / y)
}

/// Element-wise power with broadcasting.
pub fn pow(a: &Tensor, b: &Tensor) -> Tensor {
    binary_op_broadcast(a, b, |x, y| x.powf(y))
}

// ---- Scalar ops ----

/// Add a scalar to every element.
pub fn add_scalar(a: &Tensor, s: f32) -> Tensor {
    unary_op(a, |x| x + s)
}

/// Multiply every element by a scalar.
pub fn mul_scalar(a: &Tensor, s: f32) -> Tensor {
    unary_op(a, |x| x * s)
}

// ---- Unary ops ----

/// Element-wise negation.
pub fn neg(a: &Tensor) -> Tensor {
    unary_op(a, |x| -x)
}

/// Element-wise exponential.
pub fn exp(a: &Tensor) -> Tensor {
    unary_op(a, |x| x.exp())
}

/// Element-wise natural logarithm.
pub fn log(a: &Tensor) -> Tensor {
    unary_op(a, |x| x.ln())
}

/// Element-wise ReLU: max(0, x).
pub fn relu(a: &Tensor) -> Tensor {
    unary_op(a, |x| x.max(0.0))
}

/// Element-wise sigmoid: 1 / (1 + exp(-x)).
pub fn sigmoid(a: &Tensor) -> Tensor {
    unary_op(a, |x| 1.0 / (1.0 + (-x).exp()))
}

/// Element-wise tanh.
pub fn tanh(a: &Tensor) -> Tensor {
    unary_op(a, |x| x.tanh())
}

/// Element-wise absolute value.
pub fn abs(a: &Tensor) -> Tensor {
    unary_op(a, |x| x.abs())
}

/// Element-wise square root.
pub fn sqrt(a: &Tensor) -> Tensor {
    unary_op(a, |x| x.sqrt())
}

/// Element-wise clamp.
pub fn clamp(a: &Tensor, min: f32, max: f32) -> Tensor {
    unary_op(a, |x| x.clamp(min, max))
}

// ---- Comparison ops ----

/// Element-wise equality (returns 1.0 for true, 0.0 for false).
pub fn eq(a: &Tensor, b: &Tensor) -> Tensor {
    binary_op_broadcast(a, b, |x, y| if (x - y).abs() < 1e-7 { 1.0 } else { 0.0 })
}

/// Element-wise greater than.
pub fn gt(a: &Tensor, b: &Tensor) -> Tensor {
    binary_op_broadcast(a, b, |x, y| if x > y { 1.0 } else { 0.0 })
}

// ---- Reduction ops ----

/// Sum all elements, returning a scalar tensor.
pub fn sum(a: &Tensor) -> Tensor {
    let data = a.to_vec();
    let s: f32 = data.iter().sum();
    Tensor::scalar(s)
}

/// Sum along a specific dimension.
pub fn sum_dim(a: &Tensor, dim: usize, keepdim: bool) -> Tensor {
    assert!(dim < a.ndim(), "Dimension out of range");
    let shape = a.shape();
    let dim_size = shape[dim];

    // Build output shape
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape[dim] = 1;

    let out_numel: usize = out_shape.iter().product();
    let mut result = vec![0.0f32; out_numel];
    let out_strides = Tensor::compute_contiguous_strides(&out_shape);

    let data = a.to_vec();
    let in_strides = Tensor::compute_contiguous_strides(shape);

    // Iterate over all positions in the output
    let ndim = shape.len();
    let mut coord = vec![0usize; ndim];
    for _ in 0..out_numel {
        let mut out_idx = 0;
        for d in 0..ndim {
            out_idx += coord[d] * out_strides[d];
        }

        // Sum along the reduction dimension
        let mut s = 0.0;
        for k in 0..dim_size {
            let mut in_idx = 0;
            for d in 0..ndim {
                if d == dim {
                    in_idx += k * in_strides[d];
                } else {
                    in_idx += coord[d] * in_strides[d];
                }
            }
            s += data[in_idx];
        }
        result[out_idx] = s;

        // Increment coord (skip the reduction dim)
        for d in (0..ndim).rev() {
            if d == dim {
                continue;
            }
            coord[d] += 1;
            if coord[d] < out_shape[d] {
                break;
            }
            coord[d] = 0;
        }
    }

    if keepdim {
        Tensor::from_data(result, out_shape)
    } else {
        let mut squeezed_shape: Vec<usize> = Vec::new();
        for (i, &s) in out_shape.iter().enumerate() {
            if i != dim {
                squeezed_shape.push(s);
            }
        }
        if squeezed_shape.is_empty() {
            Tensor::scalar(result[0])
        } else {
            Tensor::from_data(result, squeezed_shape)
        }
    }
}

/// Mean of all elements, returning a scalar tensor.
pub fn mean(a: &Tensor) -> Tensor {
    let s = sum(a).item();
    Tensor::scalar(s / a.numel() as f32)
}

/// Mean along a specific dimension.
pub fn mean_dim(a: &Tensor, dim: usize, keepdim: bool) -> Tensor {
    let dim_size = a.shape()[dim] as f32;
    let s = sum_dim(a, dim, keepdim);
    mul_scalar(&s, 1.0 / dim_size)
}

/// Max of all elements, returning a scalar tensor.
pub fn max(a: &Tensor) -> Tensor {
    let data = a.to_vec();
    let m = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    Tensor::scalar(m)
}

/// Min of all elements, returning a scalar tensor.
pub fn min(a: &Tensor) -> Tensor {
    let data = a.to_vec();
    let m = data.iter().cloned().fold(f32::INFINITY, f32::min);
    Tensor::scalar(m)
}

/// Argmax along a dimension.
pub fn argmax(a: &Tensor, dim: usize) -> Tensor {
    assert!(dim < a.ndim(), "Dimension out of range");
    let shape = a.shape();
    let dim_size = shape[dim];

    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(dim);
    if out_shape.is_empty() {
        // 1D tensor
        let data = a.to_vec();
        let (idx, _) = data.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        return Tensor::scalar(idx as f32);
    }

    let out_numel: usize = out_shape.iter().product();
    let mut result = vec![0.0f32; out_numel];

    let data = a.to_vec();
    let in_strides = Tensor::compute_contiguous_strides(shape);

    let ndim = out_shape.len();
    let mut coord = vec![0usize; ndim];
    for out_i in 0..out_numel {
        let mut best_val = f32::NEG_INFINITY;
        let mut best_idx = 0usize;

        for k in 0..dim_size {
            // Reconstruct the full input coordinate
            let mut in_idx = 0;
            let mut out_d = 0;
            for d in 0..shape.len() {
                if d == dim {
                    in_idx += k * in_strides[d];
                } else {
                    in_idx += coord[out_d] * in_strides[d];
                    out_d += 1;
                }
            }
            if data[in_idx] > best_val {
                best_val = data[in_idx];
                best_idx = k;
            }
        }
        result[out_i] = best_idx as f32;

        // Increment coord
        for d in (0..ndim).rev() {
            coord[d] += 1;
            if coord[d] < out_shape[d] {
                break;
            }
            coord[d] = 0;
        }
    }

    Tensor::from_data(result, out_shape)
}

// ---- Matrix operations ----

/// Matrix multiplication for 2D tensors.
/// a: (M, K), b: (K, N) -> result: (M, N)
pub fn matmul(a: &Tensor, b: &Tensor) -> Tensor {
    assert_eq!(a.ndim(), 2, "matmul requires 2D tensors");
    assert_eq!(b.ndim(), 2, "matmul requires 2D tensors");
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];
    assert_eq!(k, b.shape()[0], "Inner dimensions must match for matmul");

    let a_data = a.to_vec();
    let b_data = b.contiguous().to_vec();
    let mut result = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f32;
            for p in 0..k {
                s += a_data[i * k + p] * b_data[p * n + j];
            }
            result[i * n + j] = s;
        }
    }

    Tensor::from_data(result, vec![m, n])
}

/// Batched matrix multiplication.
/// Handles broadcasting of batch dimensions.
/// a: (..., M, K), b: (..., K, N) -> (..., M, N)
pub fn bmm(a: &Tensor, b: &Tensor) -> Tensor {
    assert!(a.ndim() >= 2 && b.ndim() >= 2, "bmm requires at least 2D tensors");

    if a.ndim() == 2 && b.ndim() == 2 {
        return matmul(a, b);
    }

    let a_shape = a.shape();
    let b_shape = b.shape();
    let m = a_shape[a_shape.len() - 2];
    let k = a_shape[a_shape.len() - 1];
    let n = b_shape[b_shape.len() - 1];
    assert_eq!(k, b_shape[b_shape.len() - 2], "Inner dimensions must match");

    // For now, handle 3D case: (B, M, K) x (B, K, N)
    assert_eq!(a.ndim(), 3, "bmm currently supports 3D tensors");
    assert_eq!(b.ndim(), 3, "bmm currently supports 3D tensors");
    let batch = a_shape[0];
    assert_eq!(batch, b_shape[0], "Batch dimensions must match");

    let a_data = a.contiguous().to_vec();
    let b_data = b.contiguous().to_vec();
    let mut result = vec![0.0f32; batch * m * n];

    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for p in 0..k {
                    s += a_data[bi * m * k + i * k + p] * b_data[bi * k * n + p * n + j];
                }
                result[bi * m * n + i * n + j] = s;
            }
        }
    }

    Tensor::from_data(result, vec![batch, m, n])
}

// ---- Softmax ----

/// Softmax along a dimension.
pub fn softmax(a: &Tensor, dim: usize) -> Tensor {
    // Subtract max for numerical stability
    let shape = a.shape();
    let data = a.to_vec();
    let strides = Tensor::compute_contiguous_strides(shape);
    let ndim = shape.len();

    let mut out_shape = shape.to_vec();
    out_shape[dim] = 1;
    let out_numel: usize = out_shape.iter().product();
    let out_strides = Tensor::compute_contiguous_strides(&out_shape);

    // First pass: find max along dim
    let mut maxes = vec![f32::NEG_INFINITY; out_numel];
    let mut coord = vec![0usize; ndim];
    let total: usize = shape.iter().product();
    for _ in 0..total {
        let mut in_idx = 0;
        let mut out_idx = 0;
        for d in 0..ndim {
            in_idx += coord[d] * strides[d];
            if d != dim {
                out_idx += coord[d] * out_strides[d];
            }
        }
        if data[in_idx] > maxes[out_idx] {
            maxes[out_idx] = data[in_idx];
        }
        for d in (0..ndim).rev() {
            coord[d] += 1;
            if coord[d] < shape[d] { break; }
            coord[d] = 0;
        }
    }

    // Second pass: exp(x - max) and sum
    let mut exp_sums = vec![0.0f32; out_numel];
    let mut result = vec![0.0f32; total];
    coord = vec![0usize; ndim];
    for flat_i in 0..total {
        let mut in_idx = 0;
        let mut out_idx = 0;
        for d in 0..ndim {
            in_idx += coord[d] * strides[d];
            if d != dim {
                out_idx += coord[d] * out_strides[d];
            }
        }
        let e = (data[in_idx] - maxes[out_idx]).exp();
        result[flat_i] = e;
        exp_sums[out_idx] += e;
        for d in (0..ndim).rev() {
            coord[d] += 1;
            if coord[d] < shape[d] { break; }
            coord[d] = 0;
        }
    }

    // Third pass: normalize
    coord = vec![0usize; ndim];
    for flat_i in 0..total {
        let mut out_idx = 0;
        for d in 0..ndim {
            if d != dim {
                out_idx += coord[d] * out_strides[d];
            }
        }
        result[flat_i] /= exp_sums[out_idx];
        for d in (0..ndim).rev() {
            coord[d] += 1;
            if coord[d] < shape[d] { break; }
            coord[d] = 0;
        }
    }

    Tensor::from_data(result, shape.to_vec())
}

/// Log-softmax along a dimension.
pub fn log_softmax(a: &Tensor, dim: usize) -> Tensor {
    let sm = softmax(a, dim);
    log(&sm)
}

// ---- Gather / indexing ----

/// Select elements along a dimension by index.
/// Like PyTorch's index_select.
pub fn index_select(a: &Tensor, dim: usize, indices: &[usize]) -> Tensor {
    let shape = a.shape();
    assert!(dim < a.ndim());
    let data = a.contiguous().to_vec();
    let in_strides = Tensor::compute_contiguous_strides(shape);

    let mut out_shape = shape.to_vec();
    out_shape[dim] = indices.len();
    let out_numel: usize = out_shape.iter().product();
    let out_strides = Tensor::compute_contiguous_strides(&out_shape);
    let ndim = shape.len();

    let mut result = vec![0.0f32; out_numel];
    let mut coord = vec![0usize; ndim];
    for _ in 0..out_numel {
        let mut out_idx = 0;
        let mut in_idx = 0;
        for d in 0..ndim {
            out_idx += coord[d] * out_strides[d];
            if d == dim {
                in_idx += indices[coord[d]] * in_strides[d];
            } else {
                in_idx += coord[d] * in_strides[d];
            }
        }
        result[out_idx] = data[in_idx];

        for d in (0..ndim).rev() {
            coord[d] += 1;
            if coord[d] < out_shape[d] { break; }
            coord[d] = 0;
        }
    }

    Tensor::from_data(result, out_shape)
}

// ---- One-hot encoding ----

/// Create one-hot encoded tensor from class indices.
/// indices: (N,) tensor with class indices, num_classes: number of classes
/// Returns: (N, num_classes) tensor
pub fn one_hot(indices: &Tensor, num_classes: usize) -> Tensor {
    assert_eq!(indices.ndim(), 1, "one_hot requires 1D input");
    let n = indices.shape()[0];
    let idx_data = indices.to_vec();
    let mut result = vec![0.0f32; n * num_classes];
    for (i, &idx) in idx_data.iter().enumerate() {
        let c = idx as usize;
        assert!(c < num_classes, "Class index out of range");
        result[i * num_classes + c] = 1.0;
    }
    Tensor::from_data(result, vec![n, num_classes])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::from_data(vec![4.0, 5.0, 6.0], vec![3]);
        let c = add(&a, &b);
        assert_eq!(c.to_vec(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_broadcast_add() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_data(vec![10.0, 20.0, 30.0], vec![3]);
        let c = add(&a, &b);
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.to_vec(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_mul() {
        let a = Tensor::from_data(vec![2.0, 3.0], vec![2]);
        let b = Tensor::from_data(vec![4.0, 5.0], vec![2]);
        let c = mul(&a, &b);
        assert_eq!(c.to_vec(), vec![8.0, 15.0]);
    }

    #[test]
    fn test_neg() {
        let a = Tensor::from_data(vec![1.0, -2.0, 3.0], vec![3]);
        let b = neg(&a);
        assert_eq!(b.to_vec(), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_exp_log() {
        let a = Tensor::from_data(vec![0.0, 1.0], vec![2]);
        let e = exp(&a);
        assert!((e.get(&[0]) - 1.0).abs() < 1e-6);
        assert!((e.get(&[1]) - std::f32::consts::E).abs() < 1e-5);

        let l = log(&e);
        assert!((l.get(&[0]) - 0.0).abs() < 1e-6);
        assert!((l.get(&[1]) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sum() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let s = sum(&a);
        assert_eq!(s.item(), 10.0);
    }

    #[test]
    fn test_sum_dim() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let s0 = sum_dim(&a, 0, false);
        assert_eq!(s0.shape(), &[3]);
        assert_eq!(s0.to_vec(), vec![5.0, 7.0, 9.0]);

        let s1 = sum_dim(&a, 1, false);
        assert_eq!(s1.shape(), &[2]);
        assert_eq!(s1.to_vec(), vec![6.0, 15.0]);

        let s1k = sum_dim(&a, 1, true);
        assert_eq!(s1k.shape(), &[2, 1]);
    }

    #[test]
    fn test_mean() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let m = mean(&a);
        assert_eq!(m.item(), 2.5);
    }

    #[test]
    fn test_max_min() {
        let a = Tensor::from_data(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], vec![6]);
        assert_eq!(max(&a).item(), 9.0);
        assert_eq!(min(&a).item(), 1.0);
    }

    #[test]
    fn test_matmul() {
        // [[1,2],[3,4]] x [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_non_square() {
        // (2,3) x (3,2) = (2,2)
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let c = matmul(&a, &b);
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.to_vec(), vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_relu() {
        let a = Tensor::from_data(vec![-1.0, 0.0, 1.0, -0.5, 2.0], vec![5]);
        let r = relu(&a);
        assert_eq!(r.to_vec(), vec![0.0, 0.0, 1.0, 0.0, 2.0]);
    }

    #[test]
    fn test_softmax() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let s = softmax(&a, 1);
        let data = s.to_vec();
        let total: f32 = data.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
        // values should be increasing
        assert!(data[0] < data[1]);
        assert!(data[1] < data[2]);
    }

    #[test]
    fn test_argmax() {
        let a = Tensor::from_data(vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0], vec![2, 3]);
        let am = argmax(&a, 1);
        assert_eq!(am.shape(), &[2]);
        assert_eq!(am.to_vec(), vec![1.0, 2.0]);
    }

    #[test]
    fn test_scalar_ops() {
        let a = Tensor::from_data(vec![1.0, 2.0, 3.0], vec![3]);
        let b = add_scalar(&a, 10.0);
        assert_eq!(b.to_vec(), vec![11.0, 12.0, 13.0]);

        let c = mul_scalar(&a, 2.0);
        assert_eq!(c.to_vec(), vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_one_hot() {
        let indices = Tensor::from_data(vec![0.0, 2.0, 1.0], vec![3]);
        let oh = one_hot(&indices, 3);
        assert_eq!(oh.shape(), &[3, 3]);
        assert_eq!(oh.to_vec(), vec![
            1.0, 0.0, 0.0,
            0.0, 0.0, 1.0,
            0.0, 1.0, 0.0,
        ]);
    }
}

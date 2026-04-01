"""Autograd operations with backward (gradient) implementations."""

from __future__ import annotations
from typing import Tuple, Optional
import cakelamp._core as _C
from cakelamp.autograd.function import Function
from cakelamp.autograd.engine import AutogradTensor


class Add(Function):
    def forward(self, a: AutogradTensor, b: AutogradTensor):
        self.save_for_backward(a, b)
        return a.data + b.data

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        return grad_output, grad_output


class Sub(Function):
    def forward(self, a: AutogradTensor, b: AutogradTensor):
        self.save_for_backward(a, b)
        return a.data - b.data

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        return grad_output, AutogradTensor(-grad_output.data)


class Mul(Function):
    def forward(self, a: AutogradTensor, b: AutogradTensor):
        self.save_for_backward(a, b)
        return a.data * b.data

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        a, b = self.saved_tensors
        grad_a = AutogradTensor(grad_output.data * b.data)
        grad_b = AutogradTensor(grad_output.data * a.data)
        return grad_a, grad_b


class Div(Function):
    def forward(self, a: AutogradTensor, b: AutogradTensor):
        self.save_for_backward(a, b)
        return a.data / b.data

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        a, b = self.saved_tensors
        grad_a = AutogradTensor(grad_output.data / b.data)
        # d(a/b)/db = -a/b^2
        grad_b = AutogradTensor(-(grad_output.data * a.data) / (b.data * b.data))
        return grad_a, grad_b


class Neg(Function):
    def forward(self, a: AutogradTensor):
        return -a.data

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        return (AutogradTensor(-grad_output.data),)


class Pow(Function):
    def forward(self, base: AutogradTensor, exponent: AutogradTensor):
        self.save_for_backward(base, exponent)
        # Use the Rust pow op
        return _C.Tensor(
            [b ** e for b, e in zip(base.data.tolist(), exponent.data.tolist())],
            base.data.shape
        )

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        base, exponent = self.saved_tensors
        # d(x^n)/dx = n * x^(n-1)
        exp_data = exponent.data.tolist()
        base_data = base.data.tolist()
        grad_base_data = [
            g * e * (b ** (e - 1))
            for g, b, e in zip(grad_output.data.tolist(), base_data, exp_data)
        ]
        grad_base = AutogradTensor(_C.Tensor(grad_base_data, base.data.shape))
        return grad_base, None


class MatMul(Function):
    def forward(self, a: AutogradTensor, b: AutogradTensor):
        self.save_for_backward(a, b)
        return a.data.matmul(b.data)

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        a, b = self.saved_tensors
        # grad_a = grad_output @ b.T
        grad_a = AutogradTensor(grad_output.data.matmul(b.data.t()))
        # grad_b = a.T @ grad_output
        grad_b = AutogradTensor(a.data.t().contiguous().matmul(grad_output.data))
        return grad_a, grad_b


class Sum(Function):
    def forward(self, a: AutogradTensor):
        self.save_for_backward(a)
        return a.data.sum()

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        a, = self.saved_tensors
        # Gradient of sum is ones * grad_output
        grad = AutogradTensor(
            _C.Tensor(
                [grad_output.data.item()] * a.data.numel(),
                a.data.shape
            )
        )
        return (grad,)


class SumDim(Function):
    def forward(self, a: AutogradTensor, dim: int, keepdim: bool):
        self.save_for_backward(a)
        self.dim = dim
        self.keepdim = keepdim
        return a.data.sum_dim(dim, keepdim)

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        a, = self.saved_tensors
        g = grad_output.data
        if not self.keepdim:
            # Need to unsqueeze the reduced dimension back
            # Create tensor with shape matching input by expanding
            shape = list(a.data.shape)
            shape[self.dim] = 1
            g = _C.Tensor(g.tolist(), shape)
        # Expand to original shape
        expanded = g.expand(a.data.shape)
        return (AutogradTensor(_C.Tensor(expanded.tolist(), a.data.shape)),)


class Mean(Function):
    def forward(self, a: AutogradTensor):
        self.save_for_backward(a)
        return a.data.mean()

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        a, = self.saved_tensors
        n = a.data.numel()
        val = grad_output.data.item() / n
        grad = AutogradTensor(_C.Tensor([val] * n, a.data.shape))
        return (grad,)


class Exp(Function):
    def forward(self, a: AutogradTensor):
        result = a.data.exp()
        self.save_for_backward(AutogradTensor(result))
        return result

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        exp_a, = self.saved_tensors
        return (AutogradTensor(grad_output.data * exp_a.data),)


class Log(Function):
    def forward(self, a: AutogradTensor):
        self.save_for_backward(a)
        return a.data.log()

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        a, = self.saved_tensors
        return (AutogradTensor(grad_output.data / a.data),)


class ReLU(Function):
    def forward(self, a: AutogradTensor):
        self.save_for_backward(a)
        return a.data.relu()

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        a, = self.saved_tensors
        # Gradient is 1 where input > 0, else 0
        mask_data = [1.0 if x > 0 else 0.0 for x in a.data.tolist()]
        mask = _C.Tensor(mask_data, a.data.shape)
        return (AutogradTensor(grad_output.data * mask),)


class Sigmoid(Function):
    def forward(self, a: AutogradTensor):
        result = a.data.sigmoid()
        self.save_for_backward(AutogradTensor(result))
        return result

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        sig, = self.saved_tensors
        # sigmoid' = sigmoid * (1 - sigmoid)
        ones = _C.ones(sig.data.shape)
        grad = grad_output.data * sig.data * (ones - sig.data)
        return (AutogradTensor(grad),)


class Tanh(Function):
    def forward(self, a: AutogradTensor):
        result = a.data.tanh()
        self.save_for_backward(AutogradTensor(result))
        return result

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        tanh_a, = self.saved_tensors
        # tanh' = 1 - tanh^2
        ones = _C.ones(tanh_a.data.shape)
        grad = grad_output.data * (ones - tanh_a.data * tanh_a.data)
        return (AutogradTensor(grad),)


class Reshape(Function):
    def forward(self, a: AutogradTensor, shape):
        self.save_for_backward(a)
        self.orig_shape = a.data.shape
        return a.data.reshape(list(shape))

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        return (AutogradTensor(grad_output.data.reshape(list(self.orig_shape))),)


class Transpose(Function):
    """Transpose for 2D tensors (.t())."""
    def forward(self, a: AutogradTensor):
        return a.data.t()

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        return (AutogradTensor(grad_output.data.t().contiguous()),)


class TransposeDims(Function):
    def forward(self, a: AutogradTensor, dim0: int, dim1: int):
        self.dim0 = dim0
        self.dim1 = dim1
        return a.data.transpose(dim0, dim1)

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        return (AutogradTensor(
            _C.Tensor(
                grad_output.data.transpose(self.dim0, self.dim1).contiguous().tolist(),
                grad_output.data.transpose(self.dim0, self.dim1).shape
            )
        ),)


class Softmax(Function):
    def forward(self, a: AutogradTensor, dim: int):
        self.dim = dim
        result = a.data.softmax(dim)
        self.save_for_backward(AutogradTensor(result))
        return result

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        sm, = self.saved_tensors
        # Jacobian-vector product for softmax
        # grad_input = sm * (grad_output - sum(grad_output * sm, dim))
        prod = grad_output.data * sm.data
        s = prod.sum_dim(self.dim, keepdim=True)
        s_expanded = _C.Tensor(s.expand(sm.data.shape).tolist(), sm.data.shape)
        grad = sm.data * (grad_output.data - s_expanded)
        return (AutogradTensor(grad),)


class LogSoftmax(Function):
    def forward(self, a: AutogradTensor, dim: int):
        self.dim = dim
        result = a.data.log_softmax(dim)
        self.save_for_backward(AutogradTensor(result))
        return result

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        log_sm, = self.saved_tensors
        # grad = grad_output - softmax * sum(grad_output, dim)
        sm_data = log_sm.data.exp()
        s = grad_output.data.sum_dim(self.dim, keepdim=True)
        s_expanded = _C.Tensor(s.expand(sm_data.shape).tolist(), sm_data.shape)
        grad = grad_output.data - sm_data * s_expanded
        return (AutogradTensor(grad),)


class Unsqueeze(Function):
    def forward(self, a: AutogradTensor, dim: int):
        self.dim = dim
        return a.data.unsqueeze(dim)

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        return (AutogradTensor(grad_output.data.squeeze(self.dim)),)


class Squeeze(Function):
    def forward(self, a: AutogradTensor, dim):
        self.save_for_backward(a)
        self.dim = dim
        return a.data.squeeze(dim)

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        a, = self.saved_tensors
        # Restore original shape
        return (AutogradTensor(grad_output.data.reshape(list(a.data.shape))),)


class Expand(Function):
    def forward(self, a: AutogradTensor, shape):
        self.save_for_backward(a)
        self.new_shape = shape
        result = a.data.expand(list(shape))
        return _C.Tensor(result.tolist(), list(shape))

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        a, = self.saved_tensors
        # Sum over expanded dimensions
        from cakelamp.autograd.engine import _unbroadcast
        return (_unbroadcast(grad_output, a.data.shape),)


class Clamp(Function):
    def forward(self, a: AutogradTensor, min_val: float, max_val: float):
        self.save_for_backward(a)
        self.min_val = min_val
        self.max_val = max_val
        return a.data.clamp(min_val, max_val)

    def backward(self, grad_output: AutogradTensor) -> Tuple:
        a, = self.saved_tensors
        mask_data = [
            1.0 if self.min_val < x < self.max_val else 0.0
            for x in a.data.tolist()
        ]
        mask = _C.Tensor(mask_data, a.data.shape)
        return (AutogradTensor(grad_output.data * mask),)

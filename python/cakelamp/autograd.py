"""Autograd engine: reverse-mode automatic differentiation.

Each differentiable op creates a Function (backward node) that records
the computation graph. Tensor.backward() does topological sort + reverse
traversal to compute gradients.
"""

from cakelamp import backend as B
import math


class Function:
    """Base class for autograd functions (backward nodes)."""

    def inputs(self):
        """Return list of input tensors."""
        raise NotImplementedError

    def backward(self, grad_output):
        """Compute gradients w.r.t. inputs. Returns list of gradients."""
        raise NotImplementedError


def _sum_to_shape(grad, target_shape):
    """Sum grad to match target_shape (undo broadcasting)."""
    from cakelamp.tensor import Tensor

    if not isinstance(grad, Tensor):
        return grad

    grad_shape = list(grad._shape)
    target = list(target_shape)

    if grad_shape == target:
        return grad

    # Handle scalar target
    if not target:
        return grad.sum()

    # Pad target shape with 1s on the left to match grad ndim
    while len(target) < len(grad_shape):
        target = [1] + target

    # Sum along dimensions that were broadcast
    data = grad._contiguous_data()
    shape = grad_shape[:]
    for i in range(len(shape)):
        if i >= len(target_shape) + (len(shape) - len(target_shape)):
            continue
        ti = i - (len(shape) - len(target_shape))
        if ti < 0:
            # Extra leading dims - sum them out
            data, shape = B.sum_dim(data, shape, 0, False)
            continue
        t = target_shape[ti] if ti < len(target_shape) else 1
        if shape[i] != t and t == 1:
            data, shape = B.sum_dim(data, shape, i, True)

    # Remove extra leading dims
    while len(shape) > len(target_shape):
        data, shape = B.sum_dim(data, shape, 0, False)

    result = Tensor._make(data, list(target_shape))
    return result


class AddBackward(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def inputs(self):
        return [self.a, self.b]

    def backward(self, grad_output):
        grad_a = _sum_to_shape(grad_output, self.a._shape)
        grad_b = _sum_to_shape(grad_output, self.b._shape)
        return [grad_a, grad_b]


class SubBackward(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def inputs(self):
        return [self.a, self.b]

    def backward(self, grad_output):
        grad_a = _sum_to_shape(grad_output, self.a._shape)
        grad_b = _sum_to_shape(-grad_output, self.b._shape)
        return [grad_a, grad_b]


class MulBackward(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def inputs(self):
        return [self.a, self.b]

    def backward(self, grad_output):
        from cakelamp.tensor import Tensor
        # d(a*b)/da = b, d(a*b)/db = a
        grad_a = _sum_to_shape(grad_output * self.b.detach(), self.a._shape)
        grad_b = _sum_to_shape(grad_output * self.a.detach(), self.b._shape)
        return [grad_a, grad_b]


class DivBackward(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def inputs(self):
        return [self.a, self.b]

    def backward(self, grad_output):
        from cakelamp.tensor import Tensor
        # d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        grad_a = _sum_to_shape(grad_output / self.b.detach(), self.a._shape)
        grad_b = _sum_to_shape(
            -grad_output * self.a.detach() / (self.b.detach() ** 2),
            self.b._shape
        )
        return [grad_a, grad_b]


class NegBackward(Function):
    def __init__(self, a):
        self.a = a

    def inputs(self):
        return [self.a]

    def backward(self, grad_output):
        return [-grad_output]


class PowBackward(Function):
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent

    def inputs(self):
        return [self.base, self.exponent]

    def backward(self, grad_output):
        from cakelamp.tensor import Tensor
        # d(a^b)/da = b * a^(b-1)
        grad_base = _sum_to_shape(
            grad_output * self.exponent.detach() * (self.base.detach() ** (self.exponent.detach() - 1)),
            self.base._shape
        )
        # d(a^b)/db = a^b * ln(a)  (if needed)
        grad_exp = None
        if self.exponent.requires_grad:
            grad_exp = _sum_to_shape(
                grad_output * (self.base.detach() ** self.exponent.detach()) * self.base.detach().log(),
                self.exponent._shape
            )
        return [grad_base, grad_exp]


class ExpBackward(Function):
    def __init__(self, input_tensor, output_tensor):
        self.input = input_tensor
        self.output = output_tensor  # exp(x) is its own derivative

    def inputs(self):
        return [self.input]

    def backward(self, grad_output):
        # d(exp(x))/dx = exp(x)
        return [grad_output * self.output.detach()]


class LogBackward(Function):
    def __init__(self, input_tensor):
        self.input = input_tensor

    def inputs(self):
        return [self.input]

    def backward(self, grad_output):
        from cakelamp.tensor import Tensor, _ensure_tensor
        # d(ln(x))/dx = 1/x
        return [grad_output / self.input.detach()]


class ReluBackward(Function):
    def __init__(self, input_tensor):
        self.input = input_tensor

    def inputs(self):
        return [self.input]

    def backward(self, grad_output):
        from cakelamp.tensor import Tensor
        # d(relu(x))/dx = 1 if x > 0, else 0
        input_data = self.input._contiguous_data()
        mask_data = [1.0 if x > 0 else 0.0 for x in input_data]
        mask = Tensor._make(mask_data, list(self.input._shape))
        return [grad_output * mask]


class SigmoidBackward(Function):
    def __init__(self, input_tensor, output_tensor):
        self.input = input_tensor
        self.output = output_tensor

    def inputs(self):
        return [self.input]

    def backward(self, grad_output):
        from cakelamp.tensor import Tensor
        # d(sigmoid(x))/dx = sigmoid(x) * (1 - sigmoid(x))
        s = self.output.detach()
        return [grad_output * s * (Tensor.ones(list(s._shape)) - s)]


class TanhBackward(Function):
    def __init__(self, input_tensor, output_tensor):
        self.input = input_tensor
        self.output = output_tensor

    def inputs(self):
        return [self.input]

    def backward(self, grad_output):
        from cakelamp.tensor import Tensor
        # d(tanh(x))/dx = 1 - tanh(x)^2
        t = self.output.detach()
        return [grad_output * (Tensor.ones(list(t._shape)) - t * t)]


class SumBackward(Function):
    def __init__(self, input_tensor, dim=None, keepdim=False):
        self.input = input_tensor
        self.dim = dim
        self.keepdim = keepdim

    def inputs(self):
        return [self.input]

    def backward(self, grad_output):
        from cakelamp.tensor import Tensor
        # Gradient of sum is all ones (or expand)
        if self.dim is None:
            # Sum of all elements: grad is scalar expanded to input shape
            return [Tensor.ones(list(self.input._shape)) * grad_output]
        else:
            # Sum along dim: expand grad back
            if not self.keepdim:
                # Re-insert the reduced dimension
                grad = grad_output.unsqueeze(self.dim)
            else:
                grad = grad_output
            return [grad.expand(list(self.input._shape))]


class MeanBackward(Function):
    def __init__(self, input_tensor, dim=None, keepdim=False):
        self.input = input_tensor
        self.dim = dim
        self.keepdim = keepdim

    def inputs(self):
        return [self.input]

    def backward(self, grad_output):
        from cakelamp.tensor import Tensor
        if self.dim is None:
            n = self.input.numel
            return [Tensor.ones(list(self.input._shape)) * grad_output / n]
        else:
            n = self.input._shape[self.dim]
            if not self.keepdim:
                grad = grad_output.unsqueeze(self.dim)
            else:
                grad = grad_output
            return [grad.expand(list(self.input._shape)) / n]


class MatmulBackward(Function):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    def inputs(self):
        return [self.a, self.b]

    def backward(self, grad_output):
        # d(A @ B)/dA = grad_output @ B^T
        # d(A @ B)/dB = A^T @ grad_output
        grad_a = grad_output.mm(self.b.detach().transpose(0, 1))
        grad_b = self.a.detach().transpose(0, 1).mm(grad_output)
        return [grad_a, grad_b]


class ReshapeBackward(Function):
    def __init__(self, input_tensor):
        self.input = input_tensor

    def inputs(self):
        return [self.input]

    def backward(self, grad_output):
        return [grad_output.reshape(list(self.input._shape))]


class TransposeBackward(Function):
    def __init__(self, input_tensor, dim0, dim1):
        self.input = input_tensor
        self.dim0 = dim0
        self.dim1 = dim1

    def inputs(self):
        return [self.input]

    def backward(self, grad_output):
        return [grad_output.transpose(self.dim0, self.dim1)]


class LogSoftmaxBackward(Function):
    def __init__(self, input_tensor, softmax_output, dim):
        self.input = input_tensor
        self.softmax = softmax_output
        self.dim = dim

    def inputs(self):
        return [self.input]

    def backward(self, grad_output):
        # d(log_softmax)/dx = grad - softmax * sum(grad, dim)
        sm = self.softmax
        grad_sum = grad_output.sum(dim=self.dim, keepdim=True)
        return [grad_output - sm * grad_sum.expand(list(self.input._shape))]

"""Autograd Function (Node) base class and built-in backward functions.

Each differentiable operation creates a Function node in the computation
graph. The node stores:
  - saved_tensors: input tensors needed for backward computation
  - inputs: direct references to input AutogradTensors (for graph traversal)

During backward(), we do topological sort + reverse traversal.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Optional, List

if TYPE_CHECKING:
    from cakelamp.autograd.tensor import AutogradTensor


class Function:
    """Base class for autograd functions (computation graph nodes)."""

    def __init__(self):
        self.saved_tensors: List[AutogradTensor] = []
        self.inputs: List[AutogradTensor] = []

    def save_for_backward(self, *tensors: AutogradTensor):
        """Save tensors for use in backward()."""
        self.saved_tensors = list(tensors)

    def backward(self, grad_output: AutogradTensor) -> Tuple[Optional[AutogradTensor], ...]:
        """Compute gradients. Must be overridden by subclasses."""
        raise NotImplementedError


# ── Built-in Functions ────────────────────────────────────────────────


class AddBackward(Function):
    """Backward for a + b."""

    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = _unbroadcast(grad_output, a.shape)
        grad_b = _unbroadcast(grad_output, b.shape)
        return grad_a, grad_b


class SubBackward(Function):
    """Backward for a - b."""

    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = _unbroadcast(grad_output, a.shape)
        grad_b = _unbroadcast(-grad_output, b.shape)
        return grad_a, grad_b


class MulBackward(Function):
    """Backward for a * b (element-wise)."""

    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = _unbroadcast(grad_output * b, a.shape)
        grad_b = _unbroadcast(grad_output * a, b.shape)
        return grad_a, grad_b


class DivBackward(Function):
    """Backward for a / b."""

    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = _unbroadcast(grad_output / b, a.shape)
        grad_b = _unbroadcast(grad_output * (-a / (b * b)), b.shape)
        return grad_a, grad_b


class NegBackward(Function):
    """Backward for -a."""

    def backward(self, grad_output):
        return (-grad_output,)


class MatmulBackward(Function):
    """Backward for a @ b (2D matrix multiply)."""

    def backward(self, grad_output):
        a, b = self.saved_tensors
        grad_a = grad_output.matmul(b.t())
        grad_b = a.t().matmul(grad_output)
        return grad_a, grad_b


class ExpBackward(Function):
    """Backward for exp(a)."""

    def __init__(self):
        super().__init__()
        self.result = None

    def backward(self, grad_output):
        return (grad_output * self.result,)


class LogBackward(Function):
    """Backward for log(a)."""

    def backward(self, grad_output):
        (a,) = self.saved_tensors
        return (grad_output / a,)


class ReluBackward(Function):
    """Backward for relu(a)."""

    def backward(self, grad_output):
        (a,) = self.saved_tensors
        from cakelamp.autograd.tensor import AutogradTensor
        import cakelamp._core as _core
        zero = AutogradTensor(_core.zeros(list(a.shape)), requires_grad=False)
        mask = AutogradTensor(a.data.gt(zero.data), requires_grad=False)
        return (grad_output * mask,)


class SigmoidBackward(Function):
    """Backward for sigmoid(a). dsigmoid/dx = sigmoid(x) * (1 - sigmoid(x))"""

    def __init__(self):
        super().__init__()
        self.result = None

    def backward(self, grad_output):
        from cakelamp.autograd.tensor import AutogradTensor
        import cakelamp._core as _core
        ones = AutogradTensor(_core.ones(list(self.result.shape)), requires_grad=False)
        return (grad_output * self.result * (ones - self.result),)


class TanhBackward(Function):
    """Backward for tanh(a). dtanh/dx = 1 - tanh(x)^2"""

    def __init__(self):
        super().__init__()
        self.result = None

    def backward(self, grad_output):
        from cakelamp.autograd.tensor import AutogradTensor
        import cakelamp._core as _core
        ones = AutogradTensor(_core.ones(list(self.result.shape)), requires_grad=False)
        return (grad_output * (ones - self.result * self.result),)


class SumBackward(Function):
    """Backward for sum (all elements)."""

    def __init__(self):
        super().__init__()
        self.input_shape = None

    def backward(self, grad_output):
        from cakelamp.autograd.tensor import AutogradTensor
        import cakelamp._core as _core
        grad = AutogradTensor(_core.ones(list(self.input_shape)), requires_grad=False)
        scalar_val = grad_output.item()
        return (grad * AutogradTensor.from_scalar(scalar_val),)


class SumDimBackward(Function):
    """Backward for sum along a dimension."""

    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.dim = None
        self.keepdim = None

    def backward(self, grad_output):
        go = grad_output
        if not self.keepdim:
            go = go.unsqueeze(self.dim)
        return (go.expand(list(self.input_shape)),)


class MeanBackward(Function):
    """Backward for mean (all elements)."""

    def __init__(self):
        super().__init__()
        self.input_shape = None

    def backward(self, grad_output):
        from cakelamp.autograd.tensor import AutogradTensor
        import cakelamp._core as _core
        n = 1
        for s in self.input_shape:
            n *= s
        grad = AutogradTensor(
            _core.full(list(self.input_shape), 1.0 / n), requires_grad=False
        )
        scalar_val = grad_output.item()
        return (grad * AutogradTensor.from_scalar(scalar_val),)


class MeanDimBackward(Function):
    """Backward for mean along a dimension."""

    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.dim = None
        self.keepdim = None

    def backward(self, grad_output):
        from cakelamp.autograd.tensor import AutogradTensor
        go = grad_output
        if not self.keepdim:
            go = go.unsqueeze(self.dim)
        dim_size = self.input_shape[self.dim]
        scale = AutogradTensor.from_scalar(1.0 / dim_size)
        return ((go * scale).expand(list(self.input_shape)),)


class ReshapeBackward(Function):
    """Backward for reshape."""

    def __init__(self):
        super().__init__()
        self.input_shape = None

    def backward(self, grad_output):
        return (grad_output.reshape(list(self.input_shape)),)


class TransposeBackward(Function):
    """Backward for transpose."""

    def __init__(self):
        super().__init__()
        self.dim0 = None
        self.dim1 = None

    def backward(self, grad_output):
        return (grad_output.transpose(self.dim0, self.dim1),)


class ExpandBackward(Function):
    """Backward for expand."""

    def __init__(self):
        super().__init__()
        self.input_shape = None

    def backward(self, grad_output):
        return (_unbroadcast(grad_output, self.input_shape),)


class LogSoftmaxBackward(Function):
    """Backward for log_softmax along a dimension."""

    def __init__(self):
        super().__init__()
        self.result = None
        self.dim = None

    def backward(self, grad_output):
        softmax = self.result.exp()
        grad_sum = grad_output.sum_dim(self.dim, keepdim=True)
        return (grad_output - softmax * grad_sum,)


class SoftmaxBackward(Function):
    """Backward for softmax along a dimension."""

    def __init__(self):
        super().__init__()
        self.result = None
        self.dim = None

    def backward(self, grad_output):
        s = self.result
        sg = s * grad_output
        sg_sum = sg.sum_dim(self.dim, keepdim=True)
        return (s * (grad_output - sg_sum),)


# ── Helper ────────────────────────────────────────────────────────────


def _unbroadcast(grad, target_shape):
    """Sum-reduce gradient to match target_shape after broadcasting."""
    from cakelamp.autograd.tensor import AutogradTensor

    grad_shape = list(grad.shape)
    target_shape = list(target_shape)

    if len(target_shape) == 0:
        return grad.sum()

    # Prepend 1s to match ndim
    while len(target_shape) < len(grad_shape):
        target_shape = [1] + target_shape

    # Sum along broadcast dimensions
    result = grad
    for i in range(len(grad_shape)):
        if target_shape[i] == 1 and grad_shape[i] != 1:
            result = result.sum_dim(i, keepdim=True)

    return result.reshape(list(target_shape))

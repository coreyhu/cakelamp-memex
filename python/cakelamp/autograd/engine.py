"""AutogradTensor: Tensor wrapper with automatic differentiation support."""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import cakelamp._core as _C

if TYPE_CHECKING:
    from cakelamp.autograd.function import Function


class AutogradTensor:
    """A tensor that tracks computation for automatic differentiation.

    Wraps a cakelamp._core.Tensor and adds:
    - requires_grad: whether to track gradients
    - grad: accumulated gradient (another AutogradTensor)
    - grad_fn: the Function that created this tensor (None for leaf tensors)
    """

    def __init__(self, data, requires_grad: bool = False):
        if isinstance(data, _C.Tensor):
            self.data = data
        elif isinstance(data, AutogradTensor):
            self.data = data.data
        elif isinstance(data, (list, tuple)):
            raise ValueError("Use cakelamp.tensor() to create tensors from lists")
        else:
            self.data = data

        self.requires_grad = requires_grad
        self.grad: Optional[AutogradTensor] = None
        self.grad_fn: Optional[Function] = None
        self._is_leaf = True

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    def numel(self):
        return self.data.numel()

    def item(self):
        return self.data.item()

    def tolist(self):
        return self.data.tolist()

    def detach(self) -> AutogradTensor:
        """Return a new tensor detached from the computation graph."""
        t = AutogradTensor(self.data, requires_grad=False)
        return t

    def zero_grad(self):
        """Zero out the gradient."""
        self.grad = None

    # ---- Backward pass ----

    def backward(self, gradient: Optional[AutogradTensor] = None):
        """Compute gradients via reverse-mode autodiff.

        Uses topological sort + reverse traversal of the computation graph.
        """
        if gradient is None:
            # For scalar outputs, gradient is 1.0
            assert self.data.numel() == 1, \
                "backward() requires gradient argument for non-scalar tensors"
            gradient = AutogradTensor(_C.Tensor.scalar(1.0))

        # Topological sort
        topo_order = []
        visited = set()

        def _topo_sort(node):
            if id(node) in visited:
                return
            visited.add(id(node))
            if node.grad_fn is not None:
                for inp in node.grad_fn.inputs:
                    _topo_sort(inp)
            topo_order.append(node)

        _topo_sort(self)

        # Set gradient for the output
        self.grad = gradient

        # Reverse traversal
        for node in reversed(topo_order):
            if node.grad_fn is not None:
                grads = node.grad_fn.backward(node.grad)
                if not isinstance(grads, tuple):
                    grads = (grads,)

                for inp, g in zip(node.grad_fn.inputs, grads):
                    if g is None:
                        continue
                    if not inp.requires_grad:
                        continue

                    # Handle broadcasting: reduce grad to match input shape
                    g = _unbroadcast(g, inp.shape)

                    if inp.grad is None:
                        inp.grad = g
                    else:
                        inp.grad = AutogradTensor(
                            _C.Tensor(
                                (inp.grad.data + g.data).tolist(),
                                inp.grad.data.shape
                            )
                        )

    # ---- Operator overloading (calls autograd functions) ----

    def __add__(self, other):
        from cakelamp.autograd.ops import Add
        if isinstance(other, (int, float)):
            other = AutogradTensor(_C.Tensor.scalar(float(other)))
        return Add.apply(self, other)

    def __radd__(self, other):
        if isinstance(other, (int, float)):
            other = AutogradTensor(_C.Tensor.scalar(float(other)))
        return other.__add__(self)

    def __sub__(self, other):
        from cakelamp.autograd.ops import Sub
        if isinstance(other, (int, float)):
            other = AutogradTensor(_C.Tensor.scalar(float(other)))
        return Sub.apply(self, other)

    def __rsub__(self, other):
        if isinstance(other, (int, float)):
            other = AutogradTensor(_C.Tensor.scalar(float(other)))
        return other.__sub__(self)

    def __mul__(self, other):
        from cakelamp.autograd.ops import Mul
        if isinstance(other, (int, float)):
            other = AutogradTensor(_C.Tensor.scalar(float(other)))
        return Mul.apply(self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from cakelamp.autograd.ops import Div
        if isinstance(other, (int, float)):
            other = AutogradTensor(_C.Tensor.scalar(float(other)))
        return Div.apply(self, other)

    def __neg__(self):
        from cakelamp.autograd.ops import Neg
        return Neg.apply(self)

    def __matmul__(self, other):
        from cakelamp.autograd.ops import MatMul
        return MatMul.apply(self, other)

    def __pow__(self, exponent):
        from cakelamp.autograd.ops import Pow
        if isinstance(exponent, (int, float)):
            exponent = AutogradTensor(_C.Tensor.scalar(float(exponent)))
        return Pow.apply(self, exponent)

    # ---- Methods that produce autograd tensors ----

    def sum(self):
        from cakelamp.autograd.ops import Sum
        return Sum.apply(self)

    def sum_dim(self, dim, keepdim=False):
        from cakelamp.autograd.ops import SumDim
        return SumDim.apply(self, dim, keepdim)

    def mean(self):
        from cakelamp.autograd.ops import Mean
        return Mean.apply(self)

    def exp(self):
        from cakelamp.autograd.ops import Exp
        return Exp.apply(self)

    def log(self):
        from cakelamp.autograd.ops import Log
        return Log.apply(self)

    def relu(self):
        from cakelamp.autograd.ops import ReLU
        return ReLU.apply(self)

    def sigmoid(self):
        from cakelamp.autograd.ops import Sigmoid
        return Sigmoid.apply(self)

    def tanh(self):
        from cakelamp.autograd.ops import Tanh
        return Tanh.apply(self)

    def reshape(self, shape):
        from cakelamp.autograd.ops import Reshape
        return Reshape.apply(self, shape)

    def t(self):
        from cakelamp.autograd.ops import Transpose
        return Transpose.apply(self)

    def transpose(self, dim0, dim1):
        from cakelamp.autograd.ops import TransposeDims
        return TransposeDims.apply(self, dim0, dim1)

    def matmul(self, other):
        return self.__matmul__(other)

    def mm(self, other):
        return self.__matmul__(other)

    def softmax(self, dim=1):
        from cakelamp.autograd.ops import Softmax
        return Softmax.apply(self, dim)

    def log_softmax(self, dim=1):
        from cakelamp.autograd.ops import LogSoftmax
        return LogSoftmax.apply(self, dim)

    def unsqueeze(self, dim):
        from cakelamp.autograd.ops import Unsqueeze
        return Unsqueeze.apply(self, dim)

    def squeeze(self, dim=None):
        from cakelamp.autograd.ops import Squeeze
        return Squeeze.apply(self, dim)

    def expand(self, shape):
        from cakelamp.autograd.ops import Expand
        return Expand.apply(self, shape)

    def contiguous(self):
        return AutogradTensor(self.data.contiguous(), requires_grad=self.requires_grad)

    def argmax(self, dim):
        # argmax is not differentiable, returns detached tensor
        return AutogradTensor(self.data.argmax(dim), requires_grad=False)

    def clamp(self, min_val, max_val):
        from cakelamp.autograd.ops import Clamp
        return Clamp.apply(self, min_val, max_val)

    # ---- Comparison ops (not differentiable) ----

    def eq(self, other):
        if isinstance(other, AutogradTensor):
            return AutogradTensor(self.data.eq(other.data), requires_grad=False)
        other_data = _C.Tensor.scalar(float(other))
        return AutogradTensor(self.data.eq(other_data), requires_grad=False)

    def gt(self, other):
        if isinstance(other, AutogradTensor):
            return AutogradTensor(self.data.gt(other.data), requires_grad=False)
        other_data = _C.Tensor.scalar(float(other))
        return AutogradTensor(self.data.gt(other_data), requires_grad=False)

    # ---- In-place ops for optimizer updates ----

    def add_(self, other):
        """In-place add (for optimizer updates, not autograd-tracked)."""
        if isinstance(other, AutogradTensor):
            self.data = _C.Tensor(
                (self.data + other.data).tolist(), self.data.shape
            )
        else:
            self.data = self.data.add_scalar(float(other))

    def mul_(self, scalar):
        """In-place mul by scalar (for optimizer updates)."""
        self.data.mul_scalar_(float(scalar))

    def sub_(self, other):
        """In-place sub."""
        if isinstance(other, AutogradTensor):
            result = self.data - other.data
            self.data = _C.Tensor(result.tolist(), self.data.shape)

    # ---- Utilities ----

    def zeros_like(self):
        return AutogradTensor(_C.zeros(self.data.shape))

    def ones_like(self):
        return AutogradTensor(_C.ones(self.data.shape))

    def __repr__(self):
        grad_str = f", grad_fn={self.grad_fn.__class__.__name__}" if self.grad_fn else ""
        req_grad = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        return f"AutogradTensor({self.data}{req_grad}{grad_str})"


def _unbroadcast(grad: AutogradTensor, target_shape) -> AutogradTensor:
    """Reduce gradient to match target shape by summing over broadcast dims."""
    grad_shape = grad.data.shape
    target_shape = list(target_shape)

    if grad_shape == target_shape:
        return grad

    # Handle scalar target
    if len(target_shape) == 0:
        s = grad.data.sum()
        return AutogradTensor(s)

    data = grad.data

    # Sum over leading dimensions that were added by broadcasting
    while len(grad_shape) > len(target_shape):
        data = data.sum_dim(0)
        grad_shape = data.shape

    # Sum over dimensions that were broadcast (size 1 in target)
    for i in range(len(target_shape)):
        if target_shape[i] == 1 and grad_shape[i] != 1:
            data = data.sum_dim(i, keepdim=True)

    return AutogradTensor(data)

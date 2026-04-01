"""AutogradTensor: Tensor with automatic differentiation support.

Wraps the Rust-backed cakelamp._core.Tensor and adds:
  - requires_grad: whether to track this tensor in the computation graph
  - grad: accumulated gradient after backward()
  - grad_fn: the Function node that created this tensor
"""

from __future__ import annotations
from typing import Optional, List, Tuple
import cakelamp._core as _core
from cakelamp.autograd import function as F


class AutogradTensor:
    """Tensor with autograd support.

    Wraps a cakelamp._core.Tensor (Rust) and layers Python-level
    computation graph tracking on top.
    """

    def __init__(
        self,
        data: _core.Tensor,
        requires_grad: bool = False,
        grad_fn: Optional[F.Function] = None,
    ):
        self.data: _core.Tensor = data
        self.requires_grad: bool = requires_grad
        self.grad: Optional[AutogradTensor] = None
        self.grad_fn: Optional[F.Function] = grad_fn
        self._is_leaf = grad_fn is None

    # ── Factory methods ──────────────────────────────────────────────

    @staticmethod
    def from_scalar(value: float, requires_grad: bool = False) -> AutogradTensor:
        return AutogradTensor(_core.Tensor.scalar(value), requires_grad=requires_grad)

    @staticmethod
    def from_data(data: list, shape: list, requires_grad: bool = False) -> AutogradTensor:
        return AutogradTensor(_core.tensor(data, shape), requires_grad=requires_grad)

    @staticmethod
    def zeros(shape: list, requires_grad: bool = False) -> AutogradTensor:
        return AutogradTensor(_core.zeros(shape), requires_grad=requires_grad)

    @staticmethod
    def ones(shape: list, requires_grad: bool = False) -> AutogradTensor:
        return AutogradTensor(_core.ones(shape), requires_grad=requires_grad)

    @staticmethod
    def rand(shape: list, requires_grad: bool = False) -> AutogradTensor:
        return AutogradTensor(_core.rand(shape), requires_grad=requires_grad)

    @staticmethod
    def randn(shape: list, requires_grad: bool = False) -> AutogradTensor:
        return AutogradTensor(_core.randn(shape), requires_grad=requires_grad)

    @staticmethod
    def full(shape: list, value: float, requires_grad: bool = False) -> AutogradTensor:
        return AutogradTensor(_core.full(shape, value), requires_grad=requires_grad)

    # ── Properties ───────────────────────────────────────────────────

    @property
    def shape(self) -> list:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def is_leaf(self) -> bool:
        return self._is_leaf

    def numel(self) -> int:
        return self.data.numel()

    def item(self) -> float:
        return self.data.item()

    def tolist(self) -> list:
        return self.data.tolist()

    def detach(self) -> AutogradTensor:
        return AutogradTensor(self.data, requires_grad=False)

    def clone(self) -> AutogradTensor:
        new_data = _core.tensor(self.data.tolist(), list(self.shape))
        return AutogradTensor(new_data, requires_grad=self.requires_grad)

    # ── Backward ─────────────────────────────────────────────────────

    def backward(self, grad_output: Optional[AutogradTensor] = None):
        """Run reverse-mode automatic differentiation."""
        if grad_output is None:
            if self.numel() != 1:
                raise RuntimeError("backward() requires grad_output for non-scalar tensors")
            grad_output = AutogradTensor(_core.ones(list(self.shape)), requires_grad=False)

        topo_order = _topological_sort(self)
        grads = {id(self): grad_output}

        for node in topo_order:
            if node.grad_fn is None:
                continue
            grad = grads.get(id(node))
            if grad is None:
                continue

            input_grads = node.grad_fn.backward(grad)

            for inp, ig in zip(node.grad_fn.inputs, input_grads):
                if ig is None:
                    continue
                nid = id(inp)
                if nid in grads:
                    old = grads[nid]
                    grads[nid] = AutogradTensor(old.data + ig.data, requires_grad=False)
                else:
                    grads[nid] = ig

        for node in topo_order:
            if node._is_leaf and node.requires_grad:
                g = grads.get(id(node))
                if g is not None:
                    if node.grad is None:
                        node.grad = g.detach()
                    else:
                        node.grad.data.add_(g.data)

    def zero_grad(self):
        self.grad = None

    # ── Graph node creation helper ────────────────────────────────────

    def _make_result(self, result_data, grad_fn_cls, inputs, **extra):
        requires_grad = any(getattr(inp, "requires_grad", False) for inp in inputs)
        grad_fn = None
        if requires_grad:
            grad_fn = grad_fn_cls()
            grad_fn.inputs = list(inputs)
            grad_fn.save_for_backward(*inputs)
            for k, v in extra.items():
                setattr(grad_fn, k, v)
        return AutogradTensor(result_data, requires_grad=requires_grad, grad_fn=grad_fn)

    # ── Arithmetic operators ─────────────────────────────────────────

    def __add__(self, other) -> AutogradTensor:
        other = _ensure_tensor(other)
        return self._make_result(self.data + other.data, F.AddBackward, [self, other])

    def __radd__(self, other) -> AutogradTensor:
        return self.__add__(_ensure_tensor(other))

    def __sub__(self, other) -> AutogradTensor:
        other = _ensure_tensor(other)
        return self._make_result(self.data - other.data, F.SubBackward, [self, other])

    def __rsub__(self, other) -> AutogradTensor:
        other = _ensure_tensor(other)
        return other.__sub__(self)

    def __mul__(self, other) -> AutogradTensor:
        other = _ensure_tensor(other)
        return self._make_result(self.data * other.data, F.MulBackward, [self, other])

    def __rmul__(self, other) -> AutogradTensor:
        return self.__mul__(_ensure_tensor(other))

    def __truediv__(self, other) -> AutogradTensor:
        other = _ensure_tensor(other)
        return self._make_result(self.data / other.data, F.DivBackward, [self, other])

    def __neg__(self) -> AutogradTensor:
        return self._make_result(-self.data, F.NegBackward, [self])

    def __matmul__(self, other) -> AutogradTensor:
        return self.matmul(other)

    # ── Unary operations ─────────────────────────────────────────────

    def exp(self) -> AutogradTensor:
        result = self._make_result(self.data.exp(), F.ExpBackward, [self])
        if result.grad_fn:
            result.grad_fn.result = result
        return result

    def log(self) -> AutogradTensor:
        return self._make_result(self.data.log(), F.LogBackward, [self])

    def relu(self) -> AutogradTensor:
        return self._make_result(self.data.relu(), F.ReluBackward, [self])

    def sigmoid(self) -> AutogradTensor:
        result = self._make_result(self.data.sigmoid(), F.SigmoidBackward, [self])
        if result.grad_fn:
            result.grad_fn.result = result
        return result

    def tanh(self) -> AutogradTensor:
        result = self._make_result(self.data.tanh(), F.TanhBackward, [self])
        if result.grad_fn:
            result.grad_fn.result = result
        return result

    def matmul(self, other) -> AutogradTensor:
        other = _ensure_tensor(other)
        return self._make_result(self.data.matmul(other.data), F.MatmulBackward, [self, other])

    def softmax(self, dim: int = 1) -> AutogradTensor:
        result = self._make_result(self.data.softmax(dim), F.SoftmaxBackward, [self], dim=dim)
        if result.grad_fn:
            result.grad_fn.result = result
        return result

    def log_softmax(self, dim: int = 1) -> AutogradTensor:
        result = self._make_result(self.data.log_softmax(dim), F.LogSoftmaxBackward, [self], dim=dim)
        if result.grad_fn:
            result.grad_fn.result = result
        return result

    # ── Reduction operations ─────────────────────────────────────────

    def sum(self) -> AutogradTensor:
        return self._make_result(
            self.data.sum(), F.SumBackward, [self], input_shape=tuple(self.shape)
        )

    def sum_dim(self, dim: int, keepdim: bool = False) -> AutogradTensor:
        return self._make_result(
            self.data.sum_dim(dim, keepdim), F.SumDimBackward, [self],
            input_shape=tuple(self.shape), dim=dim, keepdim=keepdim
        )

    def mean(self) -> AutogradTensor:
        return self._make_result(
            self.data.mean(), F.MeanBackward, [self], input_shape=tuple(self.shape)
        )

    def mean_dim(self, dim: int, keepdim: bool = False) -> AutogradTensor:
        return self._make_result(
            self.data.mean_dim(dim, keepdim), F.MeanDimBackward, [self],
            input_shape=tuple(self.shape), dim=dim, keepdim=keepdim
        )

    # ── View operations ──────────────────────────────────────────────

    def reshape(self, shape: list) -> AutogradTensor:
        return self._make_result(
            self.data.reshape(shape), F.ReshapeBackward, [self],
            input_shape=tuple(self.shape)
        )

    def transpose(self, dim0: int, dim1: int) -> AutogradTensor:
        return self._make_result(
            self.data.transpose(dim0, dim1), F.TransposeBackward, [self],
            dim0=dim0, dim1=dim1
        )

    def t(self) -> AutogradTensor:
        return self.transpose(0, 1)

    def unsqueeze(self, dim: int) -> AutogradTensor:
        return AutogradTensor(self.data.unsqueeze(dim), requires_grad=self.requires_grad, grad_fn=self.grad_fn)

    def squeeze(self, dim: Optional[int] = None) -> AutogradTensor:
        return AutogradTensor(self.data.squeeze(dim), requires_grad=self.requires_grad, grad_fn=self.grad_fn)

    def expand(self, shape: list) -> AutogradTensor:
        return self._make_result(
            self.data.expand(shape), F.ExpandBackward, [self],
            input_shape=tuple(self.shape)
        )

    def contiguous(self) -> AutogradTensor:
        return AutogradTensor(self.data.contiguous(), requires_grad=self.requires_grad, grad_fn=self.grad_fn)

    # ── Comparison / utility (not differentiable) ─────────────────────

    def argmax(self, dim: int) -> AutogradTensor:
        return AutogradTensor(self.data.argmax(dim), requires_grad=False)

    def eq(self, other) -> AutogradTensor:
        other = _ensure_tensor(other)
        return AutogradTensor(self.data.eq(other.data), requires_grad=False)

    def gt(self, other) -> AutogradTensor:
        other = _ensure_tensor(other)
        return AutogradTensor(self.data.gt(other.data), requires_grad=False)

    def max(self) -> AutogradTensor:
        return AutogradTensor(self.data.max(), requires_grad=False)

    def min(self) -> AutogradTensor:
        return AutogradTensor(self.data.min(), requires_grad=False)

    # ── In-place operations (for optimizer use) ───────────────────────

    def add_(self, other):
        if isinstance(other, AutogradTensor):
            self.data.add_(other.data)
        else:
            self.data.add_(other)
        return self

    def mul_scalar_(self, scalar: float):
        self.data.mul_scalar_(scalar)
        return self

    def fill_(self, value: float):
        self.data.fill_(value)
        return self

    def sub_alpha_(self, other, alpha: float):
        if isinstance(other, AutogradTensor):
            self.data.sub_alpha_(other.data, alpha)
        else:
            self.data.sub_alpha_(other, alpha)
        return self

    def copy_from(self, src):
        if isinstance(src, AutogradTensor):
            self.data.copy_from(src.data)
        else:
            self.data.copy_from(src)
        return self

    # ── Scalar ops ────────────────────────────────────────────────────

    def add_scalar(self, s: float) -> AutogradTensor:
        return AutogradTensor(self.data.add_scalar(s), requires_grad=self.requires_grad)

    def mul_scalar(self, s: float) -> AutogradTensor:
        return AutogradTensor(self.data.mul_scalar(s), requires_grad=self.requires_grad)

    def get(self, indices: list) -> float:
        return self.data.get(indices)

    def set(self, indices: list, value: float):
        self.data.set(indices, value)

    # ── Repr ─────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"AutogradTensor(shape={self.shape}, "
            f"requires_grad={self.requires_grad}, "
            f"grad_fn={type(self.grad_fn).__name__ if self.grad_fn else None})"
        )


# ── Helpers ───────────────────────────────────────────────────────────


def _ensure_tensor(x) -> AutogradTensor:
    if isinstance(x, AutogradTensor):
        return x
    if isinstance(x, (int, float)):
        return AutogradTensor.from_scalar(float(x))
    if isinstance(x, _core.Tensor):
        return AutogradTensor(x, requires_grad=False)
    raise TypeError(f"Cannot convert {type(x)} to AutogradTensor")


def _topological_sort(root: AutogradTensor) -> list:
    """Topological sort of the computation graph (DFS post-order)."""
    visited = set()
    order = []

    def _visit(node: AutogradTensor):
        node_id = id(node)
        if node_id in visited:
            return
        visited.add(node_id)
        if node.grad_fn is not None:
            for inp in node.grad_fn.inputs:
                _visit(inp)
        order.append(node)

    _visit(root)
    return list(reversed(order))

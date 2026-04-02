"""AutogradTensor: A tensor with automatic differentiation support.

This is a pure Python tensor that provides the computation backend
needed for autograd. It mirrors the Rust/PyO3 Tensor API but adds
requires_grad, grad, and grad_fn for reverse-mode autodiff.
"""

import math
import random
import builtins

from cakelamp.autograd.engine import GradMode


class AutogradTensor:
    """Tensor with autograd support.

    Stores data as a flat list of floats with shape/stride metadata.
    Tracks computation graph via grad_fn for backward().
    """

    __slots__ = ('_data', '_shape', '_strides', '_offset',
                 'requires_grad', 'grad', '_grad_fn', '_version')

    def __init__(self, data, requires_grad=False):
        if isinstance(data, (list, tuple)):
            flat, shape = _flatten(data)
            self._data = [float(x) for x in flat]
            self._shape = list(shape)
        elif isinstance(data, (int, float)):
            self._data = [float(data)]
            self._shape = []
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")
        self._strides = _compute_strides(self._shape)
        self._offset = 0
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        self._version = 0

    @staticmethod
    def _make(data, shape, requires_grad=False):
        t = AutogradTensor.__new__(AutogradTensor)
        t._data = data
        t._shape = list(shape)
        t._strides = _compute_strides(shape)
        t._offset = 0
        t.requires_grad = requires_grad
        t.grad = None
        t._grad_fn = None
        t._version = 0
        return t

    # ---- Constructors ----

    @staticmethod
    def zeros(shape, requires_grad=False):
        n = _numel(shape)
        return AutogradTensor._make([0.0] * n, shape, requires_grad)

    @staticmethod
    def ones(shape, requires_grad=False):
        n = _numel(shape)
        return AutogradTensor._make([1.0] * n, shape, requires_grad)

    @staticmethod
    def ones_like(tensor):
        return AutogradTensor.ones(list(tensor._shape))

    @staticmethod
    def zeros_like(tensor):
        return AutogradTensor.zeros(list(tensor._shape))

    @staticmethod
    def rand(shape, requires_grad=False):
        n = _numel(shape)
        return AutogradTensor._make([random.random() for _ in range(n)], shape, requires_grad)

    @staticmethod
    def randn(shape, requires_grad=False):
        n = _numel(shape)
        data = []
        for _ in range(n):
            u1 = max(random.random(), 1e-10)
            u2 = random.random()
            data.append(math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2))
        return AutogradTensor._make(data, shape, requires_grad)

    @staticmethod
    def full(shape, value, requires_grad=False):
        n = _numel(shape)
        return AutogradTensor._make([float(value)] * n, shape, requires_grad)

    @staticmethod
    def eye(n, requires_grad=False):
        data = [0.0] * (n * n)
        for i in range(n):
            data[i * n + i] = 1.0
        return AutogradTensor._make(data, [n, n], requires_grad)

    @staticmethod
    def arange(start, end, step=1.0):
        data = []
        v = float(start)
        end = float(end)
        step = float(step)
        if step > 0:
            while v < end:
                data.append(v)
                v += step
        elif step < 0:
            while v > end:
                data.append(v)
                v += step
        return AutogradTensor._make(data, [len(data)])

    # ---- Properties ----

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def numel(self):
        return _numel(self._shape)

    @property
    def data(self):
        t = AutogradTensor._make(list(self._data), list(self._shape))
        return t

    @data.setter
    def data(self, value):
        if isinstance(value, AutogradTensor):
            self._data = list(value._data)
            self._shape = list(value._shape)
            self._strides = _compute_strides(self._shape)
            self._offset = 0
            self._version += 1
        else:
            raise TypeError("data must be an AutogradTensor")

    def item(self):
        assert self.numel == 1, f"item() requires single-element tensor, got {self.numel}"
        if not self._shape:
            return self._data[self._offset]
        return self._data[0]

    def detach(self):
        t = AutogradTensor._make(list(self._data), list(self._shape))
        return t

    def clone(self):
        t = AutogradTensor._make(list(self._data), list(self._shape), self.requires_grad)
        return t

    def tolist(self):
        if not self._shape:
            return self._data[0] if self._data else 0.0
        def _build(data, shape, offset):
            if len(shape) == 1:
                return [data[offset + i] for i in range(shape[0])]
            stride = _numel(shape[1:])
            return [_build(data, shape[1:], offset + i * stride) for i in range(shape[0])]
        return _build(self._data, self._shape, 0)

    # ---- Shape ops ----

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = list(shape[0])
        else:
            shape = list(shape)
        # Handle -1
        neg_idx = None
        known = 1
        for i, s in enumerate(shape):
            if s == -1:
                neg_idx = i
            else:
                known *= s
        if neg_idx is not None:
            shape[neg_idx] = self.numel // known
        assert _numel(shape) == self.numel
        from cakelamp.autograd.function import ReshapeBackward
        result = AutogradTensor._make(list(self._data), shape, self.requires_grad)
        if self.requires_grad and GradMode.is_enabled():
            result._grad_fn = ReshapeBackward(self)
        return result

    def view(self, *shape):
        return self.reshape(*shape)

    def transpose(self, dim0, dim1):
        from cakelamp.autograd.function import TransposeBackward
        new_shape = list(self._shape)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        # Need to physically rearrange data
        ndim = len(self._shape)
        n = self.numel
        old_strides = _compute_strides(self._shape)
        new_strides_logical = list(old_strides)
        new_strides_logical[dim0], new_strides_logical[dim1] = new_strides_logical[dim1], new_strides_logical[dim0]
        # Materialize transposed data
        new_data = [0.0] * n
        coord = [0] * ndim
        for i in range(n):
            old_idx = sum(coord[d] * old_strides[d] for d in range(ndim))
            new_coord = list(coord)
            new_coord[dim0], new_coord[dim1] = new_coord[dim1], new_coord[dim0]
            new_idx = sum(new_coord[d] * _compute_strides(new_shape)[d] for d in range(ndim))
            new_data[new_idx] = self._data[old_idx]
            for d in range(ndim - 1, -1, -1):
                coord[d] += 1
                if coord[d] < self._shape[d]:
                    break
                coord[d] = 0

        result = AutogradTensor._make(new_data, new_shape, self.requires_grad)
        if self.requires_grad and GradMode.is_enabled():
            result._grad_fn = TransposeBackward(self, dim0, dim1)
        return result

    @property
    def T(self):
        assert self.ndim == 2
        return self.transpose(0, 1)

    def unsqueeze(self, dim):
        new_shape = list(self._shape)
        if dim < 0:
            dim = self.ndim + 1 + dim
        new_shape.insert(dim, 1)
        return AutogradTensor._make(list(self._data), new_shape, self.requires_grad)

    def squeeze(self, dim=None):
        new_shape = [s for i, s in enumerate(self._shape)
                     if not ((dim is None and s == 1) or (dim is not None and i == dim and s == 1))]
        return AutogradTensor._make(list(self._data), new_shape, self.requires_grad)

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = list(shape[0])
        else:
            shape = list(shape)
        # Broadcast self._data to target shape
        target = shape
        src_shape = list(self._shape)
        # Pad src_shape on the left
        while len(src_shape) < len(target):
            src_shape = [1] + src_shape

        n = _numel(target)
        src_strides = _compute_strides(src_shape)
        # Zero out strides where src dim is 1 and target dim > 1
        bc_strides = list(src_strides)
        for i in range(len(target)):
            if src_shape[i] == 1 and target[i] > 1:
                bc_strides[i] = 0

        result = [0.0] * n
        ndim = len(target)
        coord = [0] * ndim
        for flat_i in range(n):
            src_idx = sum(coord[d] * bc_strides[d] for d in range(ndim))
            result[flat_i] = self._data[src_idx]
            for d in range(ndim - 1, -1, -1):
                coord[d] += 1
                if coord[d] < target[d]:
                    break
                coord[d] = 0
        return AutogradTensor._make(result, target, self.requires_grad)

    # ---- Element-wise ops ----

    def __add__(self, other):
        from cakelamp.autograd.function import AddBackward
        other = _ensure_tensor(other)
        data, shape = _binary_op(self._data, self._shape, other._data, other._shape, lambda a, b: a + b)
        rg = self.requires_grad or other.requires_grad
        result = AutogradTensor._make(data, shape, rg)
        if rg and GradMode.is_enabled():
            result._grad_fn = AddBackward(self, other)
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from cakelamp.autograd.function import SubBackward
        other = _ensure_tensor(other)
        data, shape = _binary_op(self._data, self._shape, other._data, other._shape, lambda a, b: a - b)
        rg = self.requires_grad or other.requires_grad
        result = AutogradTensor._make(data, shape, rg)
        if rg and GradMode.is_enabled():
            result._grad_fn = SubBackward(self, other)
        return result

    def __rsub__(self, other):
        return _ensure_tensor(other).__sub__(self)

    def __mul__(self, other):
        from cakelamp.autograd.function import MulBackward
        other = _ensure_tensor(other)
        data, shape = _binary_op(self._data, self._shape, other._data, other._shape, lambda a, b: a * b)
        rg = self.requires_grad or other.requires_grad
        result = AutogradTensor._make(data, shape, rg)
        if rg and GradMode.is_enabled():
            result._grad_fn = MulBackward(self, other)
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from cakelamp.autograd.function import DivBackward
        other = _ensure_tensor(other)
        data, shape = _binary_op(self._data, self._shape, other._data, other._shape, lambda a, b: a / b)
        rg = self.requires_grad or other.requires_grad
        result = AutogradTensor._make(data, shape, rg)
        if rg and GradMode.is_enabled():
            result._grad_fn = DivBackward(self, other)
        return result

    def __rtruediv__(self, other):
        return _ensure_tensor(other).__truediv__(self)

    def __neg__(self):
        from cakelamp.autograd.function import NegBackward
        data = [-x for x in self._data]
        result = AutogradTensor._make(data, list(self._shape), self.requires_grad)
        if self.requires_grad and GradMode.is_enabled():
            result._grad_fn = NegBackward(self)
        return result

    def __pow__(self, other):
        from cakelamp.autograd.function import PowBackward
        other = _ensure_tensor(other)
        data, shape = _binary_op(self._data, self._shape, other._data, other._shape, lambda a, b: a ** b)
        rg = self.requires_grad or other.requires_grad
        result = AutogradTensor._make(data, shape, rg)
        if rg and GradMode.is_enabled():
            result._grad_fn = PowBackward(self, other)
        return result

    def exp(self):
        from cakelamp.autograd.function import ExpBackward
        data = [math.exp(x) for x in self._data]
        result = AutogradTensor._make(data, list(self._shape), self.requires_grad)
        if self.requires_grad and GradMode.is_enabled():
            result._grad_fn = ExpBackward(self, result)
        return result

    def log(self):
        from cakelamp.autograd.function import LogBackward
        data = [math.log(x) for x in self._data]
        result = AutogradTensor._make(data, list(self._shape), self.requires_grad)
        if self.requires_grad and GradMode.is_enabled():
            result._grad_fn = LogBackward(self)
        return result

    def relu(self):
        from cakelamp.autograd.function import ReluBackward
        data = [builtins.max(0.0, x) for x in self._data]
        result = AutogradTensor._make(data, list(self._shape), self.requires_grad)
        if self.requires_grad and GradMode.is_enabled():
            result._grad_fn = ReluBackward(self)
        return result

    def sigmoid(self):
        from cakelamp.autograd.function import SigmoidBackward
        data = [1.0 / (1.0 + math.exp(-x)) for x in self._data]
        result = AutogradTensor._make(data, list(self._shape), self.requires_grad)
        if self.requires_grad and GradMode.is_enabled():
            result._grad_fn = SigmoidBackward(self, result)
        return result

    def tanh(self):
        from cakelamp.autograd.function import TanhBackward
        data = [math.tanh(x) for x in self._data]
        result = AutogradTensor._make(data, list(self._shape), self.requires_grad)
        if self.requires_grad and GradMode.is_enabled():
            result._grad_fn = TanhBackward(self, result)
        return result

    def sqrt(self):
        return self ** 0.5

    def clamp(self, min_val=None, max_val=None):
        def _c(x):
            if min_val is not None: x = builtins.max(x, min_val)
            if max_val is not None: x = builtins.min(x, max_val)
            return x
        return AutogradTensor._make([_c(x) for x in self._data], list(self._shape), self.requires_grad)

    # ---- Reduction ops ----

    def sum(self, dim=None, keepdim=False):
        from cakelamp.autograd.function import SumBackward
        if dim is None:
            s = builtins.sum(self._data)
            result = AutogradTensor._make([s], [], self.requires_grad)
        else:
            data, shape = _sum_dim(self._data, self._shape, dim, keepdim)
            result = AutogradTensor._make(data, shape, self.requires_grad)
        if self.requires_grad and GradMode.is_enabled():
            result._grad_fn = SumBackward(self, dim, keepdim)
        return result

    def mean(self, dim=None, keepdim=False):
        from cakelamp.autograd.function import MeanBackward
        if dim is None:
            s = builtins.sum(self._data) / len(self._data)
            result = AutogradTensor._make([s], [], self.requires_grad)
        else:
            data, shape = _sum_dim(self._data, self._shape, dim, keepdim)
            n = self._shape[dim]
            data = [x / n for x in data]
            result = AutogradTensor._make(data, shape, self.requires_grad)
        if self.requires_grad and GradMode.is_enabled():
            result._grad_fn = MeanBackward(self, dim, keepdim)
        return result

    def argmax(self, dim=None):
        if dim is None:
            best = 0
            for i in range(1, len(self._data)):
                if self._data[i] > self._data[best]:
                    best = i
            return AutogradTensor._make([float(best)], [])
        return _argmax_dim(self._data, self._shape, dim)

    # ---- Matrix ops ----

    def mm(self, other):
        from cakelamp.autograd.function import MatmulBackward
        assert self.ndim == 2 and other.ndim == 2
        m, k = self._shape
        k2, n = other._shape
        assert k == k2
        result_data = [0.0] * (m * n)
        for i in range(m):
            for p in range(k):
                a_val = self._data[i * k + p]
                for j in range(n):
                    result_data[i * n + j] += a_val * other._data[p * n + j]
        rg = self.requires_grad or other.requires_grad
        result = AutogradTensor._make(result_data, [m, n], rg)
        if rg and GradMode.is_enabled():
            result._grad_fn = MatmulBackward(self, other)
        return result

    def matmul(self, other):
        return self.mm(other)

    def __matmul__(self, other):
        return self.mm(other)

    # ---- Softmax ----

    def softmax(self, dim=-1):
        if dim < 0:
            dim = self.ndim + dim
        data, shape = _softmax(self._data, self._shape, dim)
        return AutogradTensor._make(data, shape, self.requires_grad)

    def log_softmax(self, dim=-1):
        from cakelamp.autograd.function import LogSoftmaxBackward
        if dim < 0:
            dim = self.ndim + dim
        sm_data, sm_shape = _softmax(self._data, self._shape, dim)
        log_data = [math.log(builtins.max(x, 1e-12)) for x in sm_data]
        result = AutogradTensor._make(log_data, sm_shape, self.requires_grad)
        if self.requires_grad and GradMode.is_enabled():
            sm = AutogradTensor._make(sm_data, sm_shape)
            result._grad_fn = LogSoftmaxBackward(self, sm, dim)
        return result

    # ---- In-place ops ----

    def add_(self, other):
        other = _ensure_tensor(other)
        result = self + other
        self._data = list(result._data)
        self._shape = list(result._shape)
        self._strides = _compute_strides(self._shape)
        self._version += 1
        return self

    def mul_(self, other):
        other = _ensure_tensor(other)
        result = self * other
        self._data = list(result._data)
        self._shape = list(result._shape)
        self._strides = _compute_strides(self._shape)
        self._version += 1
        return self

    def zero_(self):
        self._data = [0.0] * self.numel
        self._version += 1
        return self

    def fill_(self, val):
        self._data = [float(val)] * self.numel
        self._version += 1
        return self

    # ---- Backward ----

    def backward(self, gradient=None):
        from cakelamp.autograd.engine import backward
        backward(self, gradient)

    # ---- Comparison ----

    def __eq__(self, other):
        other = _ensure_tensor(other)
        data, shape = _binary_op(self._data, self._shape, other._data, other._shape,
                                 lambda a, b: 1.0 if abs(a - b) < 1e-7 else 0.0)
        return AutogradTensor._make(data, shape)

    def __gt__(self, other):
        other = _ensure_tensor(other)
        data, shape = _binary_op(self._data, self._shape, other._data, other._shape,
                                 lambda a, b: 1.0 if a > b else 0.0)
        return AutogradTensor._make(data, shape)

    # ---- Repr ----

    def __repr__(self):
        if not self._shape:
            return f"tensor({self._data[0]:.4f})"
        if self.numel <= 20:
            return f"tensor({self.tolist()}, shape={self.shape})"
        return f"tensor(shape={self.shape})"

    def __len__(self):
        if not self._shape:
            raise TypeError("len() of 0-d tensor")
        return self._shape[0]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if idx < 0:
                idx += self._shape[0]
            new_shape = self._shape[1:]
            n = _numel(new_shape) if new_shape else 1
            start = idx * n
            new_data = self._data[start:start + n]
            return AutogradTensor._make(new_data, new_shape, self.requires_grad)
        raise NotImplementedError("Only int indexing supported")


# ---- Utility functions ----

def _ensure_tensor(x):
    if isinstance(x, AutogradTensor):
        return x
    if isinstance(x, (int, float)):
        return AutogradTensor._make([float(x)], [])
    raise TypeError(f"Cannot convert {type(x)} to AutogradTensor")


def _flatten(data):
    if not isinstance(data, (list, tuple)):
        return [data], ()
    if len(data) == 0:
        return [], (0,)
    sub_results = []
    sub_shape = None
    for item in data:
        flat, shape = _flatten(item)
        if sub_shape is not None and shape != sub_shape:
            raise ValueError("Inconsistent shapes")
        sub_shape = shape
        sub_results.extend(flat)
    return sub_results, (len(data),) + sub_shape


def _compute_strides(shape):
    if not shape:
        return []
    ndim = len(shape)
    strides = [1] * ndim
    for i in range(ndim - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


def _numel(shape):
    r = 1
    for s in shape:
        r *= s
    return r


def _broadcast_shape(a, b):
    ndim = builtins.max(len(a), len(b))
    result = [0] * ndim
    for i in range(ndim):
        da = a[i - ndim + len(a)] if i >= ndim - len(a) else 1
        db = b[i - ndim + len(b)] if i >= ndim - len(b) else 1
        if da == db:
            result[i] = da
        elif da == 1:
            result[i] = db
        elif db == 1:
            result[i] = da
        else:
            raise ValueError(f"Cannot broadcast {a} and {b}")
    return result


def _broadcast_strides(shape, strides, target):
    ndim = len(target)
    pad = ndim - len(shape)
    result = [0] * ndim
    for i in range(len(shape)):
        ni = i + pad
        if shape[i] == target[ni]:
            result[ni] = strides[i]
        elif shape[i] == 1:
            result[ni] = 0
        else:
            raise ValueError(f"Cannot broadcast dim {i}")
    return result


def _binary_op(a_data, a_shape, b_data, b_shape, op):
    out_shape = _broadcast_shape(a_shape, b_shape)
    a_strides = _broadcast_strides(a_shape, _compute_strides(a_shape), out_shape)
    b_strides = _broadcast_strides(b_shape, _compute_strides(b_shape), out_shape)
    n = _numel(out_shape)
    ndim = len(out_shape)
    result = [0.0] * n
    if ndim == 0:
        return [op(a_data[0], b_data[0])], []
    coord = [0] * ndim
    for fi in range(n):
        ai = builtins.sum(coord[d] * a_strides[d] for d in range(ndim))
        bi = builtins.sum(coord[d] * b_strides[d] for d in range(ndim))
        result[fi] = op(a_data[ai], b_data[bi])
        for d in range(ndim - 1, -1, -1):
            coord[d] += 1
            if coord[d] < out_shape[d]:
                break
            coord[d] = 0
    return result, out_shape


def _sum_dim(data, shape, dim, keepdim):
    ndim = len(shape)
    dim_size = shape[dim]
    out_shape = list(shape)
    out_shape[dim] = 1
    out_n = _numel(out_shape)
    result = [0.0] * out_n
    in_strides = _compute_strides(shape)
    out_strides = _compute_strides(out_shape)

    coord = [0] * ndim
    for _ in range(out_n):
        out_idx = builtins.sum(coord[d] * out_strides[d] for d in range(ndim))
        s = 0.0
        for k in range(dim_size):
            in_idx = builtins.sum(
                (k if d == dim else coord[d]) * in_strides[d]
                for d in range(ndim)
            )
            s += data[in_idx]
        result[out_idx] = s
        for d in range(ndim - 1, -1, -1):
            if d == dim:
                continue
            coord[d] += 1
            if coord[d] < out_shape[d]:
                break
            coord[d] = 0

    if keepdim:
        return result, out_shape
    else:
        squeezed = [s for i, s in enumerate(out_shape) if i != dim]
        if not squeezed:
            return result, []
        return result, squeezed


def _argmax_dim(data, shape, dim):
    ndim = len(shape)
    dim_size = shape[dim]
    out_shape = [s for i, s in enumerate(shape) if i != dim]
    if not out_shape:
        best = 0
        for i in range(1, len(data)):
            if data[i] > data[best]:
                best = i
        return AutogradTensor._make([float(best)], [])
    out_n = _numel(out_shape)
    result = [0.0] * out_n
    in_strides = _compute_strides(shape)
    coord = [0] * len(out_shape)
    for oi in range(out_n):
        best_val = float('-inf')
        best_idx = 0
        for k in range(dim_size):
            in_idx = 0
            od = 0
            for d in range(ndim):
                if d == dim:
                    in_idx += k * in_strides[d]
                else:
                    in_idx += coord[od] * in_strides[d]
                    od += 1
            if data[in_idx] > best_val:
                best_val = data[in_idx]
                best_idx = k
        result[oi] = float(best_idx)
        for d in range(len(out_shape) - 1, -1, -1):
            coord[d] += 1
            if coord[d] < out_shape[d]:
                break
            coord[d] = 0
    return AutogradTensor._make(result, out_shape)


def _softmax(data, shape, dim):
    ndim = len(shape)
    n = _numel(shape)
    strides = _compute_strides(shape)
    out_shape = list(shape)
    out_shape[dim] = 1
    out_n = _numel(out_shape)
    out_strides = _compute_strides(out_shape)

    maxes = [float('-inf')] * out_n
    coord = [0] * ndim
    for _ in range(n):
        in_idx = builtins.sum(coord[d] * strides[d] for d in range(ndim))
        out_idx = builtins.sum(coord[d] * out_strides[d] for d in range(ndim) if d != dim)
        if data[in_idx] > maxes[out_idx]:
            maxes[out_idx] = data[in_idx]
        for d in range(ndim - 1, -1, -1):
            coord[d] += 1
            if coord[d] < shape[d]: break
            coord[d] = 0

    exp_sums = [0.0] * out_n
    result = [0.0] * n
    coord = [0] * ndim
    for fi in range(n):
        in_idx = builtins.sum(coord[d] * strides[d] for d in range(ndim))
        out_idx = builtins.sum(coord[d] * out_strides[d] for d in range(ndim) if d != dim)
        e = math.exp(data[in_idx] - maxes[out_idx])
        result[fi] = e
        exp_sums[out_idx] += e
        for d in range(ndim - 1, -1, -1):
            coord[d] += 1
            if coord[d] < shape[d]: break
            coord[d] = 0

    coord = [0] * ndim
    for fi in range(n):
        out_idx = builtins.sum(coord[d] * out_strides[d] for d in range(ndim) if d != dim)
        result[fi] /= exp_sums[out_idx]
        for d in range(ndim - 1, -1, -1):
            coord[d] += 1
            if coord[d] < shape[d]: break
            coord[d] = 0

    return result, list(shape)

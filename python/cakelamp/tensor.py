"""Tensor class with autograd support."""

import math
from cakelamp import backend as B


class Tensor:
    """Multi-dimensional tensor with automatic differentiation support.

    Attributes:
        data: flat list of f32 values
        shape: tuple of dimension sizes
        strides: tuple of strides
        requires_grad: whether to track gradients
        grad: accumulated gradient (Tensor or None)
        grad_fn: Function that created this tensor (for autograd graph)
    """

    def __init__(self, data, requires_grad=False, _shape=None, _strides=None, _offset=0):
        if isinstance(data, (list, tuple)):
            flat, shape = self._flatten(data)
            self._data = [float(x) for x in flat]
            self._shape = list(shape) if _shape is None else list(_shape)
        elif isinstance(data, (int, float)):
            self._data = [float(data)]
            self._shape = [] if _shape is None else list(_shape)
        else:
            raise TypeError(f"Unsupported data type: {type(data)}")

        if _strides is not None:
            self._strides = list(_strides)
        else:
            self._strides = B.compute_strides(self._shape)

        self._offset = _offset
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self._version = 0  # For in-place op tracking

    @staticmethod
    def _flatten(data):
        """Recursively flatten nested lists and infer shape."""
        if not isinstance(data, (list, tuple)):
            return [data], ()

        if len(data) == 0:
            return [], (0,)

        sub_results = []
        sub_shape = None
        for item in data:
            flat, shape = Tensor._flatten(item)
            if sub_shape is not None and shape != sub_shape:
                raise ValueError("Inconsistent shapes in nested list")
            sub_shape = shape
            sub_results.extend(flat)

        return sub_results, (len(data),) + sub_shape

    @staticmethod
    def _make(data, shape, requires_grad=False):
        """Internal constructor from flat data and shape."""
        t = Tensor.__new__(Tensor)
        t._data = data
        t._shape = list(shape)
        t._strides = B.compute_strides(shape)
        t._offset = 0
        t.requires_grad = requires_grad
        t.grad = None
        t.grad_fn = None
        t._version = 0
        return t

    # ---- Constructors ----

    @staticmethod
    def zeros(shape, requires_grad=False):
        n = B.numel(shape)
        return Tensor._make(B.zeros(n), shape, requires_grad)

    @staticmethod
    def ones(shape, requires_grad=False):
        n = B.numel(shape)
        return Tensor._make(B.ones(n), shape, requires_grad)

    @staticmethod
    def rand(shape, requires_grad=False):
        n = B.numel(shape)
        return Tensor._make(B.rand_data(n), shape, requires_grad)

    @staticmethod
    def randn(shape, requires_grad=False):
        n = B.numel(shape)
        return Tensor._make(B.randn_data(n), shape, requires_grad)

    @staticmethod
    def full(shape, value, requires_grad=False):
        n = B.numel(shape)
        return Tensor._make([float(value)] * n, shape, requires_grad)

    @staticmethod
    def eye(n, requires_grad=False):
        data = [0.0] * (n * n)
        for i in range(n):
            data[i * n + i] = 1.0
        return Tensor._make(data, [n, n], requires_grad)

    @staticmethod
    def arange(start, end, step=1.0):
        data = []
        v = start
        if step > 0:
            while v < end:
                data.append(v)
                v += step
        elif step < 0:
            while v > end:
                data.append(v)
                v += step
        return Tensor._make(data, [len(data)])

    # ---- Properties ----

    @property
    def shape(self):
        return tuple(self._shape)

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def numel(self):
        return B.numel(self._shape)

    @property
    def data(self):
        """Return a detached view (no grad tracking)."""
        t = Tensor._make(self._data[:], self._shape[:])
        t._strides = self._strides[:]
        t._offset = self._offset
        return t

    @data.setter
    def data(self, value):
        """Set data from another tensor (for optimizer updates)."""
        if isinstance(value, Tensor):
            cont = value._contiguous_data()
            self._data = cont
            self._shape = list(value._shape)
            self._strides = B.compute_strides(self._shape)
            self._offset = 0
            self._version += 1
        else:
            raise TypeError("data must be a Tensor")

    def item(self):
        assert self.numel == 1, f"item() requires single-element tensor, got {self.numel}"
        return self._get_flat(0) if not self._shape else self._contiguous_data()[0]

    def _get_flat(self, i):
        """Get element at flat contiguous index i."""
        if self._is_contiguous() and self._offset == 0:
            return self._data[i]
        # Decompose flat index into multi-dim and use strides
        coord = []
        for s in self._shape:
            coord.append(i % s) if not self._shape else None
        # Actually need proper decomposition
        return self._contiguous_data()[i]

    def _is_contiguous(self):
        return self._strides == B.compute_strides(self._shape)

    def _contiguous_data(self):
        """Return contiguous flat data."""
        data, _ = B.to_contiguous(self._data, self._shape, self._strides, self._offset)
        return data

    def tolist(self):
        """Convert to nested Python list."""
        data = self._contiguous_data()
        if not self._shape:
            return data[0] if data else 0.0

        def _build(data, shape, offset):
            if len(shape) == 1:
                return data[offset:offset + shape[0]]
            stride = B.numel(shape[1:])
            return [_build(data, shape[1:], offset + i * stride) for i in range(shape[0])]

        return _build(data, self._shape, 0)

    def numpy(self):
        """Convert to a flat list (for compatibility)."""
        return self._contiguous_data()

    def detach(self):
        """Return a new tensor detached from the computation graph."""
        t = Tensor._make(self._contiguous_data(), self._shape[:])
        return t

    def clone(self):
        """Return a deep copy that shares the computation graph."""
        t = Tensor._make(self._contiguous_data(), self._shape[:], self.requires_grad)
        return t

    def contiguous(self):
        if self._is_contiguous() and self._offset == 0:
            return self
        data = self._contiguous_data()
        t = Tensor._make(data, self._shape[:], self.requires_grad)
        t.grad_fn = self.grad_fn
        return t

    # ---- Shape ops (views) ----

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
                assert neg_idx is None, "Only one -1 allowed in reshape"
                neg_idx = i
            else:
                known *= s
        if neg_idx is not None:
            shape[neg_idx] = self.numel // known

        assert B.numel(shape) == self.numel, f"Cannot reshape {self._shape} to {shape}"
        from cakelamp.autograd import ReshapeBackward
        data = self._contiguous_data()
        result = Tensor._make(data, shape, self.requires_grad)
        if self.requires_grad:
            result.grad_fn = ReshapeBackward(self)
        return result

    def view(self, *shape):
        return self.reshape(*shape)

    def transpose(self, dim0, dim1):
        from cakelamp.autograd import TransposeBackward
        new_shape = list(self._shape)
        new_strides = list(self._strides)
        new_shape[dim0], new_shape[dim1] = new_shape[dim1], new_shape[dim0]
        new_strides[dim0], new_strides[dim1] = new_strides[dim1], new_strides[dim0]

        t = Tensor.__new__(Tensor)
        t._data = self._data  # shared storage
        t._shape = new_shape
        t._strides = new_strides
        t._offset = self._offset
        t.requires_grad = self.requires_grad
        t.grad = None
        t.grad_fn = TransposeBackward(self, dim0, dim1) if self.requires_grad else None
        t._version = 0
        return t

    @property
    def T(self):
        assert self.ndim == 2, "T requires 2-D tensor"
        return self.transpose(0, 1)

    def unsqueeze(self, dim):
        new_shape = list(self._shape)
        new_strides = list(self._strides)
        if dim < 0:
            dim = self.ndim + 1 + dim
        s = new_strides[dim] * new_shape[dim] if dim < self.ndim else 1
        new_shape.insert(dim, 1)
        new_strides.insert(dim, s)

        t = Tensor.__new__(Tensor)
        t._data = self._data
        t._shape = new_shape
        t._strides = new_strides
        t._offset = self._offset
        t.requires_grad = self.requires_grad
        t.grad = None
        t.grad_fn = None
        t._version = 0
        return t

    def squeeze(self, dim=None):
        new_shape = []
        new_strides = []
        for i, (s, st) in enumerate(zip(self._shape, self._strides)):
            should_squeeze = (dim is None and s == 1) or (dim is not None and i == dim and s == 1)
            if not should_squeeze:
                new_shape.append(s)
                new_strides.append(st)

        t = Tensor.__new__(Tensor)
        t._data = self._data
        t._shape = new_shape
        t._strides = new_strides
        t._offset = self._offset
        t.requires_grad = self.requires_grad
        t.grad = None
        t.grad_fn = None
        t._version = 0
        return t

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = list(shape[0])
        else:
            shape = list(shape)
        ndim = len(shape)
        pad = ndim - self.ndim
        new_strides = [0] * ndim
        for i in range(self.ndim):
            ni = i + pad
            if self._shape[i] == shape[ni]:
                new_strides[ni] = self._strides[i]
            elif self._shape[i] == 1:
                new_strides[ni] = 0
            else:
                raise ValueError(f"Cannot expand dim {i} from {self._shape[i]} to {shape[ni]}")

        t = Tensor.__new__(Tensor)
        t._data = self._data
        t._shape = shape
        t._strides = new_strides
        t._offset = self._offset
        t.requires_grad = self.requires_grad
        t.grad = None
        t.grad_fn = None
        t._version = 0
        return t

    # ---- Element-wise ops ----

    def __add__(self, other):
        from cakelamp.autograd import AddBackward
        other = _ensure_tensor(other)
        a_data = self._contiguous_data()
        b_data = other._contiguous_data()
        result_data, result_shape = B.binary_op(
            a_data, self._shape, B.compute_strides(self._shape), 0,
            b_data, other._shape, B.compute_strides(other._shape), 0,
            lambda a, b: a + b
        )
        result = Tensor._make(result_data, result_shape,
                              self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.grad_fn = AddBackward(self, other)
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        from cakelamp.autograd import SubBackward
        other = _ensure_tensor(other)
        a_data = self._contiguous_data()
        b_data = other._contiguous_data()
        result_data, result_shape = B.binary_op(
            a_data, self._shape, B.compute_strides(self._shape), 0,
            b_data, other._shape, B.compute_strides(other._shape), 0,
            lambda a, b: a - b
        )
        result = Tensor._make(result_data, result_shape,
                              self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.grad_fn = SubBackward(self, other)
        return result

    def __rsub__(self, other):
        other = _ensure_tensor(other)
        return other.__sub__(self)

    def __mul__(self, other):
        from cakelamp.autograd import MulBackward
        other = _ensure_tensor(other)
        a_data = self._contiguous_data()
        b_data = other._contiguous_data()
        result_data, result_shape = B.binary_op(
            a_data, self._shape, B.compute_strides(self._shape), 0,
            b_data, other._shape, B.compute_strides(other._shape), 0,
            lambda a, b: a * b
        )
        result = Tensor._make(result_data, result_shape,
                              self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.grad_fn = MulBackward(self, other)
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        from cakelamp.autograd import DivBackward
        other = _ensure_tensor(other)
        a_data = self._contiguous_data()
        b_data = other._contiguous_data()
        result_data, result_shape = B.binary_op(
            a_data, self._shape, B.compute_strides(self._shape), 0,
            b_data, other._shape, B.compute_strides(other._shape), 0,
            lambda a, b: a / b
        )
        result = Tensor._make(result_data, result_shape,
                              self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.grad_fn = DivBackward(self, other)
        return result

    def __rtruediv__(self, other):
        other = _ensure_tensor(other)
        return other.__truediv__(self)

    def __neg__(self):
        from cakelamp.autograd import NegBackward
        data = self._contiguous_data()
        result_data = [-x for x in data]
        result = Tensor._make(result_data, self._shape[:], self.requires_grad)
        if self.requires_grad:
            result.grad_fn = NegBackward(self)
        return result

    def __pow__(self, other):
        from cakelamp.autograd import PowBackward
        other = _ensure_tensor(other)
        a_data = self._contiguous_data()
        b_data = other._contiguous_data()
        result_data, result_shape = B.binary_op(
            a_data, self._shape, B.compute_strides(self._shape), 0,
            b_data, other._shape, B.compute_strides(other._shape), 0,
            lambda a, b: a ** b
        )
        result = Tensor._make(result_data, result_shape,
                              self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.grad_fn = PowBackward(self, other)
        return result

    def exp(self):
        from cakelamp.autograd import ExpBackward
        data = self._contiguous_data()
        result_data = [math.exp(x) for x in data]
        result = Tensor._make(result_data, self._shape[:], self.requires_grad)
        if self.requires_grad:
            result.grad_fn = ExpBackward(self, result)
        return result

    def log(self):
        from cakelamp.autograd import LogBackward
        data = self._contiguous_data()
        result_data = [math.log(x) for x in data]
        result = Tensor._make(result_data, self._shape[:], self.requires_grad)
        if self.requires_grad:
            result.grad_fn = LogBackward(self)
        return result

    def relu(self):
        from cakelamp.autograd import ReluBackward
        data = self._contiguous_data()
        result_data = [max(0.0, x) for x in data]
        result = Tensor._make(result_data, self._shape[:], self.requires_grad)
        if self.requires_grad:
            result.grad_fn = ReluBackward(self)
        return result

    def sigmoid(self):
        from cakelamp.autograd import SigmoidBackward
        data = self._contiguous_data()
        result_data = [1.0 / (1.0 + math.exp(-x)) for x in data]
        result = Tensor._make(result_data, self._shape[:], self.requires_grad)
        if self.requires_grad:
            result.grad_fn = SigmoidBackward(self, result)
        return result

    def tanh(self):
        from cakelamp.autograd import TanhBackward
        data = self._contiguous_data()
        result_data = [math.tanh(x) for x in data]
        result = Tensor._make(result_data, self._shape[:], self.requires_grad)
        if self.requires_grad:
            result.grad_fn = TanhBackward(self, result)
        return result

    def sqrt(self):
        return self ** 0.5

    def abs(self):
        data = self._contiguous_data()
        result_data = [builtins_abs(x) for x in data]
        return Tensor._make(result_data, self._shape[:], self.requires_grad)

    def clamp(self, min_val=None, max_val=None):
        data = self._contiguous_data()
        def _clamp(x):
            if min_val is not None:
                x = max(x, min_val)
            if max_val is not None:
                x = min(x, max_val)
            return x
        result_data = [_clamp(x) for x in data]
        return Tensor._make(result_data, self._shape[:], self.requires_grad)

    # ---- Reduction ops ----

    def sum(self, dim=None, keepdim=False):
        from cakelamp.autograd import SumBackward
        if dim is None:
            data = self._contiguous_data()
            s = sum(data)
            result = Tensor._make([s], [], self.requires_grad)
            if self.requires_grad:
                result.grad_fn = SumBackward(self, dim=None, keepdim=keepdim)
            return result
        else:
            data = self._contiguous_data()
            result_data, result_shape = B.sum_dim(data, self._shape, dim, keepdim)
            result = Tensor._make(result_data, result_shape, self.requires_grad)
            if self.requires_grad:
                result.grad_fn = SumBackward(self, dim=dim, keepdim=keepdim)
            return result

    def mean(self, dim=None, keepdim=False):
        from cakelamp.autograd import MeanBackward
        if dim is None:
            s = self.sum()
            n = self.numel
            result = s / Tensor._make([float(n)], [])
            if self.requires_grad:
                result.grad_fn = MeanBackward(self, dim=None, keepdim=keepdim)
            return result
        else:
            s = self.sum(dim=dim, keepdim=keepdim)
            n = self._shape[dim]
            result = s / Tensor._make([float(n)], [])
            if self.requires_grad:
                result.grad_fn = MeanBackward(self, dim=dim, keepdim=keepdim)
            return result

    def max(self, dim=None):
        data = self._contiguous_data()
        if dim is None:
            return Tensor._make([builtins_max(data)], [])
        # Along dim
        result_data, result_shape = B.sum_dim(data, self._shape, dim, False)
        # Actually need max, not sum - recompute
        return self._reduce_dim(dim, max_fn=True)

    def _reduce_dim(self, dim, max_fn=False):
        """Generic reduction along dim."""
        data = self._contiguous_data()
        ndim = len(self._shape)
        dim_size = self._shape[dim]
        out_shape = [s for i, s in enumerate(self._shape) if i != dim]
        if not out_shape:
            if max_fn:
                return Tensor._make([builtins_max(data)], [])
            return Tensor._make([sum(data)], [])

        out_n = B.numel(out_shape)
        result = [0.0] * out_n
        in_strides = B.compute_strides(self._shape)

        coord = [0] * len(out_shape)
        for out_i in range(out_n):
            vals = []
            for k in range(dim_size):
                in_idx = 0
                out_d = 0
                for d in range(ndim):
                    if d == dim:
                        in_idx += k * in_strides[d]
                    else:
                        in_idx += coord[out_d] * in_strides[d]
                        out_d += 1
                vals.append(data[in_idx])

            result[out_i] = builtins_max(vals) if max_fn else sum(vals)

            for d in range(len(out_shape) - 1, -1, -1):
                coord[d] += 1
                if coord[d] < out_shape[d]:
                    break
                coord[d] = 0

        return Tensor._make(result, out_shape)

    def argmax(self, dim=None):
        data = self._contiguous_data()
        if dim is None:
            best_idx = 0
            best_val = data[0]
            for i in range(1, len(data)):
                if data[i] > best_val:
                    best_val = data[i]
                    best_idx = i
            return Tensor._make([float(best_idx)], [])
        result_data, result_shape = B.argmax_dim(data, self._shape, dim)
        return Tensor._make(result_data, result_shape)

    # ---- Matrix ops ----

    def mm(self, other):
        """Matrix multiply: (M,K) @ (K,N) -> (M,N)"""
        from cakelamp.autograd import MatmulBackward
        assert self.ndim == 2 and other.ndim == 2
        a_data = self._contiguous_data()
        b_data = other._contiguous_data()
        result_data, result_shape = B.matmul(a_data, list(self._shape), b_data, list(other._shape))
        result = Tensor._make(result_data, result_shape,
                              self.requires_grad or other.requires_grad)
        if result.requires_grad:
            result.grad_fn = MatmulBackward(self, other)
        return result

    def matmul(self, other):
        return self.mm(other)

    def __matmul__(self, other):
        return self.mm(other)

    # ---- Softmax ----

    def softmax(self, dim=-1):
        if dim < 0:
            dim = self.ndim + dim
        data = self._contiguous_data()
        result_data, result_shape = B.softmax_data(data, list(self._shape), dim)
        return Tensor._make(result_data, result_shape, self.requires_grad)

    def log_softmax(self, dim=-1):
        from cakelamp.autograd import LogSoftmaxBackward
        if dim < 0:
            dim = self.ndim + dim
        # Numerically stable: log_softmax = x - log(sum(exp(x)))
        data = self._contiguous_data()
        sm_data, sm_shape = B.softmax_data(data, list(self._shape), dim)
        result_data = [math.log(max(x, 1e-12)) for x in sm_data]
        result = Tensor._make(result_data, sm_shape, self.requires_grad)
        if self.requires_grad:
            sm = Tensor._make(sm_data, sm_shape)
            result.grad_fn = LogSoftmaxBackward(self, sm, dim)
        return result

    # ---- In-place ops ----

    def add_(self, other):
        other = _ensure_tensor(other)
        result = self + other
        self._data = result._contiguous_data()
        self._shape = list(result._shape)
        self._strides = B.compute_strides(self._shape)
        self._offset = 0
        self._version += 1
        return self

    def sub_(self, other):
        other = _ensure_tensor(other)
        result = self - other
        self._data = result._contiguous_data()
        self._shape = list(result._shape)
        self._strides = B.compute_strides(self._shape)
        self._offset = 0
        self._version += 1
        return self

    def mul_(self, other):
        other = _ensure_tensor(other)
        result = self * other
        self._data = result._contiguous_data()
        self._shape = list(result._shape)
        self._strides = B.compute_strides(self._shape)
        self._offset = 0
        self._version += 1
        return self

    def zero_(self):
        n = self.numel
        self._data = [0.0] * n
        self._strides = B.compute_strides(self._shape)
        self._offset = 0
        self._version += 1
        return self

    def fill_(self, val):
        n = self.numel
        self._data = [float(val)] * n
        self._strides = B.compute_strides(self._shape)
        self._offset = 0
        self._version += 1
        return self

    def copy_(self, src):
        """Copy data from src tensor."""
        assert self.shape == src.shape, f"Shape mismatch: {self.shape} vs {src.shape}"
        self._data = src._contiguous_data()
        self._strides = B.compute_strides(self._shape)
        self._offset = 0
        self._version += 1
        return self

    # ---- Autograd ----

    def backward(self, gradient=None):
        """Compute gradients via reverse-mode autodiff."""
        if gradient is None:
            assert self.numel == 1, "backward() requires scalar output or explicit gradient"
            gradient = Tensor.ones(list(self._shape))

        # Topological sort
        topo = []
        visited = set()

        def _topo_sort(t):
            if id(t) in visited:
                return
            visited.add(id(t))
            if t.grad_fn is not None:
                for parent in t.grad_fn.inputs():
                    if isinstance(parent, Tensor) and parent.requires_grad:
                        _topo_sort(parent)
                topo.append(t)

        _topo_sort(self)
        topo.reverse()

        # Set gradient for the output
        self.grad = gradient

        # Backpropagate
        grads = {id(self): gradient}
        for t in topo:
            grad = grads.get(id(t))
            if grad is None or t.grad_fn is None:
                continue

            input_grads = t.grad_fn.backward(grad)
            inputs = t.grad_fn.inputs()

            for inp, inp_grad in zip(inputs, input_grads):
                if isinstance(inp, Tensor) and inp.requires_grad and inp_grad is not None:
                    if id(inp) in grads:
                        grads[id(inp)] = grads[id(inp)] + inp_grad
                    else:
                        grads[id(inp)] = inp_grad

        # Assign gradients
        for t_id, grad in grads.items():
            for t in topo:
                if id(t) == t_id:
                    t.grad = grad
                    break
            # Also check inputs
            for t in topo:
                if t.grad_fn:
                    for inp in t.grad_fn.inputs():
                        if isinstance(inp, Tensor) and id(inp) == t_id:
                            inp.grad = grad

    # ---- Comparison ----

    def __eq__(self, other):
        other = _ensure_tensor(other)
        a_data = self._contiguous_data()
        b_data = other._contiguous_data()
        result_data, result_shape = B.binary_op(
            a_data, self._shape, B.compute_strides(self._shape), 0,
            b_data, other._shape, B.compute_strides(other._shape), 0,
            lambda a, b: 1.0 if abs(a - b) < 1e-7 else 0.0
        )
        return Tensor._make(result_data, result_shape)

    def __gt__(self, other):
        other = _ensure_tensor(other)
        a_data = self._contiguous_data()
        b_data = other._contiguous_data()
        result_data, result_shape = B.binary_op(
            a_data, self._shape, B.compute_strides(self._shape), 0,
            b_data, other._shape, B.compute_strides(other._shape), 0,
            lambda a, b: 1.0 if a > b else 0.0
        )
        return Tensor._make(result_data, result_shape)

    def __lt__(self, other):
        other = _ensure_tensor(other)
        a_data = self._contiguous_data()
        b_data = other._contiguous_data()
        result_data, result_shape = B.binary_op(
            a_data, self._shape, B.compute_strides(self._shape), 0,
            b_data, other._shape, B.compute_strides(other._shape), 0,
            lambda a, b: 1.0 if a < b else 0.0
        )
        return Tensor._make(result_data, result_shape)

    # ---- Repr ----

    def __repr__(self):
        data = self._contiguous_data()
        if not self._shape:
            return f"tensor({data[0]:.4f})"
        if self.numel <= 20:
            return f"tensor({self.tolist()}, shape={self.shape})"
        return f"tensor(shape={self.shape})"

    def __len__(self):
        if not self._shape:
            raise TypeError("len() of a 0-d tensor")
        return self._shape[0]

    def __getitem__(self, idx):
        """Basic indexing support."""
        if isinstance(idx, int):
            if self.ndim == 0:
                raise IndexError("Cannot index 0-d tensor")
            if idx < 0:
                idx += self._shape[0]
            new_shape = self._shape[1:]
            new_strides = self._strides[1:]
            new_offset = self._offset + idx * self._strides[0]

            t = Tensor.__new__(Tensor)
            t._data = self._data
            t._shape = new_shape
            t._strides = new_strides
            t._offset = new_offset
            t.requires_grad = self.requires_grad
            t.grad = None
            t.grad_fn = None
            t._version = 0
            return t
        raise NotImplementedError("Only integer indexing is supported")


def _ensure_tensor(x):
    """Convert scalar to tensor if needed."""
    if isinstance(x, Tensor):
        return x
    if isinstance(x, (int, float)):
        return Tensor._make([float(x)], [])
    raise TypeError(f"Cannot convert {type(x)} to Tensor")


import builtins
builtins_abs = builtins.abs
builtins_max = builtins.max

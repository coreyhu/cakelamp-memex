"""Tests for cakelamp.nn module.

Uses mock objects to test Module, Parameter, Sequential, Linear,
activation modules, loss functions, and state_dict serialisation.
"""

from __future__ import annotations

import sys
import os
import math
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.dirname(__file__))

from cakelamp.nn.module import Module
from cakelamp.nn.parameter import Parameter
from cakelamp.nn.linear import Linear
from cakelamp.nn.activations import ReLU, Sigmoid, Tanh, Softmax, LogSoftmax
from cakelamp.nn.loss import MSELoss, CrossEntropyLoss, NLLLoss
from cakelamp.nn.containers import Sequential
from cakelamp.nn.dropout import Dropout


# =====================================================================
# Mock tensors for testing nn module logic
# =====================================================================

class SimpleTensor:
    """Minimal mock tensor for testing nn module infrastructure."""

    def __init__(self, data, shape=None):
        if isinstance(data, list):
            self._data = data
        else:
            self._data = [data]
        self._shape = shape or [len(self._data)]

    def clone(self):
        return SimpleTensor(list(self._data), list(self._shape))

    def t(self):
        """Transpose (for 2D: swap dimensions)."""
        if len(self._shape) == 2:
            rows, cols = self._shape
            new_data = []
            for j in range(cols):
                for i in range(rows):
                    new_data.append(self._data[i * cols + j])
            return SimpleTensor(new_data, [cols, rows])
        return self

    def matmul(self, other):
        """Simple 2D matmul."""
        assert len(self._shape) == 2 and len(other._shape) == 2
        m, k = self._shape
        k2, n = other._shape
        assert k == k2
        result = []
        for i in range(m):
            for j in range(n):
                s = 0.0
                for p in range(k):
                    s += self._data[i * k + p] * other._data[p * n + j]
                result.append(s)
        return SimpleTensor(result, [m, n])

    def add(self, other):
        if isinstance(other, SimpleTensor):
            # Broadcasting: if other is 1D and self is 2D, broadcast across rows
            if len(self._shape) == 2 and len(other._shape) == 1:
                rows, cols = self._shape
                result = []
                for i in range(rows):
                    for j in range(cols):
                        result.append(self._data[i * cols + j] + other._data[j])
                return SimpleTensor(result, [rows, cols])
            return SimpleTensor([a + b for a, b in zip(self._data, other._data)], list(self._shape))
        return SimpleTensor([a + other for a in self._data], list(self._shape))

    def sub(self, other):
        if isinstance(other, SimpleTensor):
            return SimpleTensor([a - b for a, b in zip(self._data, other._data)], list(self._shape))
        return SimpleTensor([a - other for a in self._data], list(self._shape))

    def mul(self, other):
        if isinstance(other, SimpleTensor):
            return SimpleTensor([a * b for a, b in zip(self._data, other._data)], list(self._shape))
        return SimpleTensor([a * other for a in self._data], list(self._shape))

    def relu(self):
        return SimpleTensor([max(0.0, x) for x in self._data], list(self._shape))

    def sigmoid(self):
        return SimpleTensor([1.0 / (1.0 + math.exp(-x)) for x in self._data], list(self._shape))

    def tanh(self):
        return SimpleTensor([math.tanh(x) for x in self._data], list(self._shape))

    def softmax(self, dim):
        # Simple 1D softmax
        max_val = max(self._data)
        exps = [math.exp(x - max_val) for x in self._data]
        s = sum(exps)
        return SimpleTensor([e / s for e in exps], list(self._shape))

    def log_softmax(self, dim):
        sm = self.softmax(dim)
        return SimpleTensor([math.log(x) for x in sm._data], list(self._shape))

    def mean(self):
        return SimpleTensor([sum(self._data) / len(self._data)])

    def sum(self):
        return SimpleTensor([sum(self._data)])

    def zero_(self):
        self._data = [0.0] * len(self._data)
        return self

    def dropout(self, p, training):
        if not training:
            return self
        return self  # Simplified for testing

    def tolist(self):
        return list(self._data)

    @property
    def shape(self):
        return self._shape

    def __repr__(self):
        return f"SimpleTensor({self._data})"


# =====================================================================
# Tests: Parameter
# =====================================================================

class TestParameter:
    def test_creation(self):
        t = SimpleTensor([1.0, 2.0, 3.0])
        p = Parameter(t)
        assert p.data is t
        assert p.requires_grad is True
        assert p.grad is None

    def test_no_grad(self):
        t = SimpleTensor([1.0])
        p = Parameter(t, requires_grad=False)
        assert p.requires_grad is False

    def test_zero_grad(self):
        p = Parameter(SimpleTensor([1.0]))
        p.grad = SimpleTensor([0.5])
        p.zero_grad()
        assert p.grad is None

    def test_repr(self):
        p = Parameter(SimpleTensor([1.0]))
        assert "Parameter" in repr(p)


# =====================================================================
# Tests: Module
# =====================================================================

class SimpleModule(Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.w = Parameter(SimpleTensor([0.5] * (in_f * out_f), [out_f, in_f]))
        self.in_f = in_f
        self.out_f = out_f

    def forward(self, x):
        return x.matmul(self.w.data.t())


class TestModule:
    def test_parameter_registration(self):
        m = SimpleModule(2, 3)
        params = list(m.parameters())
        assert len(params) == 1
        assert params[0] is m.w

    def test_named_parameters(self):
        m = SimpleModule(2, 3)
        named = dict(m.named_parameters())
        assert "w" in named

    def test_children(self):
        parent = Module()
        child1 = SimpleModule(2, 3)
        child2 = SimpleModule(3, 1)
        parent._modules["c1"] = child1
        parent._modules["c2"] = child2
        children = list(parent.children())
        assert len(children) == 2

    def test_recursive_parameters(self):
        parent = Module()
        parent._modules["child"] = SimpleModule(2, 3)
        params = list(parent.parameters())
        assert len(params) == 1

    def test_train_eval(self):
        m = SimpleModule(2, 3)
        assert m.training is True
        m.eval()
        assert m.training is False
        m.train()
        assert m.training is True

    def test_train_propagates_to_children(self):
        parent = Module()
        child = SimpleModule(2, 3)
        parent._modules["child"] = child
        parent.eval()
        assert child.training is False
        parent.train()
        assert child.training is True

    def test_forward_not_implemented(self):
        m = Module()
        with pytest.raises(NotImplementedError):
            m(SimpleTensor([1.0]))

    def test_call_delegates_to_forward(self):
        m = SimpleModule(2, 3)
        x = SimpleTensor([1.0, 2.0], [1, 2])
        result = m(x)
        assert len(result._data) == 3

    def test_state_dict(self):
        m = SimpleModule(2, 3)
        sd = m.state_dict()
        assert "w" in sd

    def test_load_state_dict(self):
        m = SimpleModule(2, 3)
        new_data = SimpleTensor([9.0] * 6, [3, 2])
        m.load_state_dict({"w": new_data})
        assert m.w.data._data == [9.0] * 6

    def test_zero_grad(self):
        m = SimpleModule(2, 3)
        m.w.grad = SimpleTensor([1.0] * 6)
        m.zero_grad()
        assert m.w.grad is None

    def test_modules_recursive(self):
        parent = Module()
        child = SimpleModule(2, 3)
        parent._modules["child"] = child
        all_mods = list(parent.modules())
        assert len(all_mods) == 2
        assert parent in all_mods
        assert child in all_mods

    def test_named_modules(self):
        parent = Module()
        parent._modules["child"] = SimpleModule(2, 3)
        named = dict(parent.named_modules())
        assert "" in named  # parent itself
        assert "child" in named

    def test_repr(self):
        m = SimpleModule(2, 3)
        r = repr(m)
        assert "SimpleModule" in r

    def test_setattr_auto_registers_parameter(self):
        m = Module()
        p = Parameter(SimpleTensor([1.0]))
        m.p = p
        assert "p" in m._parameters
        assert list(m.parameters()) == [p]

    def test_setattr_auto_registers_module(self):
        m = Module()
        child = SimpleModule(2, 3)
        m.child = child
        assert "child" in m._modules
        assert list(m.children()) == [child]


# =====================================================================
# Tests: Linear
# =====================================================================

class TestLinear:
    def test_creation(self):
        layer = Linear(10, 5)
        assert layer.in_features == 10
        assert layer.out_features == 5
        assert layer.weight is not None
        assert layer.bias is not None

    def test_no_bias(self):
        layer = Linear(10, 5, bias=False)
        assert layer.bias is None

    def test_parameters_count(self):
        layer = Linear(10, 5)
        params = list(layer.parameters())
        assert len(params) == 2  # weight + bias

    def test_parameters_count_no_bias(self):
        layer = Linear(10, 5, bias=False)
        params = list(layer.parameters())
        assert len(params) == 1  # weight only

    def test_named_parameters(self):
        layer = Linear(10, 5)
        named = dict(layer.named_parameters())
        assert "weight" in named
        assert "bias" in named

    def test_repr(self):
        layer = Linear(10, 5)
        r = repr(layer)
        assert "in_features=10" in r
        assert "out_features=5" in r
        assert "bias=True" in r

    def test_weight_shape(self):
        layer = Linear(4, 3)
        # Weight should have out_features * in_features elements
        assert len(layer.weight.data) == 12  # 3*4

    def test_state_dict(self):
        layer = Linear(4, 3)
        sd = layer.state_dict()
        assert "weight" in sd
        assert "bias" in sd


# =====================================================================
# Tests: Activations
# =====================================================================

class TestActivations:
    def test_relu(self):
        act = ReLU()
        x = SimpleTensor([-1.0, 0.0, 1.0, 2.0])
        result = act(x)
        assert result.tolist() == [0.0, 0.0, 1.0, 2.0]

    def test_sigmoid(self):
        act = Sigmoid()
        x = SimpleTensor([0.0])
        result = act(x)
        assert abs(result.tolist()[0] - 0.5) < 1e-6

    def test_sigmoid_large(self):
        act = Sigmoid()
        x = SimpleTensor([10.0])
        result = act(x)
        assert result.tolist()[0] > 0.99

    def test_tanh(self):
        act = Tanh()
        x = SimpleTensor([0.0])
        result = act(x)
        assert abs(result.tolist()[0]) < 1e-6

    def test_softmax(self):
        act = Softmax(dim=-1)
        x = SimpleTensor([1.0, 2.0, 3.0])
        result = act(x)
        # Should sum to 1
        assert abs(sum(result.tolist()) - 1.0) < 1e-6
        # Should be monotonically increasing
        vals = result.tolist()
        assert vals[0] < vals[1] < vals[2]

    def test_log_softmax(self):
        act = LogSoftmax(dim=-1)
        x = SimpleTensor([1.0, 2.0, 3.0])
        result = act(x)
        # All values should be negative
        assert all(v < 0 for v in result.tolist())
        # exp should sum to ~1
        assert abs(sum(math.exp(v) for v in result.tolist()) - 1.0) < 1e-6

    def test_relu_training_mode(self):
        act = ReLU()
        assert act.training is True
        act.eval()
        # ReLU behaves the same in eval
        x = SimpleTensor([-1.0, 1.0])
        result = act(x)
        assert result.tolist() == [0.0, 1.0]


# =====================================================================
# Tests: Loss functions
# =====================================================================

class TestLoss:
    def test_mse_loss_mean(self):
        loss_fn = MSELoss(reduction="mean")
        pred = SimpleTensor([1.0, 2.0, 3.0])
        target = SimpleTensor([1.5, 2.5, 3.5])
        loss = loss_fn(pred, target)
        # mean((0.5^2, 0.5^2, 0.5^2)) = 0.25
        assert abs(loss._data[0] - 0.25) < 1e-6

    def test_mse_loss_sum(self):
        loss_fn = MSELoss(reduction="sum")
        pred = SimpleTensor([1.0, 2.0, 3.0])
        target = SimpleTensor([1.5, 2.5, 3.5])
        loss = loss_fn(pred, target)
        # sum(0.25, 0.25, 0.25) = 0.75
        assert abs(loss._data[0] - 0.75) < 1e-6

    def test_mse_loss_none(self):
        loss_fn = MSELoss(reduction="none")
        pred = SimpleTensor([1.0, 2.0])
        target = SimpleTensor([2.0, 2.0])
        loss = loss_fn(pred, target)
        assert len(loss._data) == 2
        assert abs(loss._data[0] - 1.0) < 1e-6
        assert abs(loss._data[1] - 0.0) < 1e-6

    def test_mse_loss_zero(self):
        loss_fn = MSELoss()
        pred = SimpleTensor([1.0, 2.0])
        target = SimpleTensor([1.0, 2.0])
        loss = loss_fn(pred, target)
        assert abs(loss._data[0]) < 1e-6

    def test_mse_invalid_reduction(self):
        with pytest.raises(ValueError, match="Invalid reduction"):
            MSELoss(reduction="invalid")

    def test_mse_repr(self):
        loss_fn = MSELoss()
        assert "mean" in repr(loss_fn)

    def test_cross_entropy_invalid_reduction(self):
        with pytest.raises(ValueError, match="Invalid reduction"):
            CrossEntropyLoss(reduction="bad")

    def test_nll_invalid_reduction(self):
        with pytest.raises(ValueError, match="Invalid reduction"):
            NLLLoss(reduction="bad")


# =====================================================================
# Tests: Sequential
# =====================================================================

class TestSequential:
    def test_forward(self):
        seq = Sequential(ReLU())
        x = SimpleTensor([-1.0, 0.0, 1.0])
        result = seq(x)
        assert result.tolist() == [0.0, 0.0, 1.0]

    def test_multiple_layers(self):
        seq = Sequential(ReLU(), ReLU())
        x = SimpleTensor([-1.0, 1.0])
        result = seq(x)
        assert result.tolist() == [0.0, 1.0]

    def test_len(self):
        seq = Sequential(ReLU(), Sigmoid(), Tanh())
        assert len(seq) == 3

    def test_getitem(self):
        relu = ReLU()
        sigmoid = Sigmoid()
        seq = Sequential(relu, sigmoid)
        assert seq[0] is relu
        assert seq[1] is sigmoid
        assert seq[-1] is sigmoid

    def test_iter(self):
        relu = ReLU()
        sigmoid = Sigmoid()
        seq = Sequential(relu, sigmoid)
        modules = list(seq)
        assert modules[0] is relu
        assert modules[1] is sigmoid

    def test_append(self):
        seq = Sequential(ReLU())
        seq.append(Sigmoid())
        assert len(seq) == 2

    def test_parameters(self):
        # Sequential with Linear layers
        l1 = Linear(2, 3)
        l2 = Linear(3, 1)
        seq = Sequential(l1, ReLU(), l2)
        params = list(seq.parameters())
        # l1 has weight+bias, l2 has weight+bias = 4
        assert len(params) == 4

    def test_train_eval(self):
        seq = Sequential(ReLU(), Dropout(0.5))
        seq.eval()
        assert not seq.training
        for m in seq:
            assert not m.training
        seq.train()
        assert seq.training

    def test_state_dict(self):
        l = Linear(2, 3)
        seq = Sequential(l, ReLU())
        sd = seq.state_dict()
        assert "0.weight" in sd
        assert "0.bias" in sd

    def test_repr(self):
        seq = Sequential(ReLU(), Sigmoid())
        r = repr(seq)
        assert "Sequential" in r
        assert "ReLU" in r
        assert "Sigmoid" in r

    def test_from_ordered_dict(self):
        from collections import OrderedDict
        seq = Sequential(OrderedDict([
            ("relu", ReLU()),
            ("sigmoid", Sigmoid()),
        ]))
        assert len(seq) == 2
        sd = seq.state_dict()
        # No parameters in activations, but module should still work
        x = SimpleTensor([-1.0, 1.0])
        result = seq(x)


# =====================================================================
# Tests: Dropout
# =====================================================================

class TestDropout:
    def test_creation(self):
        d = Dropout(0.5)
        assert d.p == 0.5

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            Dropout(1.0)
        with pytest.raises(ValueError):
            Dropout(-0.1)

    def test_eval_mode_passthrough(self):
        d = Dropout(0.5)
        d.eval()
        x = SimpleTensor([1.0, 2.0, 3.0])
        result = d(x)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_repr(self):
        d = Dropout(0.3)
        assert "0.3" in repr(d)


# =====================================================================
# Tests: Nested module state_dict
# =====================================================================

class TestNestedStateDict:
    def test_nested_state_dict(self):
        """Test state_dict with nested modules."""
        model = Sequential(
            Linear(4, 3),
            ReLU(),
            Linear(3, 2),
        )
        sd = model.state_dict()
        assert "0.weight" in sd
        assert "0.bias" in sd
        assert "2.weight" in sd
        assert "2.bias" in sd

    def test_load_nested_state_dict(self):
        """Test loading state_dict into nested modules."""
        model = Sequential(Linear(2, 2), Linear(2, 1))
        sd = model.state_dict()
        # Modify weights
        sd["0.weight"] = SimpleTensor([9.0, 9.0, 9.0, 9.0], [2, 2])
        model.load_state_dict(sd)
        assert model[0].weight.data._data == [9.0, 9.0, 9.0, 9.0]

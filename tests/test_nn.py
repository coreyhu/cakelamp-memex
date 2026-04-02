"""Tests for cakelamp.nn module.

Tests Module, Parameter, Sequential, Linear, activation modules,
loss functions, and state_dict serialisation using the real Rust backend.
"""

from __future__ import annotations

import math
import pytest

import cakelamp._core as _C
from cakelamp.autograd.tensor import AutogradTensor
from cakelamp.nn.module import Module
from cakelamp.nn.parameter import Parameter
from cakelamp.nn.linear import Linear
from cakelamp.nn.activations import ReLU, Sigmoid, Tanh, Softmax, LogSoftmax
from cakelamp.nn.loss import MSELoss, CrossEntropyLoss, NLLLoss
from cakelamp.nn.containers import Sequential
from cakelamp.nn.dropout import Dropout


# =====================================================================
# Tests: Parameter
# =====================================================================


class TestParameter:
    def test_creation(self):
        t = _C.Tensor([1.0, 2.0, 3.0], [3])
        p = Parameter(t)
        assert p.data is t
        assert p.requires_grad is True
        assert p.grad is None

    def test_no_grad(self):
        t = _C.Tensor([1.0], [1])
        p = Parameter(t, requires_grad=False)
        assert p.requires_grad is False

    def test_zero_grad(self):
        p = Parameter(_C.Tensor([1.0], [1]))
        p.grad = AutogradTensor(_C.Tensor([0.5], [1]))
        p.zero_grad()
        assert p.grad is None

    def test_is_autograd_tensor(self):
        p = Parameter(_C.Tensor([1.0], [1]))
        assert isinstance(p, AutogradTensor)


# =====================================================================
# Tests: Module
# =====================================================================


class SimpleModule(Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.w = Parameter(_C.Tensor([0.5] * (in_f * out_f), [out_f, in_f]))
        self.in_f = in_f
        self.out_f = out_f

    def forward(self, x):
        return x @ self.w.t()


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
            m(AutogradTensor(_C.Tensor([1.0], [1])))

    def test_call_delegates_to_forward(self):
        m = SimpleModule(2, 3)
        x = AutogradTensor(_C.Tensor([1.0, 2.0], [1, 2]))
        result = m(x)
        assert result.shape == [1, 3]

    def test_state_dict(self):
        m = SimpleModule(2, 3)
        sd = m.state_dict()
        assert "w" in sd

    def test_zero_grad(self):
        m = SimpleModule(2, 3)
        m.w.grad = AutogradTensor(_C.Tensor([1.0] * 6, [3, 2]))
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
        assert "" in named
        assert "child" in named

    def test_repr(self):
        m = SimpleModule(2, 3)
        r = repr(m)
        assert "SimpleModule" in r

    def test_setattr_auto_registers_parameter(self):
        m = Module()
        p = Parameter(_C.Tensor([1.0], [1]))
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

    def test_forward_shape(self):
        layer = Linear(4, 3)
        x = AutogradTensor(_C.Tensor([1.0, 2.0, 3.0, 4.0], [1, 4]))
        out = layer(x)
        assert out.shape == [1, 3]

    def test_forward_batch(self):
        layer = Linear(2, 3)
        x = AutogradTensor(_C.Tensor([1.0, 2.0, 3.0, 4.0], [2, 2]))
        out = layer(x)
        assert out.shape == [2, 3]

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
        x = AutogradTensor(_C.Tensor([-1.0, 0.0, 1.0, 2.0], [4]))
        result = act(x)
        assert result.tolist() == [0.0, 0.0, 1.0, 2.0]

    def test_sigmoid(self):
        act = Sigmoid()
        x = AutogradTensor(_C.Tensor([0.0], [1]))
        result = act(x)
        assert abs(result.tolist()[0] - 0.5) < 1e-6

    def test_sigmoid_large(self):
        act = Sigmoid()
        x = AutogradTensor(_C.Tensor([10.0], [1]))
        result = act(x)
        assert result.tolist()[0] > 0.99

    def test_tanh(self):
        act = Tanh()
        x = AutogradTensor(_C.Tensor([0.0], [1]))
        result = act(x)
        assert abs(result.tolist()[0]) < 1e-6

    def test_softmax(self):
        act = Softmax(dim=1)
        x = AutogradTensor(_C.Tensor([1.0, 2.0, 3.0], [1, 3]))
        result = act(x)
        vals = result.tolist()
        assert abs(sum(vals) - 1.0) < 1e-5
        assert vals[0] < vals[1] < vals[2]

    def test_log_softmax(self):
        act = LogSoftmax(dim=1)
        x = AutogradTensor(_C.Tensor([1.0, 2.0, 3.0], [1, 3]))
        result = act(x)
        vals = result.tolist()
        assert all(v < 0 for v in vals)
        assert abs(sum(math.exp(v) for v in vals) - 1.0) < 1e-5

    def test_relu_training_mode(self):
        act = ReLU()
        assert act.training is True
        act.eval()
        x = AutogradTensor(_C.Tensor([-1.0, 1.0], [2]))
        result = act(x)
        assert result.tolist() == [0.0, 1.0]


# =====================================================================
# Tests: Loss functions
# =====================================================================


class TestLoss:
    def test_mse_loss_zero(self):
        loss_fn = MSELoss()
        pred = AutogradTensor(_C.Tensor([1.0, 2.0], [2]))
        target = AutogradTensor(_C.Tensor([1.0, 2.0], [2]))
        loss = loss_fn(pred, target)
        assert abs(loss.item()) < 1e-6

    def test_mse_loss_nonzero(self):
        loss_fn = MSELoss()
        pred = AutogradTensor(_C.Tensor([1.0, 2.0, 3.0], [3]))
        target = AutogradTensor(_C.Tensor([1.5, 2.5, 3.5], [3]))
        loss = loss_fn(pred, target)
        # mean((0.5^2, 0.5^2, 0.5^2)) = 0.25
        assert abs(loss.item() - 0.25) < 1e-6

    def test_cross_entropy_loss(self):
        loss_fn = CrossEntropyLoss()
        logits = AutogradTensor(
            _C.Tensor([2.0, 1.0, 0.1, 0.1, 1.0, 2.0], [2, 3])
        )
        targets = AutogradTensor(_C.Tensor([0.0, 2.0], [2]))
        loss = loss_fn(logits, targets)
        assert loss.item() > 0

    def test_nll_loss(self):
        loss_fn = NLLLoss()
        log_probs = AutogradTensor(
            _C.Tensor([-0.5, -1.0, -2.0, -2.0, -1.0, -0.5], [2, 3])
        )
        targets = AutogradTensor(_C.Tensor([0.0, 2.0], [2]))
        loss = loss_fn(log_probs, targets)
        # -((-0.5) + (-0.5)) / 2 = 0.5
        assert abs(loss.item() - 0.5) < 1e-5

    def test_mse_repr(self):
        loss_fn = MSELoss()
        assert "MSELoss" in repr(loss_fn)


# =====================================================================
# Tests: Sequential
# =====================================================================


class TestSequential:
    def test_forward(self):
        seq = Sequential(ReLU())
        x = AutogradTensor(_C.Tensor([-1.0, 0.0, 1.0], [3]))
        result = seq(x)
        assert result.tolist() == [0.0, 0.0, 1.0]

    def test_multiple_layers(self):
        seq = Sequential(ReLU(), ReLU())
        x = AutogradTensor(_C.Tensor([-1.0, 1.0], [2]))
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
        l1 = Linear(2, 3)
        l2 = Linear(3, 1)
        seq = Sequential(l1, ReLU(), l2)
        params = list(seq.parameters())
        assert len(params) == 4  # l1 weight+bias, l2 weight+bias

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
        x = AutogradTensor(_C.Tensor([-1.0, 1.0], [2]))
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
        x = AutogradTensor(_C.Tensor([1.0, 2.0, 3.0], [3]))
        result = d(x)
        assert result.tolist() == [1.0, 2.0, 3.0]

    def test_repr(self):
        d = Dropout(0.3)
        assert "0.3" in repr(d)

    def test_zero_p_passthrough(self):
        d = Dropout(0.0)
        x = AutogradTensor(_C.Tensor([1.0, 2.0, 3.0], [3]))
        result = d(x)
        assert result.tolist() == [1.0, 2.0, 3.0]


# =====================================================================
# Tests: Nested module state_dict
# =====================================================================


class TestNestedStateDict:
    def test_nested_state_dict(self):
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

    def test_forward_through_mlp(self):
        """Test that a full MLP forward pass produces correct shapes."""
        model = Sequential(
            Linear(4, 3),
            ReLU(),
            Linear(3, 2),
        )
        x = AutogradTensor(_C.Tensor([1.0, 2.0, 3.0, 4.0], [1, 4]))
        out = model(x)
        assert out.shape == [1, 2]

"""Tests for cakelamp.optim optimizers.

Tests optimizer logic using real _C.Tensor backend and AutogradTensor.
"""

from __future__ import annotations

import math
import pytest

import cakelamp._core as _C
from cakelamp.autograd.tensor import AutogradTensor
from cakelamp.nn.parameter import Parameter
from cakelamp.optim import Optimizer, SGD, Adam


# =====================================================================
# Helper
# =====================================================================

def approx(a: list[float], b: list[float], tol: float = 1e-5) -> bool:
    """Element-wise approximate equality check."""
    return all(abs(x - y) < tol for x, y in zip(a, b))


def make_param(data: list[float]) -> Parameter:
    """Create a Parameter from a flat list."""
    return Parameter(_C.Tensor(data, [len(data)]))


def set_grad(param: Parameter, grad_data: list[float]):
    """Set gradient on a parameter."""
    param.grad = AutogradTensor(_C.Tensor(grad_data, [len(grad_data)]))


# =====================================================================
# Optimizer base class
# =====================================================================

class TestOptimizerBase:
    def test_empty_params_raises(self):
        with pytest.raises(ValueError, match="empty parameter list"):
            Optimizer([], defaults={"lr": 0.01})

    def test_single_group_from_flat_list(self):
        p1 = make_param([1.0, 2.0])
        p2 = make_param([3.0])
        opt = SGD([p1, p2], lr=0.1)
        assert len(opt.param_groups) == 1
        assert len(opt.param_groups[0]["params"]) == 2

    def test_multiple_groups(self):
        p1 = make_param([1.0])
        p2 = make_param([2.0])
        opt = SGD(
            [{"params": [p1], "lr": 0.01}, {"params": [p2], "lr": 0.1}],
            lr=0.05,
        )
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["lr"] == 0.01
        assert opt.param_groups[1]["lr"] == 0.1

    def test_duplicate_param_raises(self):
        p = make_param([1.0])
        with pytest.raises(ValueError, match="more than one parameter group"):
            SGD([{"params": [p]}, {"params": [p]}], lr=0.1)

    def test_zero_grad_set_to_none(self):
        p = make_param([1.0, 2.0])
        set_grad(p, [0.5, 0.5])
        opt = SGD([p], lr=0.1)
        opt.zero_grad(set_to_none=True)
        assert p.grad is None

    def test_step_not_implemented(self):
        p = make_param([1.0])
        opt = Optimizer([p], defaults={"lr": 0.01})
        with pytest.raises(NotImplementedError):
            opt.step()

    def test_repr(self):
        p = make_param([1.0])
        opt = SGD([p], lr=0.1, momentum=0.9)
        s = repr(opt)
        assert "SGD" in s


# =====================================================================
# SGD
# =====================================================================

class TestSGD:
    def test_basic_sgd_step(self):
        """p -= lr * grad"""
        p = make_param([2.0, 4.0])
        set_grad(p, [1.0, 2.0])
        opt = SGD([p], lr=0.1)
        opt.step()
        # 2.0 - 0.1*1.0 = 1.9,  4.0 - 0.1*2.0 = 3.8
        assert approx(p.data.tolist(), [1.9, 3.8])

    def test_sgd_no_grad_skips(self):
        p = make_param([2.0, 4.0])
        opt = SGD([p], lr=0.1)
        opt.step()
        assert p.data.tolist() == [2.0, 4.0]

    def test_sgd_weight_decay(self):
        """grad_eff = grad + weight_decay * param"""
        p = make_param([2.0, 4.0])
        set_grad(p, [1.0, 2.0])
        opt = SGD([p], lr=0.1, weight_decay=0.01)
        opt.step()
        # grad_eff = [1.0 + 0.01*2.0, 2.0 + 0.01*4.0] = [1.02, 2.04]
        # p -= 0.1 * grad_eff = [2.0 - 0.102, 4.0 - 0.204] = [1.898, 3.796]
        assert approx(p.data.tolist(), [1.898, 3.796])

    def test_sgd_momentum(self):
        """Two steps with momentum."""
        p = make_param([10.0])
        set_grad(p, [2.0])
        opt = SGD([p], lr=0.1, momentum=0.9)

        # Step 1: buf = grad = 2.0;  p = 10.0 - 0.1*2.0 = 9.8
        opt.step()
        assert approx(p.data.tolist(), [9.8])

        # Step 2: buf = 0.9*2.0 + 2.0 = 3.8;  p = 9.8 - 0.1*3.8 = 9.42
        set_grad(p, [2.0])
        opt.step()
        assert approx(p.data.tolist(), [9.42])

    def test_sgd_invalid_lr(self):
        with pytest.raises(ValueError, match="Invalid learning rate"):
            SGD([make_param([1.0])], lr=-0.1)

    def test_sgd_invalid_momentum(self):
        with pytest.raises(ValueError, match="Invalid momentum"):
            SGD([make_param([1.0])], lr=0.1, momentum=-1)

    def test_sgd_multiple_params(self):
        p1 = make_param([1.0])
        p2 = make_param([2.0])
        set_grad(p1, [0.5])
        set_grad(p2, [1.0])
        opt = SGD([p1, p2], lr=0.2)
        opt.step()
        assert approx(p1.data.tolist(), [0.9])
        assert approx(p2.data.tolist(), [1.8])

    def test_sgd_convergence(self):
        """SGD converges on f(x) = x^2, gradient = 2x."""
        p = make_param([5.0])
        opt = SGD([p], lr=0.1)
        for _ in range(50):
            set_grad(p, [2.0 * p.data.tolist()[0]])
            opt.step()
        assert abs(p.data.tolist()[0]) < 0.01


# =====================================================================
# Adam
# =====================================================================

class TestAdam:
    def test_basic_adam_step(self):
        p = make_param([5.0])
        set_grad(p, [2.0])
        opt = Adam([p], lr=0.001)
        opt.step()

        # Manual computation
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        g = 2.0
        m = (1 - beta1) * g  # 0.2
        v = (1 - beta2) * g * g  # 0.004
        m_hat = m / (1 - beta1)  # 2.0
        v_hat = v / (1 - beta2)  # 4.0
        update = 0.001 * m_hat / (math.sqrt(v_hat) + eps)
        expected = 5.0 - update
        assert approx(p.data.tolist(), [expected])

    def test_adam_no_grad_skips(self):
        p = make_param([5.0])
        opt = Adam([p], lr=0.001)
        opt.step()
        assert p.data.tolist() == [5.0]

    def test_adam_weight_decay(self):
        p = make_param([5.0])
        set_grad(p, [2.0])
        opt = Adam([p], lr=0.001, weight_decay=0.1)
        opt.step()
        assert p.data.tolist()[0] < 5.0

    def test_adam_convergence(self):
        """Adam converges on f(x) = x^2."""
        p = make_param([5.0])
        opt = Adam([p], lr=0.1)
        for _ in range(200):
            set_grad(p, [2.0 * p.data.tolist()[0]])
            opt.step()
        assert abs(p.data.tolist()[0]) < 0.1

    def test_adam_two_steps_state(self):
        p = make_param([3.0])
        set_grad(p, [1.0])
        opt = Adam([p], lr=0.001, betas=(0.9, 0.999))
        opt.step()
        state = opt.state[id(p)]
        assert state["step"] == 1

        set_grad(p, [0.5])
        opt.step()
        assert state["step"] == 2

    def test_adam_invalid_lr(self):
        with pytest.raises(ValueError, match="Invalid learning rate"):
            Adam([make_param([1.0])], lr=-0.001)

    def test_adam_invalid_beta(self):
        with pytest.raises(ValueError, match="Invalid beta"):
            Adam([make_param([1.0])], lr=0.001, betas=(1.0, 0.999))

    def test_adam_invalid_eps(self):
        with pytest.raises(ValueError, match="Invalid epsilon"):
            Adam([make_param([1.0])], lr=0.001, eps=-1e-8)

    def test_adam_multiple_params(self):
        p1 = make_param([1.0, 2.0])
        p2 = make_param([3.0])
        set_grad(p1, [0.5, 0.5])
        set_grad(p2, [1.0])
        opt = Adam([p1, p2], lr=0.01)
        opt.step()
        assert p1.data.tolist()[0] < 1.0
        assert p2.data.tolist()[0] < 3.0


# =====================================================================
# Zero grad integration
# =====================================================================

class TestZeroGradIntegration:
    @pytest.mark.parametrize("OptimizerClass,kwargs", [
        (SGD, {"lr": 0.1}),
        (Adam, {"lr": 0.001}),
    ])
    def test_zero_grad_clears_grad(self, OptimizerClass, kwargs):
        p = make_param([1.0, 2.0])
        set_grad(p, [0.5, 0.5])
        opt = OptimizerClass([p], **kwargs)
        opt.zero_grad()
        assert p.grad is None

    @pytest.mark.parametrize("OptimizerClass,kwargs", [
        (SGD, {"lr": 0.1}),
        (Adam, {"lr": 0.001}),
    ])
    def test_step_then_zero_grad(self, OptimizerClass, kwargs):
        p = make_param([1.0])
        set_grad(p, [0.5])
        opt = OptimizerClass([p], **kwargs)
        opt.step()
        opt.zero_grad()
        assert p.grad is None
        assert p.data.tolist()[0] != 1.0

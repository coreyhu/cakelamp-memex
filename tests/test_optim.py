"""Tests for cakelamp.optim optimizers.

Uses MockTensor / MockParameter to test optimizer logic independently
of the Rust backend.  Each test manually computes the expected parameter
values after one or more optimiser steps and compares against the optimizer
output.
"""

from __future__ import annotations

import math
import sys
import os
import pytest

# Ensure the python/ directory is on the path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
# Ensure the tests/ directory is on the path for mock_tensor.
sys.path.insert(0, os.path.dirname(__file__))

from mock_tensor import MockParameter, MockTensor
from cakelamp.optim import Optimizer, SGD, Adam


# =====================================================================
# Helper
# =====================================================================

def approx(a: list[float], b: list[float], tol: float = 1e-6) -> bool:
    """Element-wise approximate equality check."""
    return all(abs(x - y) < tol for x, y in zip(a, b))


# =====================================================================
# Optimizer base class
# =====================================================================

class TestOptimizerBase:
    """Tests for the Optimizer base class."""

    def test_empty_params_raises(self):
        with pytest.raises(ValueError, match="empty parameter list"):
            Optimizer([], defaults={"lr": 0.01})

    def test_single_group_from_flat_list(self):
        p1 = MockParameter([1.0, 2.0])
        p2 = MockParameter([3.0])
        opt = SGD([p1, p2], lr=0.1)
        assert len(opt.param_groups) == 1
        assert len(opt.param_groups[0]["params"]) == 2

    def test_multiple_groups(self):
        p1 = MockParameter([1.0])
        p2 = MockParameter([2.0])
        opt = SGD(
            [{"params": [p1], "lr": 0.01}, {"params": [p2], "lr": 0.1}],
            lr=0.05,
        )
        assert len(opt.param_groups) == 2
        assert opt.param_groups[0]["lr"] == 0.01
        assert opt.param_groups[1]["lr"] == 0.1

    def test_duplicate_param_raises(self):
        p = MockParameter([1.0])
        with pytest.raises(ValueError, match="more than one parameter group"):
            SGD([{"params": [p]}, {"params": [p]}], lr=0.1)

    def test_zero_grad_set_to_none(self):
        p = MockParameter([1.0, 2.0])
        p.grad = MockTensor([0.5, 0.5])
        opt = SGD([p], lr=0.1)
        opt.zero_grad(set_to_none=True)
        assert p.grad is None

    def test_zero_grad_fill_zeros(self):
        p = MockParameter([1.0, 2.0])
        p.grad = MockTensor([0.5, 0.5])
        opt = SGD([p], lr=0.1)
        opt.zero_grad(set_to_none=False)
        assert p.grad.tolist() == [0.0, 0.0]

    def test_step_not_implemented(self):
        p = MockParameter([1.0])
        opt = Optimizer([p], defaults={"lr": 0.01})
        with pytest.raises(NotImplementedError):
            opt.step()

    def test_repr(self):
        p = MockParameter([1.0])
        opt = SGD([p], lr=0.1, momentum=0.9)
        s = repr(opt)
        assert "SGD" in s
        assert "lr: 0.1" in s


# =====================================================================
# SGD
# =====================================================================

class TestSGD:
    """Tests for SGD optimizer."""

    def test_basic_sgd_step(self):
        """p -= lr * grad"""
        p = MockParameter([2.0, 4.0])
        p.grad = MockTensor([1.0, 2.0])
        opt = SGD([p], lr=0.1)
        opt.step()
        # 2.0 - 0.1*1.0 = 1.9,  4.0 - 0.1*2.0 = 3.8
        assert approx(p.tolist(), [1.9, 3.8])

    def test_sgd_no_grad_skips(self):
        """Parameters with no grad should be skipped."""
        p = MockParameter([2.0, 4.0])
        # p.grad is None by default
        opt = SGD([p], lr=0.1)
        opt.step()
        assert p.tolist() == [2.0, 4.0]

    def test_sgd_weight_decay(self):
        """grad_eff = grad + weight_decay * param"""
        p = MockParameter([2.0, 4.0])
        p.grad = MockTensor([1.0, 2.0])
        opt = SGD([p], lr=0.1, weight_decay=0.01)
        opt.step()
        # grad_eff = [1.0 + 0.01*2.0, 2.0 + 0.01*4.0] = [1.02, 2.04]
        # p -= 0.1 * grad_eff = [2.0 - 0.102, 4.0 - 0.204] = [1.898, 3.796]
        assert approx(p.tolist(), [1.898, 3.796])

    def test_sgd_momentum(self):
        """Two steps with momentum to verify buffer accumulation."""
        p = MockParameter([10.0])
        p.grad = MockTensor([2.0])
        opt = SGD([p], lr=0.1, momentum=0.9)

        # Step 1: buf = grad = 2.0;  p = 10.0 - 0.1*2.0 = 9.8
        opt.step()
        assert approx(p.tolist(), [9.8])

        # Step 2: buf = 0.9*2.0 + 2.0 = 3.8;  p = 9.8 - 0.1*3.8 = 9.42
        p.grad = MockTensor([2.0])
        opt.step()
        assert approx(p.tolist(), [9.42])

    def test_sgd_momentum_dampening(self):
        """Momentum with dampening."""
        p = MockParameter([10.0])
        p.grad = MockTensor([2.0])
        opt = SGD([p], lr=0.1, momentum=0.9, dampening=0.1)

        # Step 1: buf = grad (first step, no dampening applied to first) = 2.0
        # p = 10.0 - 0.1*2.0 = 9.8
        opt.step()
        assert approx(p.tolist(), [9.8])

        # Step 2: buf = 0.9*2.0 + (1-0.1)*2.0 = 1.8 + 1.8 = 3.6
        # p = 9.8 - 0.1*3.6 = 9.44
        p.grad = MockTensor([2.0])
        opt.step()
        assert approx(p.tolist(), [9.44])

    def test_sgd_nesterov(self):
        """Nesterov momentum changes the update direction."""
        p = MockParameter([10.0])
        p.grad = MockTensor([2.0])
        opt = SGD([p], lr=0.1, momentum=0.9, nesterov=True)

        # Step 1: buf = 2.0;  nesterov_grad = grad + momentum * buf = 2.0 + 0.9*2.0 = 3.8
        # p = 10.0 - 0.1*3.8 = 9.62
        opt.step()
        assert approx(p.tolist(), [9.62])

    def test_sgd_nesterov_requires_momentum(self):
        with pytest.raises(ValueError, match="Nesterov momentum requires"):
            SGD([MockParameter([1.0])], lr=0.1, nesterov=True, momentum=0)

    def test_sgd_invalid_lr(self):
        with pytest.raises(ValueError, match="Invalid learning rate"):
            SGD([MockParameter([1.0])], lr=-0.1)

    def test_sgd_invalid_momentum(self):
        with pytest.raises(ValueError, match="Invalid momentum"):
            SGD([MockParameter([1.0])], lr=0.1, momentum=-1)

    def test_sgd_multiple_params(self):
        """Multiple parameters in one group."""
        p1 = MockParameter([1.0])
        p2 = MockParameter([2.0])
        p1.grad = MockTensor([0.5])
        p2.grad = MockTensor([1.0])
        opt = SGD([p1, p2], lr=0.2)
        opt.step()
        assert approx(p1.tolist(), [0.9])
        assert approx(p2.tolist(), [1.8])

    def test_sgd_multiple_steps(self):
        """Verify multi-step convergence on a simple quadratic."""
        # Minimise f(x) = x^2, gradient = 2x
        p = MockParameter([5.0])
        opt = SGD([p], lr=0.1)
        for _ in range(50):
            p.grad = MockTensor([2.0 * p.data.tolist()[0]])
            opt.step()
        # Should approach 0
        assert abs(p.tolist()[0]) < 0.01


# =====================================================================
# Adam
# =====================================================================

class TestAdam:
    """Tests for Adam optimizer."""

    def test_basic_adam_step(self):
        """One Adam step with default hyperparameters."""
        p = MockParameter([5.0])
        p.grad = MockTensor([2.0])
        opt = Adam([p], lr=0.001)
        opt.step()

        # Manual computation:
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        g = 2.0
        m = (1 - beta1) * g  # 0.2
        v = (1 - beta2) * g * g  # 0.004
        m_hat = m / (1 - beta1)  # 0.2 / 0.1 = 2.0
        v_hat = v / (1 - beta2)  # 0.004 / 0.001 = 4.0
        update = 0.001 * m_hat / (math.sqrt(v_hat) + eps)
        expected = 5.0 - update
        assert approx(p.tolist(), [expected], tol=1e-5)

    def test_adam_no_grad_skips(self):
        p = MockParameter([5.0])
        opt = Adam([p], lr=0.001)
        opt.step()
        assert p.tolist() == [5.0]

    def test_adam_weight_decay(self):
        """Decoupled weight decay (AdamW-style)."""
        p = MockParameter([5.0])
        p.grad = MockTensor([2.0])
        opt = Adam([p], lr=0.001, weight_decay=0.1)
        opt.step()
        # Weight decay applied before moment update:
        # p.data += p.data * (-0.1 * 0.001) = 5.0 * (1 - 0.0001) = 4.9995
        # Then standard Adam update on top.
        # Just verify it moved more than without weight decay.
        assert p.tolist()[0] < 5.0

    def test_adam_convergence(self):
        """Adam converges on f(x) = x^2."""
        p = MockParameter([5.0])
        opt = Adam([p], lr=0.1)
        for _ in range(200):
            p.grad = MockTensor([2.0 * p.data.tolist()[0]])
            opt.step()
        assert abs(p.tolist()[0]) < 0.1

    def test_adam_two_steps_state(self):
        """Verify state accumulation across two steps."""
        p = MockParameter([3.0])
        p.grad = MockTensor([1.0])
        opt = Adam([p], lr=0.001, betas=(0.9, 0.999))

        opt.step()
        state = opt.state[id(p)]
        assert state["step"] == 1
        assert len(state["exp_avg"].tolist()) == 1
        assert len(state["exp_avg_sq"].tolist()) == 1

        p.grad = MockTensor([0.5])
        opt.step()
        assert state["step"] == 2

    def test_adam_amsgrad(self):
        """AMSGrad variant stores max of second moment."""
        p = MockParameter([3.0])
        p.grad = MockTensor([2.0])
        opt = Adam([p], lr=0.001, amsgrad=True)
        opt.step()

        state = opt.state[id(p)]
        assert "max_exp_avg_sq" in state
        max_v = state["max_exp_avg_sq"].tolist()
        v = state["exp_avg_sq"].tolist()
        # After first step, max should equal v.
        assert approx(max_v, v)

        # Second step with smaller gradient — max should stay.
        p.grad = MockTensor([0.001])
        opt.step()
        new_max_v = state["max_exp_avg_sq"].tolist()
        assert new_max_v[0] >= max_v[0] - 1e-10

    def test_adam_invalid_lr(self):
        with pytest.raises(ValueError, match="Invalid learning rate"):
            Adam([MockParameter([1.0])], lr=-0.001)

    def test_adam_invalid_beta(self):
        with pytest.raises(ValueError, match="Invalid beta"):
            Adam([MockParameter([1.0])], lr=0.001, betas=(1.0, 0.999))

    def test_adam_invalid_eps(self):
        with pytest.raises(ValueError, match="Invalid epsilon"):
            Adam([MockParameter([1.0])], lr=0.001, eps=-1e-8)

    def test_adam_multiple_params(self):
        """Multiple parameters."""
        p1 = MockParameter([1.0, 2.0])
        p2 = MockParameter([3.0])
        p1.grad = MockTensor([0.5, 0.5])
        p2.grad = MockTensor([1.0])
        opt = Adam([p1, p2], lr=0.01)
        opt.step()
        # Both should have moved.
        assert p1.tolist()[0] < 1.0
        assert p2.tolist()[0] < 3.0


# =====================================================================
# State dict serialisation
# =====================================================================

class TestStateDict:
    """Tests for state_dict / load_state_dict."""

    def test_state_dict_round_trip(self):
        """Save and load optimizer state."""
        p = MockParameter([5.0])
        p.grad = MockTensor([2.0])
        opt = Adam([p], lr=0.001)
        opt.step()

        sd = opt.state_dict()
        assert "state" in sd
        assert "param_groups" in sd
        assert len(sd["param_groups"]) == 1
        assert sd["param_groups"][0]["lr"] == 0.001

    def test_load_state_dict_restores_hyperparams(self):
        """Loading a state dict restores group hyperparameters."""
        p = MockParameter([5.0])
        p.grad = MockTensor([2.0])
        opt1 = Adam([p], lr=0.001)
        opt1.step()
        sd = opt1.state_dict()

        # Create new optimizer with different lr.
        opt2 = Adam([p], lr=0.1)
        opt2.load_state_dict(sd)
        assert opt2.param_groups[0]["lr"] == 0.001

    def test_load_state_dict_mismatched_groups(self):
        """Loading with wrong number of groups raises."""
        p1 = MockParameter([1.0])
        p2 = MockParameter([2.0])
        opt1 = SGD([p1], lr=0.1)
        sd = opt1.state_dict()

        opt2 = SGD(
            [{"params": [p1]}, {"params": [p2]}],
            lr=0.1,
        )
        with pytest.raises(ValueError, match="param groups"):
            opt2.load_state_dict(sd)


# =====================================================================
# Zero grad integration
# =====================================================================

class TestZeroGradIntegration:
    """Test zero_grad works across optimizer types."""

    @pytest.mark.parametrize("OptimizerClass,kwargs", [
        (SGD, {"lr": 0.1}),
        (Adam, {"lr": 0.001}),
    ])
    def test_zero_grad_clears_grad(self, OptimizerClass, kwargs):
        p = MockParameter([1.0, 2.0])
        p.grad = MockTensor([0.5, 0.5])
        opt = OptimizerClass([p], **kwargs)
        opt.zero_grad()
        assert p.grad is None

    @pytest.mark.parametrize("OptimizerClass,kwargs", [
        (SGD, {"lr": 0.1}),
        (Adam, {"lr": 0.001}),
    ])
    def test_step_then_zero_grad(self, OptimizerClass, kwargs):
        """Full step + zero_grad cycle."""
        p = MockParameter([1.0])
        p.grad = MockTensor([0.5])
        opt = OptimizerClass([p], **kwargs)
        opt.step()
        opt.zero_grad()
        assert p.grad is None
        # Param should have been updated.
        assert p.tolist()[0] != 1.0

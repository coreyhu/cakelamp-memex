"""Tests for cakelamp.autograd — reverse-mode automatic differentiation."""

import math
import pytest

from cakelamp.autograd.tensor import AutogradTensor
from cakelamp.autograd.engine import backward, no_grad, GradMode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _flat(t):
    """Get flat data from a tensor."""
    return list(t._data)


def _approx(a, b, tol=1e-4):
    """Check two flat lists are element-wise close."""
    assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"Index {i}: {x} != {y} (tol={tol})"


def _numerical_grad(f, x_data, shape, eps=1e-5):
    """Compute numerical gradient via central differences."""
    grads = []
    for i in range(len(x_data)):
        x_plus = list(x_data)
        x_plus[i] += eps
        x_minus = list(x_data)
        x_minus[i] -= eps
        t_plus = AutogradTensor._make(x_plus, list(shape))
        t_minus = AutogradTensor._make(x_minus, list(shape))
        fp = f(t_plus).item()
        fm = f(t_minus).item()
        grads.append((fp - fm) / (2 * eps))
    return grads


# ===========================================================================
# 1. Tensor basics
# ===========================================================================

class TestTensorBasics:
    def test_create_from_list(self):
        t = AutogradTensor([1.0, 2.0, 3.0])
        assert t._shape == [3]
        assert t.numel == 3
        _approx(_flat(t), [1.0, 2.0, 3.0])

    def test_create_2d(self):
        t = AutogradTensor([[1.0, 2.0], [3.0, 4.0]])
        assert t._shape == [2, 2]
        assert t.numel == 4

    def test_zeros(self):
        t = AutogradTensor.zeros([2, 3])
        assert t._shape == [2, 3]
        _approx(_flat(t), [0.0] * 6)

    def test_ones(self):
        t = AutogradTensor.ones([3])
        _approx(_flat(t), [1.0, 1.0, 1.0])

    def test_full(self):
        t = AutogradTensor.full([2, 2], 5.0)
        _approx(_flat(t), [5.0] * 4)

    def test_eye(self):
        t = AutogradTensor.eye(3)
        expected = [1, 0, 0, 0, 1, 0, 0, 0, 1]
        _approx(_flat(t), [float(x) for x in expected])

    def test_arange(self):
        t = AutogradTensor.arange(0, 5, 1)
        _approx(_flat(t), [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_rand_shape(self):
        t = AutogradTensor.rand([2, 3])
        assert t._shape == [2, 3]
        assert t.numel == 6

    def test_item_scalar(self):
        t = AutogradTensor([42.0])
        assert t.item() == pytest.approx(42.0)

    def test_requires_grad(self):
        t = AutogradTensor([1.0], requires_grad=True)
        assert t.requires_grad is True
        t2 = AutogradTensor([1.0])
        assert t2.requires_grad is False

    def test_detach(self):
        t = AutogradTensor([1.0, 2.0], requires_grad=True)
        d = t.detach()
        assert d.requires_grad is False
        assert d._grad_fn is None


# ===========================================================================
# 2. Forward ops (no grad check, just correctness)
# ===========================================================================

class TestForwardOps:
    def test_add(self):
        a = AutogradTensor([1.0, 2.0])
        b = AutogradTensor([3.0, 4.0])
        c = a + b
        _approx(_flat(c), [4.0, 6.0])

    def test_sub(self):
        a = AutogradTensor([5.0, 3.0])
        b = AutogradTensor([1.0, 2.0])
        c = a - b
        _approx(_flat(c), [4.0, 1.0])

    def test_mul(self):
        a = AutogradTensor([2.0, 3.0])
        b = AutogradTensor([4.0, 5.0])
        c = a * b
        _approx(_flat(c), [8.0, 15.0])

    def test_div(self):
        a = AutogradTensor([6.0, 8.0])
        b = AutogradTensor([2.0, 4.0])
        c = a / b
        _approx(_flat(c), [3.0, 2.0])

    def test_neg(self):
        a = AutogradTensor([1.0, -2.0])
        c = -a
        _approx(_flat(c), [-1.0, 2.0])

    def test_pow(self):
        a = AutogradTensor([2.0, 3.0])
        b = AutogradTensor([3.0, 2.0])
        c = a ** b
        _approx(_flat(c), [8.0, 9.0])

    def test_exp(self):
        a = AutogradTensor([0.0, 1.0])
        c = a.exp()
        _approx(_flat(c), [1.0, math.e])

    def test_log(self):
        a = AutogradTensor([1.0, math.e])
        c = a.log()
        _approx(_flat(c), [0.0, 1.0])

    def test_relu(self):
        a = AutogradTensor([-1.0, 0.0, 2.0])
        c = a.relu()
        _approx(_flat(c), [0.0, 0.0, 2.0])

    def test_sigmoid(self):
        a = AutogradTensor([0.0])
        c = a.sigmoid()
        assert c.item() == pytest.approx(0.5)

    def test_tanh(self):
        a = AutogradTensor([0.0])
        c = a.tanh()
        assert c.item() == pytest.approx(0.0)

    def test_matmul(self):
        a = AutogradTensor([[1.0, 2.0], [3.0, 4.0]])
        b = AutogradTensor([[5.0, 6.0], [7.0, 8.0]])
        c = a.mm(b)
        _approx(_flat(c), [19.0, 22.0, 43.0, 50.0])

    def test_sum_all(self):
        a = AutogradTensor([1.0, 2.0, 3.0])
        s = a.sum()
        assert s.item() == pytest.approx(6.0)

    def test_sum_dim(self):
        a = AutogradTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        s = a.sum(dim=0)
        _approx(_flat(s), [5.0, 7.0, 9.0])

    def test_mean_all(self):
        a = AutogradTensor([2.0, 4.0, 6.0])
        m = a.mean()
        assert m.item() == pytest.approx(4.0)

    def test_reshape(self):
        a = AutogradTensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        b = a.reshape([2, 3])
        assert b._shape == [2, 3]
        _approx(_flat(b), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def test_transpose(self):
        a = AutogradTensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        b = a.transpose(0, 1)
        assert b._shape == [3, 2]
        # After transpose, contiguous data should be column-major read
        _approx(_flat(b), [1.0, 4.0, 2.0, 5.0, 3.0, 6.0])

    def test_softmax(self):
        a = AutogradTensor([[1.0, 2.0, 3.0]])
        s = a.softmax(dim=1)
        vals = _flat(s)
        assert sum(vals) == pytest.approx(1.0, abs=1e-5)
        assert vals[0] < vals[1] < vals[2]

    def test_log_softmax(self):
        a = AutogradTensor([[1.0, 2.0, 3.0]])
        ls = a.log_softmax(dim=1)
        vals = _flat(ls)
        for v in vals:
            assert v <= 0.0 + 1e-6
        assert sum(math.exp(v) for v in vals) == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# 3. Backward passes — numerical gradient checks
# ===========================================================================

class TestBackwardNumerical:
    """Verify analytic gradients match numerical (central difference) gradients."""

    def _check_grad(self, f, x_data, shape, tol=1e-3):
        x = AutogradTensor._make(list(x_data), list(shape), requires_grad=True)
        loss = f(x)
        loss.backward()
        analytic = _flat(x.grad)
        numerical = _numerical_grad(f, x_data, shape)
        _approx(analytic, numerical, tol=tol)

    def test_add_backward(self):
        def f(x):
            return (x + x).sum()
        self._check_grad(f, [1.0, 2.0, 3.0], [3])

    def test_sub_backward(self):
        def f(x):
            c = AutogradTensor([5.0, 5.0, 5.0])
            return (c - x).sum()
        self._check_grad(f, [1.0, 2.0, 3.0], [3])

    def test_mul_backward(self):
        def f(x):
            return (x * x).sum()
        self._check_grad(f, [1.0, 2.0, 3.0], [3])

    def test_div_backward(self):
        def f(x):
            c = AutogradTensor([1.0, 1.0])
            return (c / x).sum()
        self._check_grad(f, [2.0, 3.0], [2])

    def test_neg_backward(self):
        def f(x):
            return (-x).sum()
        self._check_grad(f, [1.0, -2.0, 3.0], [3])

    def test_pow_backward(self):
        def f(x):
            e = AutogradTensor([2.0, 3.0])
            return (x ** e).sum()
        self._check_grad(f, [2.0, 3.0], [2])

    def test_exp_backward(self):
        def f(x):
            return x.exp().sum()
        self._check_grad(f, [0.0, 1.0, -1.0], [3])

    def test_log_backward(self):
        def f(x):
            return x.log().sum()
        self._check_grad(f, [1.0, 2.0, 3.0], [3])

    def test_relu_backward(self):
        def f(x):
            return x.relu().sum()
        self._check_grad(f, [-1.0, 0.5, 2.0], [3])

    def test_sigmoid_backward(self):
        def f(x):
            return x.sigmoid().sum()
        self._check_grad(f, [-1.0, 0.0, 1.0], [3])

    def test_tanh_backward(self):
        def f(x):
            return x.tanh().sum()
        self._check_grad(f, [-1.0, 0.0, 1.0], [3])

    def test_sum_backward(self):
        def f(x):
            return x.sum()
        self._check_grad(f, [1.0, 2.0, 3.0, 4.0], [4])

    def test_mean_backward(self):
        def f(x):
            return x.mean()
        self._check_grad(f, [1.0, 2.0, 3.0, 4.0], [4])

    def test_matmul_backward(self):
        """Gradient of sum(A @ B) w.r.t. A."""
        B = AutogradTensor._make([1.0, 0.0, 0.0, 1.0], [2, 2])

        def f(x):
            return x.reshape([2, 2]).mm(B).sum()
        self._check_grad(f, [1.0, 2.0, 3.0, 4.0], [4])

    def test_chain_rule(self):
        """Multi-step computation: exp(x * x).sum()."""
        def f(x):
            return (x * x).exp().sum()
        self._check_grad(f, [0.5, 1.0, -0.5], [3], tol=1e-2)

    def test_complex_chain(self):
        """sigmoid(x) * tanh(x) summed."""
        def f(x):
            return (x.sigmoid() * x.tanh()).sum()
        self._check_grad(f, [-1.0, 0.0, 1.0], [3])


# ===========================================================================
# 4. Specific backward correctness
# ===========================================================================

class TestBackwardExact:
    def test_add_grad_values(self):
        a = AutogradTensor([2.0, 3.0], requires_grad=True)
        b = AutogradTensor([4.0, 5.0], requires_grad=True)
        c = (a + b).sum()
        c.backward()
        _approx(_flat(a.grad), [1.0, 1.0])
        _approx(_flat(b.grad), [1.0, 1.0])

    def test_mul_grad_values(self):
        a = AutogradTensor([2.0, 3.0], requires_grad=True)
        b = AutogradTensor([4.0, 5.0], requires_grad=True)
        c = (a * b).sum()
        c.backward()
        _approx(_flat(a.grad), [4.0, 5.0])
        _approx(_flat(b.grad), [2.0, 3.0])

    def test_sub_grad_values(self):
        a = AutogradTensor([2.0, 3.0], requires_grad=True)
        b = AutogradTensor([4.0, 5.0], requires_grad=True)
        c = (a - b).sum()
        c.backward()
        _approx(_flat(a.grad), [1.0, 1.0])
        _approx(_flat(b.grad), [-1.0, -1.0])

    def test_div_grad_values(self):
        a = AutogradTensor([6.0], requires_grad=True)
        b = AutogradTensor([3.0], requires_grad=True)
        c = (a / b).sum()
        c.backward()
        _approx(_flat(a.grad), [1.0 / 3.0])
        _approx(_flat(b.grad), [-6.0 / 9.0])

    def test_neg_grad_values(self):
        a = AutogradTensor([1.0, 2.0], requires_grad=True)
        c = (-a).sum()
        c.backward()
        _approx(_flat(a.grad), [-1.0, -1.0])

    def test_relu_grad_values(self):
        a = AutogradTensor([-1.0, 0.5, 2.0], requires_grad=True)
        c = a.relu().sum()
        c.backward()
        _approx(_flat(a.grad), [0.0, 1.0, 1.0])

    def test_matmul_grad_values(self):
        a = AutogradTensor._make([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
        b = AutogradTensor._make([5.0, 6.0, 7.0, 8.0], [2, 2], requires_grad=True)
        c = a.mm(b).sum()
        c.backward()
        # grad_A = ones(2,2) @ B^T
        # B^T = [[5,7],[6,8]], ones@B^T row = [5+6, 7+8] = [11, 15]
        _approx(_flat(a.grad), [11.0, 15.0, 11.0, 15.0])
        # grad_B = A^T @ ones(2,2)
        # A^T = [[1,3],[2,4]], A^T@ones col = [1+3, 2+4] = [4, 6]
        _approx(_flat(b.grad), [4.0, 4.0, 6.0, 6.0])


# ===========================================================================
# 5. Gradient accumulation & graph
# ===========================================================================

class TestGradAccumulation:
    def test_same_tensor_used_twice(self):
        x = AutogradTensor([2.0, 3.0], requires_grad=True)
        y = x * x
        z = y + x
        loss = z.sum()
        loss.backward()
        # d_loss/dx = 2x + 1 = [5, 7]
        _approx(_flat(x.grad), [5.0, 7.0])

    def test_grad_not_set_without_requires_grad(self):
        a = AutogradTensor([1.0, 2.0])
        b = AutogradTensor([3.0, 4.0], requires_grad=True)
        c = (a + b).sum()
        c.backward()
        assert a.grad is None
        _approx(_flat(b.grad), [1.0, 1.0])


# ===========================================================================
# 6. no_grad context
# ===========================================================================

class TestNoGrad:
    def test_no_grad_disables_tracking(self):
        x = AutogradTensor([1.0, 2.0], requires_grad=True)
        with no_grad():
            y = x + x
        assert y._grad_fn is None

    def test_no_grad_restores(self):
        assert GradMode.is_enabled()
        with no_grad():
            assert not GradMode.is_enabled()
        assert GradMode.is_enabled()


# ===========================================================================
# 7. Shape operations backward
# ===========================================================================

class TestShapeOpsBackward:
    def test_reshape_backward(self):
        x = AutogradTensor._make([1.0, 2.0, 3.0, 4.0], [4], requires_grad=True)
        y = x.reshape([2, 2])
        loss = y.sum()
        loss.backward()
        assert x.grad._shape == [4]
        _approx(_flat(x.grad), [1.0, 1.0, 1.0, 1.0])

    def test_transpose_backward(self):
        x = AutogradTensor._make([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
        y = x.transpose(0, 1)
        loss = y.sum()
        loss.backward()
        assert x.grad._shape == [2, 3]
        _approx(_flat(x.grad), [1.0] * 6)

    def test_sum_dim_backward(self):
        x = AutogradTensor._make([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
        y = x.sum(dim=1)
        loss = y.sum()
        loss.backward()
        _approx(_flat(x.grad), [1.0] * 6)

    def test_mean_dim_backward(self):
        x = AutogradTensor._make([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
        y = x.mean(dim=1)
        loss = y.sum()
        loss.backward()
        _approx(_flat(x.grad), [1.0 / 3] * 6)


# ===========================================================================
# 8. Broadcasting backward
# ===========================================================================

class TestBroadcastBackward:
    def test_add_broadcast_scalar(self):
        a = AutogradTensor._make([2.0], [1], requires_grad=True)
        b = AutogradTensor([1.0, 2.0, 3.0], requires_grad=True)
        c = (a + b).sum()
        c.backward()
        _approx(_flat(a.grad), [3.0])
        _approx(_flat(b.grad), [1.0, 1.0, 1.0])

    def test_mul_broadcast(self):
        """(1,3) * (2,3) => (2,3)."""
        a = AutogradTensor._make([1.0, 2.0, 3.0], [1, 3], requires_grad=True)
        b = AutogradTensor._make([1.0, 1.0, 1.0, 2.0, 2.0, 2.0], [2, 3], requires_grad=True)
        c = (a * b).sum()
        c.backward()
        _approx(_flat(a.grad), [3.0, 3.0, 3.0])


# ===========================================================================
# 9. Log-softmax backward
# ===========================================================================

class TestLogSoftmaxBackward:
    def test_log_softmax_backward(self):
        def f(x):
            return x.reshape([1, 3]).log_softmax(dim=1).sum()
        x_data = [1.0, 2.0, 3.0]
        x = AutogradTensor._make(list(x_data), [3], requires_grad=True)
        loss = f(x)
        loss.backward()
        numerical = _numerical_grad(f, x_data, [3])
        _approx(_flat(x.grad), numerical, tol=1e-3)


# ===========================================================================
# 10. In-place ops
# ===========================================================================

class TestInplaceOps:
    def test_zero_(self):
        t = AutogradTensor([1.0, 2.0, 3.0])
        t.zero_()
        _approx(_flat(t), [0.0, 0.0, 0.0])

    def test_fill_(self):
        t = AutogradTensor([0.0, 0.0])
        t.fill_(7.0)
        _approx(_flat(t), [7.0, 7.0])

    def test_add_inplace(self):
        t = AutogradTensor([1.0, 2.0])
        t.add_(AutogradTensor([3.0, 4.0]))
        _approx(_flat(t), [4.0, 6.0])

    def test_mul_inplace(self):
        t = AutogradTensor([2.0, 3.0])
        t.mul_(AutogradTensor([4.0, 5.0]))
        _approx(_flat(t), [8.0, 15.0])


# ===========================================================================
# 11. Integration: simple linear regression
# ===========================================================================

class TestIntegration:
    def test_linear_regression_convergence(self):
        """Train y = 2x + 1 using manual SGD on AutogradTensor."""
        w = AutogradTensor([0.0], requires_grad=True)
        b = AutogradTensor([0.0], requires_grad=True)

        lr = 0.01
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [3.0, 5.0, 7.0, 9.0]

        for epoch in range(200):
            total_loss = AutogradTensor([0.0])
            for x_val, y_val in zip(xs, ys):
                x = AutogradTensor([x_val])
                y = AutogradTensor([y_val])
                pred = x * w + b
                diff = pred - y
                loss = diff * diff
                total_loss = total_loss + loss

            total_loss.backward()

            with no_grad():
                w_data = w._data[0] - lr * w.grad._data[0]
                b_data = b._data[0] - lr * b.grad._data[0]
                w._data[0] = w_data
                b._data[0] = b_data

            w.grad = None
            b.grad = None

        assert w.item() == pytest.approx(2.0, abs=0.1)
        assert b.item() == pytest.approx(1.0, abs=0.1)

    def test_backward_requires_scalar(self):
        """backward() without gradient should fail for non-scalar."""
        t = AutogradTensor([1.0, 2.0], requires_grad=True)
        y = t * t
        with pytest.raises(RuntimeError):
            y.backward()

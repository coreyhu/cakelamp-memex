"""Tests for CakeLamp autograd (reverse-mode automatic differentiation)."""
import math
from cakelamp.autograd.tensor import AutogradTensor


def approx(a, b, tol=1e-4):
    return abs(a - b) < tol


def approx_list(a, b, tol=1e-4):
    assert len(a) == len(b), f"Length mismatch: {len(a)} != {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        assert abs(x - y) < tol, f"Mismatch at [{i}]: {x} != {y} (diff={abs(x-y)})"


class TestBasicGradients:
    def test_add_backward(self):
        a = AutogradTensor.from_data([2.0, 3.0], [2], requires_grad=True)
        b = AutogradTensor.from_data([4.0, 5.0], [2], requires_grad=True)
        c = a + b
        loss = c.sum()
        loss.backward()
        approx_list(a.grad.tolist(), [1.0, 1.0])
        approx_list(b.grad.tolist(), [1.0, 1.0])

    def test_sub_backward(self):
        a = AutogradTensor.from_data([2.0, 3.0], [2], requires_grad=True)
        b = AutogradTensor.from_data([4.0, 5.0], [2], requires_grad=True)
        c = a - b
        loss = c.sum()
        loss.backward()
        approx_list(a.grad.tolist(), [1.0, 1.0])
        approx_list(b.grad.tolist(), [-1.0, -1.0])

    def test_mul_backward(self):
        a = AutogradTensor.from_data([2.0, 3.0], [2], requires_grad=True)
        b = AutogradTensor.from_data([4.0, 5.0], [2], requires_grad=True)
        c = a * b
        loss = c.sum()
        loss.backward()
        approx_list(a.grad.tolist(), [4.0, 5.0])
        approx_list(b.grad.tolist(), [2.0, 3.0])

    def test_neg_backward(self):
        a = AutogradTensor.from_data([2.0, 3.0], [2], requires_grad=True)
        c = -a
        loss = c.sum()
        loss.backward()
        approx_list(a.grad.tolist(), [-1.0, -1.0])


class TestChainedGradients:
    def test_chain_add_mul(self):
        a = AutogradTensor.from_data([2.0], [1], requires_grad=True)
        b = AutogradTensor.from_data([3.0], [1], requires_grad=True)
        c = a + b
        d = a * c  # d = a*(a+b) = a^2 + ab
        loss = d.sum()
        loss.backward()
        assert approx(a.grad.tolist()[0], 7.0)  # 2a + b = 7
        assert approx(b.grad.tolist()[0], 2.0)  # a = 2

    def test_linear_layer(self):
        x = AutogradTensor.from_data([1.0, 2.0], [1, 2], requires_grad=False)
        W = AutogradTensor.from_data([0.5, 0.3, 0.2, 0.4], [2, 2], requires_grad=True)
        b = AutogradTensor.from_data([0.1, 0.1], [1, 2], requires_grad=True)
        y = x.matmul(W) + b
        loss = y.sum()
        loss.backward()
        approx_list(W.grad.tolist(), [1.0, 1.0, 2.0, 2.0])
        approx_list(b.grad.tolist(), [1.0, 1.0])


class TestActivationGradients:
    def test_relu_backward(self):
        a = AutogradTensor.from_data([-1.0, 0.5, 2.0], [3], requires_grad=True)
        r = a.relu()
        loss = r.sum()
        loss.backward()
        approx_list(a.grad.tolist(), [0.0, 1.0, 1.0])

    def test_exp_backward(self):
        a = AutogradTensor.from_data([0.0, 1.0], [2], requires_grad=True)
        r = a.exp()
        loss = r.sum()
        loss.backward()
        assert approx(a.grad.tolist()[0], 1.0)
        assert approx(a.grad.tolist()[1], math.e)

    def test_log_backward(self):
        a = AutogradTensor.from_data([1.0, 2.0, 4.0], [3], requires_grad=True)
        r = a.log()
        loss = r.sum()
        loss.backward()
        approx_list(a.grad.tolist(), [1.0, 0.5, 0.25])

    def test_sigmoid_backward(self):
        a = AutogradTensor.from_data([0.0], [1], requires_grad=True)
        r = a.sigmoid()
        loss = r.sum()
        loss.backward()
        assert approx(a.grad.tolist()[0], 0.25)

    def test_tanh_backward(self):
        a = AutogradTensor.from_data([0.0], [1], requires_grad=True)
        r = a.tanh()
        loss = r.sum()
        loss.backward()
        assert approx(a.grad.tolist()[0], 1.0)


class TestReductionGradients:
    def test_sum_backward(self):
        a = AutogradTensor.from_data([1.0, 2.0, 3.0], [3], requires_grad=True)
        loss = a.sum()
        loss.backward()
        approx_list(a.grad.tolist(), [1.0, 1.0, 1.0])

    def test_mean_backward(self):
        a = AutogradTensor.from_data([1.0, 2.0, 3.0, 4.0], [4], requires_grad=True)
        loss = a.mean()
        loss.backward()
        approx_list(a.grad.tolist(), [0.25, 0.25, 0.25, 0.25])


class TestMatmulGradients:
    def test_matmul_backward(self):
        A = AutogradTensor.from_data([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
        B = AutogradTensor.from_data([5.0, 6.0, 7.0, 8.0], [2, 2], requires_grad=True)
        C = A.matmul(B)
        loss = C.sum()
        loss.backward()
        # dL/dA = ones(2,2) @ B^T = [[5,7],[6,8]] sum cols => [[11,15],[11,15]]
        approx_list(A.grad.tolist(), [11.0, 15.0, 11.0, 15.0])
        # dL/dB = A^T @ ones(2,2) = [[1,3],[2,4]] sum rows => [[4,4],[6,6]]
        approx_list(B.grad.tolist(), [4.0, 4.0, 6.0, 6.0])


class TestViewGradients:
    def test_reshape_backward(self):
        a = AutogradTensor.from_data([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=True)
        b = a.reshape([4])
        loss = b.sum()
        loss.backward()
        approx_list(a.grad.tolist(), [1.0, 1.0, 1.0, 1.0])

    def test_transpose_backward(self):
        a = AutogradTensor.from_data([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=True)
        b = a.t()
        loss = b.sum()
        loss.backward()
        approx_list(a.grad.tolist(), [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])


class TestNoGrad:
    def test_no_grad_tensors(self):
        a = AutogradTensor.from_data([1.0, 2.0], [2], requires_grad=False)
        b = AutogradTensor.from_data([3.0, 4.0], [2], requires_grad=True)
        c = a + b
        loss = c.sum()
        loss.backward()
        assert a.grad is None
        assert b.grad is not None

    def test_detach(self):
        a = AutogradTensor.from_data([1.0, 2.0], [2], requires_grad=True)
        b = a.detach()
        assert not b.requires_grad
        assert b.grad_fn is None


class TestZeroGrad:
    def test_zero_grad(self):
        a = AutogradTensor.from_data([1.0, 2.0], [2], requires_grad=True)
        b = a * AutogradTensor.from_data([3.0, 4.0], [2])
        loss = b.sum()
        loss.backward()
        assert a.grad is not None
        a.zero_grad()
        assert a.grad is None


class TestMLP:
    def test_simple_mlp_backward(self):
        x = AutogradTensor.from_data([1.0, 2.0], [1, 2])
        W1 = AutogradTensor.from_data([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3], requires_grad=True)
        b1 = AutogradTensor.from_data([0.0, 0.0, 0.0], [1, 3], requires_grad=True)
        W2 = AutogradTensor.from_data([0.7, 0.8, 0.9], [3, 1], requires_grad=True)
        b2 = AutogradTensor.from_data([0.0], [1, 1], requires_grad=True)

        h = x.matmul(W1) + b1
        h = h.relu()
        out = h.matmul(W2) + b2
        loss = out.sum()
        loss.backward()

        for name, param in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
            assert param.grad is not None, f"{name} should have grad"
            assert all(math.isfinite(v) for v in param.grad.tolist()), f"{name} grad not finite"

"""Tests for autograd (reverse-mode automatic differentiation)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cakelamp.tensor import Tensor
import math


def approx(a, b, tol=1e-4):
    if isinstance(a, (list, tuple)):
        for x, y in zip(a, b):
            assert abs(x - y) < tol, f"{x} != {y} (tol={tol})"
    else:
        assert abs(a - b) < tol, f"{a} != {b} (tol={tol})"


def numerical_grad(f, x, eps=1e-4):
    """Compute numerical gradient for checking."""
    data = x._contiguous_data()
    grad = []
    for i in range(len(data)):
        old = data[i]

        data[i] = old + eps
        x_plus = Tensor._make(data[:], list(x._shape))
        f_plus = f(x_plus).item()

        data[i] = old - eps
        x_minus = Tensor._make(data[:], list(x._shape))
        f_minus = f(x_minus).item()

        grad.append((f_plus - f_minus) / (2 * eps))
        data[i] = old
    return grad


# ---- Basic arithmetic ----

def test_add_backward():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    c = a + b
    loss = c.sum()
    loss.backward()
    approx(a.grad.tolist(), [1.0, 1.0])
    approx(b.grad.tolist(), [1.0, 1.0])


def test_mul_backward():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    c = a * b
    loss = c.sum()
    loss.backward()
    # dc/da = b, dc/db = a
    approx(a.grad.tolist(), [4.0, 5.0])
    approx(b.grad.tolist(), [2.0, 3.0])


def test_sub_backward():
    a = Tensor([5.0, 6.0], requires_grad=True)
    b = Tensor([2.0, 3.0], requires_grad=True)
    c = a - b
    loss = c.sum()
    loss.backward()
    approx(a.grad.tolist(), [1.0, 1.0])
    approx(b.grad.tolist(), [-1.0, -1.0])


def test_div_backward():
    a = Tensor([6.0, 8.0], requires_grad=True)
    b = Tensor([2.0, 4.0], requires_grad=True)
    c = a / b
    loss = c.sum()
    loss.backward()
    # dc/da = 1/b = [0.5, 0.25]
    approx(a.grad.tolist(), [0.5, 0.25])
    # dc/db = -a/b^2 = [-6/4, -8/16] = [-1.5, -0.5]
    approx(b.grad.tolist(), [-1.5, -0.5])


def test_neg_backward():
    a = Tensor([1.0, 2.0], requires_grad=True)
    c = -a
    loss = c.sum()
    loss.backward()
    approx(a.grad.tolist(), [-1.0, -1.0])


def test_pow_backward():
    a = Tensor([2.0, 3.0], requires_grad=True)
    c = a ** 2
    loss = c.sum()
    loss.backward()
    # d(x^2)/dx = 2x
    approx(a.grad.tolist(), [4.0, 6.0])


# ---- Unary ops ----

def test_exp_backward():
    a = Tensor([1.0, 2.0], requires_grad=True)
    c = a.exp()
    loss = c.sum()
    loss.backward()
    # d(exp(x))/dx = exp(x)
    approx(a.grad.tolist(), [math.exp(1.0), math.exp(2.0)])


def test_log_backward():
    a = Tensor([1.0, 2.0], requires_grad=True)
    c = a.log()
    loss = c.sum()
    loss.backward()
    # d(ln(x))/dx = 1/x
    approx(a.grad.tolist(), [1.0, 0.5])


def test_relu_backward():
    a = Tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
    c = a.relu()
    loss = c.sum()
    loss.backward()
    approx(a.grad.tolist(), [0.0, 0.0, 1.0, 1.0])


def test_sigmoid_backward():
    a = Tensor([0.0], requires_grad=True)
    c = a.sigmoid()
    loss = c.sum()
    loss.backward()
    # sigmoid(0) = 0.5, sigmoid'(0) = 0.5 * 0.5 = 0.25
    approx(a.grad.tolist(), [0.25])


# ---- Reductions ----

def test_sum_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    loss = a.sum()
    loss.backward()
    expected = [[1.0, 1.0], [1.0, 1.0]]
    assert a.grad.shape == (2, 2)
    for row_a, row_e in zip(a.grad.tolist(), expected):
        approx(row_a, row_e)


def test_mean_backward():
    a = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    loss = a.mean()
    loss.backward()
    # d(mean)/dx = 1/n
    approx(a.grad.tolist(), [0.25, 0.25, 0.25, 0.25])


# ---- Matrix ops ----

def test_mm_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
    c = a.mm(b)
    loss = c.sum()
    loss.backward()

    # dL/dA = ones @ B^T
    # B^T = [[5,7],[6,8]], ones = [[1,1],[1,1]]
    # dL/dA[i,j] = sum_k(1 * B^T[k,j]) ... actually:
    # dL/dA = grad @ B^T
    expected_a = [[11.0, 15.0], [11.0, 15.0]]  # [[5+6, 7+8], [5+6, 7+8]]
    for row_a, row_e in zip(a.grad.tolist(), expected_a):
        approx(row_a, row_e)


# ---- Chain rule ----

def test_chain_rule():
    """Test that autograd correctly applies the chain rule."""
    x = Tensor([2.0], requires_grad=True)
    # f(x) = (x * x + x) * 3
    # f'(x) = (2x + 1) * 3 = 15 at x=2
    y = x * x + x
    z = y * 3
    loss = z.sum()
    loss.backward()
    approx(x.grad.tolist(), [15.0])


def test_numerical_gradient_check():
    """Compare autograd gradient with numerical gradient."""
    x = Tensor([1.5, -0.5, 2.0], requires_grad=True)

    def f(x):
        return (x * x).sum()

    y = f(x)
    y.backward()

    num_grad = numerical_grad(f, x)
    approx(x.grad.tolist(), num_grad, tol=1e-3)


# ---- Broadcasting backward ----

def test_broadcast_add_backward():
    """Test gradient accumulation through broadcasting."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)  # (2,2)
    b = Tensor([10.0, 20.0], requires_grad=True)  # (2,) broadcasts to (2,2)
    c = a + b
    loss = c.sum()
    loss.backward()

    # grad_a should be all 1s
    approx(a.grad.tolist()[0], [1.0, 1.0])
    approx(a.grad.tolist()[1], [1.0, 1.0])

    # grad_b should sum over the broadcast dim (dim 0)
    approx(b.grad.tolist(), [2.0, 2.0])


# ---- Reshape backward ----

def test_reshape_backward():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    b = a.reshape(4)
    loss = (b * 2).sum()
    loss.backward()
    expected = [[2.0, 2.0], [2.0, 2.0]]
    for row_a, row_e in zip(a.grad.tolist(), expected):
        approx(row_a, row_e)

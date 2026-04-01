"""Tests for the Tensor class."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cakelamp.tensor import Tensor


def approx(a, b, tol=1e-5):
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b), f"Length mismatch: {len(a)} vs {len(b)}"
        for x, y in zip(a, b):
            assert abs(x - y) < tol, f"{x} != {y} (tol={tol})"
    else:
        assert abs(a - b) < tol, f"{a} != {b} (tol={tol})"


# ---- Construction ----

def test_from_list():
    t = Tensor([1.0, 2.0, 3.0])
    assert t.shape == (3,)
    assert t.numel == 3
    approx(t.tolist(), [1.0, 2.0, 3.0])


def test_from_nested_list():
    t = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t.shape == (2, 2)
    assert t.numel == 4
    assert t.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_zeros_ones():
    z = Tensor.zeros([2, 3])
    assert z.shape == (2, 3)
    assert all(v == 0.0 for v in z._contiguous_data())

    o = Tensor.ones([3])
    assert all(v == 1.0 for v in o._contiguous_data())


def test_scalar():
    s = Tensor(42.0)
    assert s.ndim == 0
    assert s.numel == 1
    assert s.item() == 42.0


def test_eye():
    e = Tensor.eye(3)
    assert e.shape == (3, 3)
    data = e.tolist()
    assert data[0] == [1.0, 0.0, 0.0]
    assert data[1] == [0.0, 1.0, 0.0]
    assert data[2] == [0.0, 0.0, 1.0]


def test_arange():
    t = Tensor.arange(0, 5, 1)
    assert t.shape == (5,)
    approx(t.tolist(), [0.0, 1.0, 2.0, 3.0, 4.0])


# ---- Element-wise ops ----

def test_add():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a + b
    approx(c.tolist(), [5.0, 7.0, 9.0])


def test_sub():
    a = Tensor([5.0, 3.0])
    b = Tensor([1.0, 2.0])
    c = a - b
    approx(c.tolist(), [4.0, 1.0])


def test_mul():
    a = Tensor([2.0, 3.0])
    b = Tensor([4.0, 5.0])
    c = a * b
    approx(c.tolist(), [8.0, 15.0])


def test_div():
    a = Tensor([6.0, 8.0])
    b = Tensor([2.0, 4.0])
    c = a / b
    approx(c.tolist(), [3.0, 2.0])


def test_neg():
    a = Tensor([1.0, -2.0])
    b = -a
    approx(b.tolist(), [-1.0, 2.0])


def test_scalar_ops():
    a = Tensor([1.0, 2.0, 3.0])
    b = a + 10
    approx(b.tolist(), [11.0, 12.0, 13.0])
    c = a * 2
    approx(c.tolist(), [2.0, 4.0, 6.0])


def test_exp_log():
    import math
    a = Tensor([0.0, 1.0])
    e = a.exp()
    approx(e.tolist()[0], 1.0)
    approx(e.tolist()[1], math.e, tol=1e-4)
    l = e.log()
    approx(l.tolist(), [0.0, 1.0])


def test_relu():
    a = Tensor([-1.0, 0.0, 1.0, -0.5, 2.0])
    r = a.relu()
    approx(r.tolist(), [0.0, 0.0, 1.0, 0.0, 2.0])


def test_sigmoid():
    a = Tensor([0.0])
    s = a.sigmoid()
    approx(s.tolist()[0], 0.5)


# ---- Broadcasting ----

def test_broadcast_add():
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = Tensor([10.0, 20.0, 30.0])
    c = a + b
    assert c.shape == (2, 3)
    approx(c.tolist()[0], [11.0, 22.0, 33.0])
    approx(c.tolist()[1], [14.0, 25.0, 36.0])


# ---- Reduction ----

def test_sum_all():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    s = a.sum()
    approx(s.item(), 10.0)


def test_sum_dim():
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    s0 = a.sum(dim=0)
    approx(s0.tolist(), [5.0, 7.0, 9.0])
    s1 = a.sum(dim=1)
    approx(s1.tolist(), [6.0, 15.0])


def test_mean():
    a = Tensor([1.0, 2.0, 3.0, 4.0])
    m = a.mean()
    approx(m.item(), 2.5)


def test_argmax():
    a = Tensor([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]])
    am = a.argmax(dim=1)
    approx(am.tolist(), [1.0, 2.0])


# ---- Matrix ops ----

def test_mm():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a.mm(b)
    assert c.shape == (2, 2)
    approx(c.tolist()[0], [19.0, 22.0])
    approx(c.tolist()[1], [43.0, 50.0])


def test_matmul_operator():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[5.0, 6.0], [7.0, 8.0]])
    c = a @ b
    approx(c.tolist()[0], [19.0, 22.0])


# ---- Shape ops ----

def test_reshape():
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = a.reshape(3, 2)
    assert b.shape == (3, 2)
    approx(b.tolist()[0], [1.0, 2.0])


def test_transpose():
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = a.T
    assert b.shape == (3, 2)
    assert b.tolist()[0] == [1.0, 4.0]


def test_unsqueeze_squeeze():
    a = Tensor([1.0, 2.0, 3.0])
    b = a.unsqueeze(0)
    assert b.shape == (1, 3)
    c = b.squeeze()
    assert c.shape == (3,)


# ---- In-place ops ----

def test_add_inplace():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([10.0, 20.0, 30.0])
    a.add_(b)
    approx(a.tolist(), [11.0, 22.0, 33.0])


def test_zero_():
    a = Tensor([1.0, 2.0, 3.0])
    a.zero_()
    approx(a.tolist(), [0.0, 0.0, 0.0])


# ---- Indexing ----

def test_getitem():
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    row0 = a[0]
    assert row0.shape == (2,)
    approx(row0.tolist(), [1.0, 2.0])
    row1 = a[1]
    approx(row1.tolist(), [3.0, 4.0])


# ---- Softmax ----

def test_softmax():
    a = Tensor([[1.0, 2.0, 3.0]])
    s = a.softmax(dim=1)
    data = s.tolist()[0]
    total = sum(data)
    approx(total, 1.0)
    assert data[0] < data[1] < data[2]


def test_log_softmax():
    import math
    a = Tensor([[1.0, 2.0, 3.0]])
    ls = a.log_softmax(dim=1)
    data = ls.tolist()[0]
    # exp of log_softmax should sum to 1
    total = sum(math.exp(x) for x in data)
    approx(total, 1.0)

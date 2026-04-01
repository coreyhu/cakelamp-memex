"""Tests for the CakeLamp Python frontend."""
import math
import cakelamp


def test_create_tensor():
    t = cakelamp.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    assert t.shape == [2, 2]
    assert t.ndim == 2
    assert t.numel() == 4
    assert t.is_contiguous()


def test_zeros_ones():
    z = cakelamp.zeros([3, 4])
    assert z.shape == [3, 4]
    assert z.tolist() == [0.0] * 12

    o = cakelamp.ones([2, 3])
    assert o.tolist() == [1.0] * 6


def test_full():
    t = cakelamp.full([2, 2], 3.14)
    data = t.tolist()
    assert all(abs(x - 3.14) < 1e-5 for x in data)


def test_rand():
    t = cakelamp.rand([10])
    data = t.tolist()
    assert len(data) == 10
    assert all(0.0 <= x < 1.0 for x in data)


def test_randn():
    t = cakelamp.randn([100])
    data = t.tolist()
    assert len(data) == 100
    # Mean should be approximately 0
    mean = sum(data) / len(data)
    assert abs(mean) < 1.0  # loose check


def test_scalar():
    s = cakelamp.Tensor.scalar(42.0)
    assert s.item() == 42.0
    assert s.ndim == 0


def test_arange():
    t = cakelamp.arange(0.0, 5.0)
    assert t.shape == [5]
    assert t.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_get_set():
    t = cakelamp.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    assert t.get([0, 0]) == 1.0
    assert t.get([1, 1]) == 4.0
    t.set([1, 1], 42.0)
    assert t.get([1, 1]) == 42.0


def test_add():
    a = cakelamp.tensor([1.0, 2.0, 3.0], [3])
    b = cakelamp.tensor([4.0, 5.0, 6.0], [3])
    c = a + b
    assert c.tolist() == [5.0, 7.0, 9.0]


def test_sub():
    a = cakelamp.tensor([5.0, 3.0, 1.0], [3])
    b = cakelamp.tensor([1.0, 2.0, 3.0], [3])
    c = a - b
    assert c.tolist() == [4.0, 1.0, -2.0]


def test_mul():
    a = cakelamp.tensor([2.0, 3.0], [2])
    b = cakelamp.tensor([4.0, 5.0], [2])
    c = a * b
    assert c.tolist() == [8.0, 15.0]


def test_div():
    a = cakelamp.tensor([6.0, 8.0], [2])
    b = cakelamp.tensor([2.0, 4.0], [2])
    c = a / b
    assert c.tolist() == [3.0, 2.0]


def test_neg():
    a = cakelamp.tensor([1.0, -2.0, 3.0], [3])
    b = -a
    assert b.tolist() == [-1.0, 2.0, -3.0]


def test_broadcast_add():
    a = cakelamp.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    b = cakelamp.tensor([10.0, 20.0, 30.0], [3])
    c = a + b
    assert c.shape == [2, 3]
    assert c.tolist() == [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]


def test_exp_log():
    a = cakelamp.tensor([0.0, 1.0], [2])
    e = a.exp()
    assert abs(e.get([0]) - 1.0) < 1e-5
    assert abs(e.get([1]) - math.e) < 1e-4

    l = e.log()
    assert abs(l.get([0]) - 0.0) < 1e-5
    assert abs(l.get([1]) - 1.0) < 1e-4


def test_relu():
    a = cakelamp.tensor([-1.0, 0.0, 1.0, -0.5, 2.0], [5])
    r = cakelamp.relu(a)
    assert r.tolist() == [0.0, 0.0, 1.0, 0.0, 2.0]


def test_sigmoid():
    a = cakelamp.tensor([0.0], [1])
    s = cakelamp.sigmoid(a)
    assert abs(s.get([0]) - 0.5) < 1e-5


def test_tanh():
    a = cakelamp.tensor([0.0], [1])
    t = cakelamp.tanh(a)
    assert abs(t.get([0])) < 1e-5


def test_sum():
    a = cakelamp.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    s = a.sum()
    assert s.item() == 10.0


def test_sum_dim():
    a = cakelamp.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    s0 = a.sum_dim(0)
    assert s0.tolist() == [5.0, 7.0, 9.0]

    s1 = a.sum_dim(1)
    assert s1.tolist() == [6.0, 15.0]


def test_mean():
    a = cakelamp.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    m = a.mean()
    assert m.item() == 2.5


def test_max_min():
    a = cakelamp.tensor([3.0, 1.0, 4.0, 1.0, 5.0, 9.0], [6])
    assert a.max().item() == 9.0
    assert a.min().item() == 1.0


def test_matmul():
    a = cakelamp.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = cakelamp.tensor([5.0, 6.0, 7.0, 8.0], [2, 2])
    c = a @ b
    assert c.tolist() == [19.0, 22.0, 43.0, 50.0]


def test_matmul_function():
    a = cakelamp.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = cakelamp.tensor([5.0, 6.0, 7.0, 8.0], [2, 2])
    c = cakelamp.matmul(a, b)
    assert c.tolist() == [19.0, 22.0, 43.0, 50.0]


def test_reshape():
    a = cakelamp.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    b = a.reshape([3, 2])
    assert b.shape == [3, 2]
    assert b.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_transpose():
    a = cakelamp.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    b = a.t()
    assert b.shape == [3, 2]
    assert b.get([0, 0]) == 1.0
    assert b.get([0, 1]) == 4.0


def test_unsqueeze_squeeze():
    a = cakelamp.tensor([1.0, 2.0, 3.0], [3])
    b = a.unsqueeze(0)
    assert b.shape == [1, 3]
    c = b.squeeze()
    assert c.shape == [3]


def test_softmax():
    a = cakelamp.tensor([1.0, 2.0, 3.0], [1, 3])
    s = a.softmax(1)
    data = s.tolist()
    assert abs(sum(data) - 1.0) < 1e-5


def test_log_softmax():
    a = cakelamp.tensor([1.0, 2.0, 3.0], [1, 3])
    ls = a.log_softmax(1)
    data = ls.tolist()
    # log_softmax values should be negative
    assert all(x < 0 for x in data)


def test_argmax():
    a = cakelamp.tensor([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], [2, 3])
    am = a.argmax(1)
    assert am.tolist() == [1.0, 2.0]


def test_one_hot():
    indices = cakelamp.tensor([0.0, 2.0, 1.0], [3])
    oh = cakelamp.one_hot(indices, 3)
    assert oh.shape == [3, 3]
    expected = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert oh.tolist() == expected


def test_inplace_ops():
    a = cakelamp.tensor([1.0, 2.0, 3.0], [3])
    b = cakelamp.tensor([10.0, 20.0, 30.0], [3])
    a.add_(b)
    assert a.tolist() == [11.0, 22.0, 33.0]


def test_mul_scalar():
    a = cakelamp.tensor([1.0, 2.0, 3.0], [3])
    b = a.mul_scalar(2.0)
    assert b.tolist() == [2.0, 4.0, 6.0]


def test_repr():
    a = cakelamp.tensor([1.0, 2.0], [2])
    r = repr(a)
    assert "Tensor" in r


def test_eq():
    a = cakelamp.tensor([1.0, 2.0, 3.0], [3])
    b = cakelamp.tensor([1.0, 5.0, 3.0], [3])
    c = a.eq(b)
    assert c.tolist() == [1.0, 0.0, 1.0]

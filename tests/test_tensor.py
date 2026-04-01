"""Tests for cakelamp Python bindings."""
import cakelamp
from cakelamp import Tensor
import math


class TestTensorCreation:
    def test_from_data(self):
        t = Tensor([1.0, 2.0, 3.0], [3])
        assert t.shape == [3]
        assert t.ndim == 1
        assert t.numel() == 3
        assert t.tolist() == [1.0, 2.0, 3.0]

    def test_from_data_2d(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        assert t.shape == [2, 3]
        assert t.ndim == 2
        assert t.numel() == 6

    def test_zeros(self):
        t = cakelamp.zeros([3, 4])
        assert t.shape == [3, 4]
        assert all(v == 0.0 for v in t.tolist())

    def test_ones(self):
        t = cakelamp.ones([2, 3])
        assert t.shape == [2, 3]
        assert all(v == 1.0 for v in t.tolist())

    def test_full(self):
        t = cakelamp.full([2, 2], 5.0)
        assert all(v == 5.0 for v in t.tolist())

    def test_rand(self):
        t = cakelamp.rand([10])
        assert t.shape == [10]
        data = t.tolist()
        assert all(0.0 <= v < 1.0 for v in data)

    def test_randn(self):
        t = cakelamp.randn([100])
        assert t.shape == [100]
        # Just check it doesn't crash and has reasonable values
        data = t.tolist()
        assert len(data) == 100

    def test_scalar(self):
        t = cakelamp.scalar(3.14)
        assert t.ndim == 0
        assert abs(t.item() - 3.14) < 1e-5

    def test_arange(self):
        t = cakelamp.arange(0.0, 5.0)
        assert t.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]

    def test_tensor_func(self):
        t = cakelamp.tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
        assert t.shape == [2, 2]
        assert t.tolist() == [1.0, 2.0, 3.0, 4.0]


class TestTensorOps:
    def test_add(self):
        a = Tensor([1.0, 2.0, 3.0], [3])
        b = Tensor([4.0, 5.0, 6.0], [3])
        c = a + b
        assert c.tolist() == [5.0, 7.0, 9.0]

    def test_sub(self):
        a = Tensor([4.0, 5.0, 6.0], [3])
        b = Tensor([1.0, 2.0, 3.0], [3])
        c = a - b
        assert c.tolist() == [3.0, 3.0, 3.0]

    def test_mul(self):
        a = Tensor([2.0, 3.0, 4.0], [3])
        b = Tensor([5.0, 6.0, 7.0], [3])
        c = a * b
        assert c.tolist() == [10.0, 18.0, 28.0]

    def test_truediv(self):
        a = Tensor([10.0, 20.0, 30.0], [3])
        b = Tensor([2.0, 4.0, 5.0], [3])
        c = a / b
        assert c.tolist() == [5.0, 5.0, 6.0]

    def test_neg(self):
        a = Tensor([1.0, -2.0, 3.0], [3])
        b = -a
        assert b.tolist() == [-1.0, 2.0, -3.0]

    def test_matmul_operator(self):
        a = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
        b = Tensor([5.0, 6.0, 7.0, 8.0], [2, 2])
        c = a @ b
        # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
        assert c.tolist() == [19.0, 22.0, 43.0, 50.0]

    def test_matmul_func(self):
        a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        b = Tensor([7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [3, 2])
        c = cakelamp.matmul(a, b)
        assert c.shape == [2, 2]
        assert c.tolist() == [58.0, 64.0, 139.0, 154.0]


class TestUnaryOps:
    def test_exp(self):
        t = Tensor([0.0, 1.0], [2])
        r = t.exp()
        data = r.tolist()
        assert abs(data[0] - 1.0) < 1e-5
        assert abs(data[1] - math.e) < 1e-4

    def test_log(self):
        t = Tensor([1.0, math.e], [2])
        r = t.log()
        data = r.tolist()
        assert abs(data[0] - 0.0) < 1e-5
        assert abs(data[1] - 1.0) < 1e-4

    def test_relu(self):
        t = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0], [5])
        r = t.relu()
        assert r.tolist() == [0.0, 0.0, 0.0, 1.0, 2.0]

    def test_relu_func(self):
        t = Tensor([-1.0, 0.0, 1.0], [3])
        r = cakelamp.relu(t)
        assert r.tolist() == [0.0, 0.0, 1.0]

    def test_sigmoid(self):
        t = Tensor([0.0], [1])
        r = t.sigmoid()
        assert abs(r.item() - 0.5) < 1e-5

    def test_tanh(self):
        t = Tensor([0.0], [1])
        r = t.tanh()
        assert abs(r.item() - 0.0) < 1e-5

    def test_abs(self):
        t = Tensor([-1.0, 2.0, -3.0], [3])
        r = t.abs()
        assert r.tolist() == [1.0, 2.0, 3.0]

    def test_sqrt(self):
        t = Tensor([4.0, 9.0, 16.0], [3])
        r = t.sqrt()
        assert r.tolist() == [2.0, 3.0, 4.0]

    def test_clamp(self):
        t = Tensor([-1.0, 0.5, 2.0], [3])
        r = t.clamp(0.0, 1.0)
        assert r.tolist() == [0.0, 0.5, 1.0]


class TestReductions:
    def test_sum_all(self):
        t = Tensor([1.0, 2.0, 3.0], [3])
        s = t.sum()
        assert abs(s.item() - 6.0) < 1e-5

    def test_sum_dim(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        s = t.sum(dim=1)
        assert s.tolist() == [6.0, 15.0]

    def test_sum_dim_keepdim(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        s = t.sum(dim=1, keepdim=True)
        assert s.shape == [2, 1]

    def test_mean_all(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0], [4])
        m = t.mean()
        assert abs(m.item() - 2.5) < 1e-5

    def test_mean_dim(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        m = t.mean(dim=0)
        data = m.tolist()
        assert abs(data[0] - 2.5) < 1e-5
        assert abs(data[1] - 3.5) < 1e-5
        assert abs(data[2] - 4.5) < 1e-5

    def test_max_all(self):
        t = Tensor([1.0, 5.0, 3.0, 2.0], [4])
        m = t.max()
        assert abs(m.item() - 5.0) < 1e-5

    def test_argmax(self):
        t = Tensor([1.0, 3.0, 2.0, 5.0, 4.0, 6.0], [2, 3])
        am = t.argmax(dim=1)
        assert am.tolist() == [1.0, 2.0]


class TestViewOps:
    def test_reshape(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        r = t.reshape([3, 2])
        assert r.shape == [3, 2]
        assert r.tolist() == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    def test_transpose(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        r = t.t()
        assert r.shape == [3, 2]
        assert r.tolist() == [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]

    def test_unsqueeze(self):
        t = Tensor([1.0, 2.0, 3.0], [3])
        r = t.unsqueeze(0)
        assert r.shape == [1, 3]

    def test_squeeze(self):
        t = Tensor([1.0, 2.0, 3.0], [1, 3])
        r = t.squeeze()
        assert r.shape == [3]

    def test_contiguous(self):
        t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        r = t.t().contiguous()
        assert r.is_contiguous()
        assert r.shape == [3, 2]

    def test_expand(self):
        t = Tensor([1.0, 2.0, 3.0], [1, 3])
        r = t.expand([4, 3])
        assert r.shape == [4, 3]
        data = r.tolist()
        assert data == [1.0, 2.0, 3.0] * 4


class TestBroadcasting:
    def test_add_broadcast(self):
        a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
        b = Tensor([10.0, 20.0, 30.0], [3])
        c = a + b
        assert c.shape == [2, 3]
        assert c.tolist() == [11.0, 22.0, 33.0, 14.0, 25.0, 36.0]

    def test_mul_broadcast(self):
        a = Tensor([1.0, 2.0, 3.0, 4.0], [2, 2])
        b = Tensor([10.0, 20.0], [1, 2])
        c = a * b
        assert c.shape == [2, 2]
        assert c.tolist() == [10.0, 40.0, 30.0, 80.0]


class TestSoftmax:
    def test_softmax(self):
        t = Tensor([1.0, 2.0, 3.0], [1, 3])
        s = t.softmax(dim=1)
        data = s.tolist()
        assert abs(sum(data) - 1.0) < 1e-5
        # Check monotonicity
        assert data[0] < data[1] < data[2]

    def test_log_softmax(self):
        t = Tensor([1.0, 2.0, 3.0], [1, 3])
        ls = t.log_softmax(dim=1)
        data = ls.tolist()
        # All log-softmax values should be negative
        assert all(v < 0 for v in data)

    def test_softmax_func(self):
        t = Tensor([1.0, 1.0, 1.0], [1, 3])
        s = cakelamp.softmax(t, dim=1)
        data = s.tolist()
        # Equal inputs => equal probabilities
        for v in data:
            assert abs(v - 1.0/3.0) < 1e-5


class TestInPlaceOps:
    def test_add_inplace(self):
        a = Tensor([1.0, 2.0, 3.0], [3])
        b = Tensor([4.0, 5.0, 6.0], [3])
        a.add_(b)
        assert a.tolist() == [5.0, 7.0, 9.0]

    def test_mul_scalar_inplace(self):
        a = Tensor([1.0, 2.0, 3.0], [3])
        a.mul_scalar_(2.0)
        assert a.tolist() == [2.0, 4.0, 6.0]

    def test_fill_inplace(self):
        a = Tensor([1.0, 2.0, 3.0], [3])
        a.fill_(0.0)
        assert a.tolist() == [0.0, 0.0, 0.0]


class TestRepr:
    def test_repr(self):
        t = Tensor([1.0, 2.0, 3.0], [3])
        r = repr(t)
        assert "Tensor" in r
        assert "1.0" in r


class TestOneHot:
    def test_one_hot(self):
        idx = Tensor([0.0, 2.0, 1.0], [3])
        oh = cakelamp.one_hot(idx, 3)
        assert oh.shape == [3, 3]
        expected = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
        assert oh.tolist() == expected

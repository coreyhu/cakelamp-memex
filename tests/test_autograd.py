"""Tests for the CakeLamp autograd system."""
import math
import cakelamp
import cakelamp._core as _C
from cakelamp.autograd.engine import AutogradTensor


def test_simple_add_backward():
    a = AutogradTensor(_C.tensor([2.0, 3.0], [2]), requires_grad=True)
    b = AutogradTensor(_C.tensor([4.0, 5.0], [2]), requires_grad=True)
    c = a + b
    loss = c.sum()
    loss.backward()
    assert a.grad.data.tolist() == [1.0, 1.0]
    assert b.grad.data.tolist() == [1.0, 1.0]


def test_mul_backward():
    a = AutogradTensor(_C.tensor([2.0, 3.0], [2]), requires_grad=True)
    b = AutogradTensor(_C.tensor([4.0, 5.0], [2]), requires_grad=True)
    c = a * b
    loss = c.sum()
    loss.backward()
    # dc/da = b, dc/db = a
    assert a.grad.data.tolist() == [4.0, 5.0]
    assert b.grad.data.tolist() == [2.0, 3.0]


def test_matmul_backward():
    # a: (1,2), b: (2,1)
    a = AutogradTensor(_C.tensor([1.0, 2.0], [1, 2]), requires_grad=True)
    b = AutogradTensor(_C.tensor([3.0, 4.0], [2, 1]), requires_grad=True)
    c = a @ b  # (1,1) = [[11.0]]
    loss = c.sum()
    loss.backward()
    # grad_a = grad @ b.T = [[1]] @ [[3,4]] = [[3,4]]
    assert a.grad.data.tolist() == [3.0, 4.0]
    # grad_b = a.T @ grad = [[1],[2]] @ [[1]] = [[1],[2]]
    assert b.grad.data.tolist() == [1.0, 2.0]


def test_relu_backward():
    a = AutogradTensor(_C.tensor([-1.0, 0.0, 1.0, 2.0], [4]), requires_grad=True)
    b = a.relu()
    loss = b.sum()
    loss.backward()
    assert a.grad.data.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_exp_backward():
    a = AutogradTensor(_C.tensor([0.0, 1.0], [2]), requires_grad=True)
    b = a.exp()
    loss = b.sum()
    loss.backward()
    # d(exp(x))/dx = exp(x)
    grads = a.grad.data.tolist()
    assert abs(grads[0] - 1.0) < 1e-5
    assert abs(grads[1] - math.e) < 1e-4


def test_log_backward():
    a = AutogradTensor(_C.tensor([1.0, 2.0], [2]), requires_grad=True)
    b = a.log()
    loss = b.sum()
    loss.backward()
    # d(ln(x))/dx = 1/x
    grads = a.grad.data.tolist()
    assert abs(grads[0] - 1.0) < 1e-5
    assert abs(grads[1] - 0.5) < 1e-5


def test_neg_backward():
    a = AutogradTensor(_C.tensor([1.0, 2.0], [2]), requires_grad=True)
    b = -a
    loss = b.sum()
    loss.backward()
    assert a.grad.data.tolist() == [-1.0, -1.0]


def test_chain_rule():
    # f(x) = (x * 2 + 1).sum()
    # df/dx = 2
    x = AutogradTensor(_C.tensor([3.0, 4.0], [2]), requires_grad=True)
    two = AutogradTensor(_C.Tensor.scalar(2.0))
    y = x * two + AutogradTensor(_C.tensor([1.0, 1.0], [2]))
    loss = y.sum()
    loss.backward()
    assert x.grad.data.tolist() == [2.0, 2.0]


def test_mean_backward():
    x = AutogradTensor(_C.tensor([1.0, 2.0, 3.0, 4.0], [4]), requires_grad=True)
    loss = x.mean()
    loss.backward()
    expected = [0.25, 0.25, 0.25, 0.25]
    grads = x.grad.data.tolist()
    for g, e in zip(grads, expected):
        assert abs(g - e) < 1e-6


def test_sigmoid_backward():
    x = AutogradTensor(_C.tensor([0.0], [1]), requires_grad=True)
    y = x.sigmoid()
    loss = y.sum()
    loss.backward()
    # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.5 * 0.5 = 0.25
    assert abs(x.grad.data.tolist()[0] - 0.25) < 1e-5


def test_reshape_backward():
    x = AutogradTensor(_C.tensor([1.0, 2.0, 3.0, 4.0], [2, 2]), requires_grad=True)
    y = x.reshape([4])
    loss = y.sum()
    loss.backward()
    assert x.grad.data.shape == [2, 2]
    assert x.grad.data.tolist() == [1.0, 1.0, 1.0, 1.0]


def test_linear_forward_backward():
    """Test that a Linear layer can forward and backward."""
    from cakelamp.nn import Linear

    layer = Linear(3, 2)
    x = AutogradTensor(_C.tensor([1.0, 2.0, 3.0], [1, 3]), requires_grad=True)
    out = layer(x)
    assert out.shape == [1, 2]

    loss = out.sum()
    loss.backward()

    # Check that gradients exist
    assert layer.weight.grad is not None
    assert layer.bias.grad is not None
    assert x.grad is not None


def test_cross_entropy_loss():
    """Test CrossEntropyLoss forward and backward."""
    from cakelamp.nn import CrossEntropyLoss

    logits = AutogradTensor(
        _C.tensor([2.0, 1.0, 0.1, 0.1, 1.0, 2.0], [2, 3]),
        requires_grad=True,
    )
    targets = AutogradTensor(_C.tensor([0.0, 2.0], [2]))

    criterion = CrossEntropyLoss()
    loss = criterion(logits, targets)

    assert loss.data.numel() == 1
    loss_val = loss.item()
    assert loss_val > 0  # Loss should be positive

    loss.backward()
    assert logits.grad is not None


def test_sgd_step():
    """Test SGD optimizer makes a step."""
    from cakelamp.nn import Linear
    from cakelamp.optim import SGD

    layer = Linear(2, 1)
    optimizer = SGD(layer.parameters(), lr=0.1)

    x = AutogradTensor(_C.tensor([1.0, 2.0], [1, 2]), requires_grad=False)
    target = AutogradTensor(_C.Tensor.scalar(5.0))

    # Get initial weight values
    initial_w = layer.weight.data.tolist()[:]

    # Forward, backward, step
    out = layer(x)
    loss = (out.sum() - target) ** AutogradTensor(_C.Tensor.scalar(2.0))
    loss.backward()
    optimizer.step()

    # Weights should have changed
    new_w = layer.weight.data.tolist()
    assert new_w != initial_w


def test_adam_step():
    """Test Adam optimizer makes a step."""
    from cakelamp.nn import Linear
    from cakelamp.optim import Adam

    layer = Linear(2, 1)
    optimizer = Adam(layer.parameters(), lr=0.01)

    x = AutogradTensor(_C.tensor([1.0, 2.0], [1, 2]), requires_grad=False)
    target = AutogradTensor(_C.Tensor.scalar(5.0))

    initial_w = layer.weight.data.tolist()[:]

    out = layer(x)
    loss = (out.sum() - target) ** AutogradTensor(_C.Tensor.scalar(2.0))
    loss.backward()
    optimizer.step()

    new_w = layer.weight.data.tolist()
    assert new_w != initial_w


def test_mlp_forward_backward():
    """Test a simple MLP forward and backward pass."""
    from cakelamp.nn import Linear, ReLU, CrossEntropyLoss

    # Simple 2-layer MLP
    linear1 = Linear(4, 3)
    relu = ReLU()
    linear2 = Linear(3, 2)
    criterion = CrossEntropyLoss()

    x = AutogradTensor(_C.tensor([1.0, 2.0, 3.0, 4.0] * 2, [2, 4]))
    targets = AutogradTensor(_C.tensor([0.0, 1.0], [2]))

    # Forward
    h = relu(linear1(x))
    out = linear2(h)
    loss = criterion(out, targets)

    assert loss.data.numel() == 1

    # Backward
    loss.backward()

    # All parameters should have gradients
    assert linear1.weight.grad is not None
    assert linear1.bias.grad is not None
    assert linear2.weight.grad is not None
    assert linear2.bias.grad is not None


def test_training_loop():
    """Test that loss decreases over a few training iterations."""
    from cakelamp.nn import Linear, ReLU, MSELoss
    from cakelamp.optim import SGD

    linear = Linear(2, 1)
    criterion = MSELoss()
    optimizer = SGD(linear.parameters(), lr=0.01)

    x = AutogradTensor(_C.tensor([1.0, 2.0, 3.0, 4.0], [2, 2]))
    target = AutogradTensor(_C.tensor([3.0, 7.0], [2, 1]))

    losses = []
    for _ in range(10):
        optimizer.zero_grad()
        out = linear(x)
        loss = criterion(out, target)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    # Loss should generally decrease (not strictly monotonic due to noise)
    assert losses[-1] < losses[0], f"Loss didn't decrease: {losses[0]} -> {losses[-1]}"

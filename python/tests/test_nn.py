"""Tests for neural network modules."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from cakelamp.tensor import Tensor
from cakelamp.nn import Linear, ReLU, Sequential, MSELoss, CrossEntropyLoss, Module, Parameter


def approx(a, b, tol=1e-4):
    assert abs(a - b) < tol, f"{a} != {b} (tol={tol})"


def test_linear_shape():
    linear = Linear(3, 5)
    x = Tensor.rand([4, 3], requires_grad=True)
    y = linear(x)
    assert y.shape == (4, 5)


def test_linear_backward():
    linear = Linear(2, 3)
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    y = linear(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == (1, 2)
    assert linear.weight.grad is not None
    assert linear.weight.grad.shape == (3, 2)


def test_relu_module():
    relu = ReLU()
    x = Tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = relu(x)
    expected = [0.0, 0.0, 1.0, 2.0]
    for a, b in zip(y.tolist(), expected):
        approx(a, b)


def test_sequential():
    model = Sequential(
        Linear(2, 4),
        ReLU(),
        Linear(4, 1),
    )
    x = Tensor.rand([3, 2], requires_grad=True)
    y = model(x)
    assert y.shape == (3, 1)

    params = model.parameters()
    assert len(params) == 4  # 2 weights + 2 biases


def test_mse_loss():
    pred = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    target = Tensor([1.5, 2.5, 3.5])
    loss = MSELoss()(pred, target)
    approx(loss.item(), 0.25)  # mean of [0.25, 0.25, 0.25]


def test_cross_entropy_loss():
    # Simple 2-class case
    logits = Tensor([[2.0, 1.0], [0.0, 3.0]], requires_grad=True)
    targets = Tensor([0.0, 1.0])  # class 0 and class 1
    loss = CrossEntropyLoss()(logits, targets)
    assert loss.item() > 0  # loss should be positive
    loss.backward()
    assert logits.grad is not None


def test_parameters():
    model = Sequential(
        Linear(10, 5),
        ReLU(),
        Linear(5, 2),
    )
    params = model.parameters()
    # 4 parameters: 2 weights + 2 biases
    assert len(params) == 4
    shapes = [p.shape for p in params]
    assert (10, 5) in shapes or (5, 10) in shapes


def test_zero_grad():
    model = Sequential(
        Linear(2, 2),
    )
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    y = model(x)
    loss = y.sum()
    loss.backward()

    # Grads should exist
    for p in model.parameters():
        assert p.grad is not None

    model.zero_grad()
    for p in model.parameters():
        assert p.grad is None


def test_training_loop():
    """Test a simple training loop converges."""
    from cakelamp.optim import SGD

    # Learn y = 2x + 1
    model = Linear(1, 1, bias=True)
    optimizer = SGD(model.parameters(), lr=0.01)
    loss_fn = MSELoss()

    # Training data
    x_data = [[1.0], [2.0], [3.0], [4.0]]
    y_data = [[3.0], [5.0], [7.0], [9.0]]

    x = Tensor(x_data, requires_grad=True)
    y = Tensor(y_data)

    initial_loss = None
    final_loss = None

    for epoch in range(100):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)

        if epoch == 0:
            initial_loss = loss.item()

        loss.backward()
        optimizer.step()

        if epoch == 99:
            final_loss = loss.item()

    # Loss should decrease
    assert final_loss < initial_loss, f"Loss did not decrease: {initial_loss} -> {final_loss}"

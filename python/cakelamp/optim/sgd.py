"""SGD optimizer."""

from __future__ import annotations
import cakelamp._core as _C
from cakelamp.autograd.engine import AutogradTensor


class SGD:
    """Stochastic Gradient Descent optimizer.

    Parameters
    ----------
    params : iterable of Parameter
        Parameters to optimize.
    lr : float
        Learning rate.
    momentum : float
        Momentum factor (default: 0).
    weight_decay : float
        L2 penalty (default: 0).
    """

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [None] * len(self.params)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad_data = p.grad.data

            # Weight decay
            if self.weight_decay != 0:
                grad_data = grad_data + p.data.mul_scalar(self.weight_decay)

            # Momentum
            if self.momentum != 0:
                if self.velocity[i] is None:
                    self.velocity[i] = _C.Tensor(grad_data.tolist(), grad_data.shape)
                else:
                    # v = momentum * v + grad
                    v_data = self.velocity[i].tolist()
                    g_data = grad_data.tolist()
                    new_v = [self.momentum * v + g for v, g in zip(v_data, g_data)]
                    self.velocity[i] = _C.Tensor(new_v, grad_data.shape)
                grad_data = self.velocity[i]

            # Update: p.data -= lr * grad
            g_list = grad_data.tolist()
            p_list = p.data.tolist()
            new_data = [pp - self.lr * gg for pp, gg in zip(p_list, g_list)]
            p.data = _C.Tensor(new_data, p.data.shape)

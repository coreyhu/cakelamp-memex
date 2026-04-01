"""Optimizers mirroring torch.optim."""

from cakelamp.tensor import Tensor
from cakelamp import backend as B


class Optimizer:
    """Base optimizer class."""

    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent with optional momentum."""

    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = [None] * len(self.params)

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad

            # Weight decay
            if self.weight_decay != 0:
                grad = grad + p.detach() * self.weight_decay

            # Momentum
            if self.momentum != 0:
                if self.velocity[i] is None:
                    self.velocity[i] = grad.clone()
                else:
                    self.velocity[i] = self.velocity[i] * self.momentum + grad
                grad = self.velocity[i]

            # Update: p = p - lr * grad
            update = grad * self.lr
            new_data = p.detach() - update
            p._data = new_data._contiguous_data()
            p._strides = B.compute_strides(p._shape)
            p._offset = 0
            p._version += 1


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = [None] * len(self.params)  # first moment
        self.v = [None] * len(self.params)  # second moment
        self.t = 0  # timestep

    def step(self):
        self.t += 1
        beta1, beta2 = self.betas

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            grad = p.grad

            # Weight decay
            if self.weight_decay != 0:
                grad = grad + p.detach() * self.weight_decay

            # Initialize moments
            if self.m[i] is None:
                self.m[i] = Tensor.zeros(list(p._shape))
                self.v[i] = Tensor.zeros(list(p._shape))

            # Update moments
            self.m[i] = self.m[i] * beta1 + grad * (1 - beta1)
            self.v[i] = self.v[i] * beta2 + (grad * grad) * (1 - beta2)

            # Bias correction
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)

            # Update: p = p - lr * m_hat / (sqrt(v_hat) + eps)
            update = m_hat / ((v_hat ** 0.5) + self.eps) * self.lr
            new_data = p.detach() - update
            p._data = new_data._contiguous_data()
            p._strides = B.compute_strides(p._shape)
            p._offset = 0
            p._version += 1

"""Activation function modules."""

from cakelamp.autograd.engine import AutogradTensor
from cakelamp.nn.module import Module


class ReLU(Module):
    def forward(self, x: AutogradTensor) -> AutogradTensor:
        return x.relu()


class Sigmoid(Module):
    def forward(self, x: AutogradTensor) -> AutogradTensor:
        return x.sigmoid()


class Tanh(Module):
    def forward(self, x: AutogradTensor) -> AutogradTensor:
        return x.tanh()


class Softmax(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x: AutogradTensor) -> AutogradTensor:
        return x.softmax(self.dim)


class LogSoftmax(Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x: AutogradTensor) -> AutogradTensor:
        return x.log_softmax(self.dim)

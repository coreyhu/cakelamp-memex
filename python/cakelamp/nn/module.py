"""Base Module and Parameter classes for neural networks."""

from __future__ import annotations
from typing import Iterator, Optional
import math
import cakelamp._core as _C
from cakelamp.autograd.engine import AutogradTensor


class Parameter(AutogradTensor):
    """A tensor that is a module parameter (always requires_grad=True)."""

    def __init__(self, data, requires_grad=True):
        super().__init__(data, requires_grad=requires_grad)


class Module:
    """Base class for all neural network modules.

    Subclasses should implement forward().
    """

    def __init__(self):
        self._parameters: dict[str, Parameter] = {}
        self._modules: dict[str, Module] = {}
        self.training = True

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.__dict__.setdefault('_parameters', {})[name] = value
        elif isinstance(value, Module) and name != '_parameters' and name != '_modules':
            self.__dict__.setdefault('_modules', {})[name] = value
        object.__setattr__(self, name, value)

    def forward(self, *args, **kwargs) -> AutogradTensor:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> AutogradTensor:
        return self.forward(*args, **kwargs)

    def parameters(self) -> Iterator[Parameter]:
        """Return an iterator over module parameters."""
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def zero_grad(self):
        """Zero all parameter gradients."""
        for p in self.parameters():
            p.grad = None

    def train(self, mode=True):
        self.training = mode
        for m in self._modules.values():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

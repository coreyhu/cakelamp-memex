"""Mock Tensor implementation for testing optimizers.

Provides a minimal tensor-like object that supports the operations
used by the optimizers (add, mul, div, sqrt, clone, zeros_like, etc.)
using plain Python lists for storage.

This allows testing optimizer logic independently of the Rust backend.
"""

from __future__ import annotations

import math
from typing import List, Optional, Union


class MockTensor:
    """A simple tensor backed by a flat Python list of floats.

    Supports element-wise arithmetic used by the optimizers.
    """

    def __init__(self, data: List[float]) -> None:
        self._data = list(data)

    # ---- factory helpers ------------------------------------------------

    def clone(self) -> MockTensor:
        return MockTensor(list(self._data))

    def zeros_like(self) -> MockTensor:
        return MockTensor([0.0] * len(self._data))

    # ---- element-wise ops (return new tensor) ---------------------------

    def add(self, other: Union[MockTensor, float]) -> MockTensor:
        if isinstance(other, MockTensor):
            return MockTensor([a + b for a, b in zip(self._data, other._data)])
        return MockTensor([a + other for a in self._data])

    def mul(self, other: Union[MockTensor, float]) -> MockTensor:
        if isinstance(other, MockTensor):
            return MockTensor([a * b for a, b in zip(self._data, other._data)])
        return MockTensor([a * other for a in self._data])

    def div(self, other: Union[MockTensor, float]) -> MockTensor:
        if isinstance(other, MockTensor):
            return MockTensor([a / b for a, b in zip(self._data, other._data)])
        return MockTensor([a / other for a in self._data])

    def sqrt(self) -> MockTensor:
        return MockTensor([math.sqrt(a) for a in self._data])

    def maximum(self, other: MockTensor) -> MockTensor:
        return MockTensor([max(a, b) for a, b in zip(self._data, other._data)])

    # ---- in-place ops (return self) -------------------------------------

    def add_(self, other: Union[MockTensor, float]) -> MockTensor:
        if isinstance(other, MockTensor):
            self._data = [a + b for a, b in zip(self._data, other._data)]
        else:
            self._data = [a + other for a in self._data]
        return self

    def mul_(self, other: Union[MockTensor, float]) -> MockTensor:
        if isinstance(other, MockTensor):
            self._data = [a * b for a, b in zip(self._data, other._data)]
        else:
            self._data = [a * other for a in self._data]
        return self

    def zero_(self) -> MockTensor:
        self._data = [0.0] * len(self._data)
        return self

    # ---- accessors ------------------------------------------------------

    def tolist(self) -> List[float]:
        return list(self._data)

    def __repr__(self) -> str:
        return f"MockTensor({self._data})"

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MockTensor):
            return NotImplemented
        return self._data == other._data


class MockParameter:
    """A mock parameter that mimics cakelamp.nn.Parameter.

    Has ``data`` and ``grad`` attributes, both of which are MockTensor.
    """

    def __init__(self, data: List[float]) -> None:
        self.data = MockTensor(data)
        self.grad: Optional[MockTensor] = None
        self.requires_grad = True

    def tolist(self) -> List[float]:
        return self.data.tolist()

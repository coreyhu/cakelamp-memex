"""Container modules for CakeLamp nn.

Provides Sequential and other module containers.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterator, Tuple, Union

from cakelamp.nn.module import Module


class Sequential(Module):
    """A sequential container.

    Modules are added in order and called sequentially during forward.
    The output of each module is fed as input to the next.

    Can be initialised with positional Module arguments or an
    OrderedDict of named modules.

    Examples
    --------
    >>> model = Sequential(
    ...     Linear(784, 128),
    ...     ReLU(),
    ...     Linear(128, 10),
    ... )
    >>> output = model(input_tensor)
    """

    def __init__(self, *args: Union[Module, OrderedDict]) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self._modules[key] = module
                super().__setattr__(key, module)
        else:
            for idx, module in enumerate(args):
                self._modules[str(idx)] = module
                super().__setattr__(str(idx), module)

    def forward(self, input: Any) -> Any:
        for module in self._modules.values():
            input = module(input)
        return input

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())

    def __getitem__(self, idx: int) -> Module:
        keys = list(self._modules.keys())
        if isinstance(idx, int):
            if idx < 0:
                idx += len(keys)
            return self._modules[keys[idx]]
        raise TypeError(f"indices must be integers, not {type(idx).__name__}")

    def append(self, module: Module) -> Sequential:
        """Append a module to the end of the sequential container."""
        idx = len(self._modules)
        self._modules[str(idx)] = module
        super().__setattr__(str(idx), module)
        return self

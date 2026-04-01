"""Base Module class for CakeLamp nn.

All neural network layers inherit from Module.  Provides automatic
parameter discovery, recursive children traversal, train/eval mode,
and state_dict serialisation.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

from cakelamp.nn.parameter import Parameter


class Module:
    """Base class for all neural network modules.

    Subclasses should implement :meth:`forward`.  The ``__call__``
    method delegates to ``forward`` (with hooks support in the future).
    """

    def __init__(self) -> None:
        self._parameters: OrderedDict[str, Parameter] = OrderedDict()
        self._modules: OrderedDict[str, Optional[Module]] = OrderedDict()
        self.training: bool = True

    # ------------------------------------------------------------------
    # Attribute registration
    # ------------------------------------------------------------------

    def __setattr__(self, name: str, value: Any) -> None:
        # Intercept Parameter and Module assignments so they register
        # automatically (like PyTorch).
        if isinstance(value, Parameter):
            # Remove from _modules if it existed there.
            modules = self.__dict__.get("_modules", {})
            if name in modules:
                del modules[name]
            params = self.__dict__.get("_parameters", {})
            params[name] = value
        elif isinstance(value, Module):
            params = self.__dict__.get("_parameters", {})
            if name in params:
                del params[name]
            modules = self.__dict__.get("_modules", {})
            modules[name] = value
        super().__setattr__(name, value)

    def __getattr__(self, name: str) -> Any:
        # Look up in _parameters and _modules *after* normal lookup fails.
        _parameters = self.__dict__.get("_parameters", {})
        if name in _parameters:
            return _parameters[name]
        _modules = self.__dict__.get("_modules", {})
        if name in _modules:
            return _modules[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

    # ------------------------------------------------------------------
    # Parameter / module access
    # ------------------------------------------------------------------

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        """Yield all parameters of this module.

        Parameters
        ----------
        recurse : bool
            If ``True``, also yield parameters of all sub-modules.
        """
        memo: Set[int] = set()
        for name, param in self.named_parameters(recurse=recurse):
            if id(param) not in memo:
                memo.add(id(param))
                yield param

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        """Yield ``(name, Parameter)`` pairs."""
        memo: Set[int] = set()
        for name, param in self._parameters.items():
            if id(param) not in memo:
                memo.add(id(param))
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, param
        if recurse:
            for mod_name, module in self._modules.items():
                if module is None:
                    continue
                sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                yield from module.named_parameters(
                    prefix=sub_prefix, recurse=True
                )

    def children(self) -> Iterator[Module]:
        """Yield immediate child modules."""
        for module in self._modules.values():
            if module is not None:
                yield module

    def named_children(self) -> Iterator[Tuple[str, Module]]:
        """Yield ``(name, Module)`` pairs for immediate children."""
        for name, module in self._modules.items():
            if module is not None:
                yield name, module

    def modules(self) -> Iterator[Module]:
        """Yield this module and all sub-modules recursively."""
        yield self
        for child in self.children():
            yield from child.modules()

    def named_modules(
        self, prefix: str = ""
    ) -> Iterator[Tuple[str, Module]]:
        """Yield ``(name, Module)`` pairs for this module and all sub-modules."""
        yield prefix, self
        for name, child in self.named_children():
            sub_prefix = f"{prefix}.{name}" if prefix else name
            yield from child.named_modules(prefix=sub_prefix)

    # ------------------------------------------------------------------
    # Forward / call
    # ------------------------------------------------------------------

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Define the computation performed at every call.

        Must be overridden by every subclass.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement forward()"
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    # ------------------------------------------------------------------
    # Train / eval
    # ------------------------------------------------------------------

    def train(self, mode: bool = True) -> Module:
        """Set the module (and all children) to training mode."""
        self.training = mode
        for child in self.children():
            child.train(mode)
        return self

    def eval(self) -> Module:
        """Set the module (and all children) to evaluation mode."""
        return self.train(False)

    # ------------------------------------------------------------------
    # State dict
    # ------------------------------------------------------------------

    def state_dict(self, prefix: str = "") -> Dict[str, Any]:
        """Return a flat dict of all parameter data, keyed by name."""
        result: Dict[str, Any] = {}
        for name, param in self._parameters.items():
            key = f"{prefix}{name}"
            result[key] = param.data
        for name, module in self._modules.items():
            if module is not None:
                child_prefix = f"{prefix}{name}."
                result.update(module.state_dict(prefix=child_prefix))
        return result

    def load_state_dict(self, state_dict: Dict[str, Any], prefix: str = "") -> None:
        """Load parameter data from a flat dict."""
        for name, param in self._parameters.items():
            key = f"{prefix}{name}"
            if key in state_dict:
                param.data = state_dict[key]
        for name, module in self._modules.items():
            if module is not None:
                child_prefix = f"{prefix}{name}."
                module.load_state_dict(state_dict, prefix=child_prefix)

    # ------------------------------------------------------------------
    # Zero grad
    # ------------------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset gradients for all parameters."""
        for param in self.parameters():
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def extra_repr(self) -> str:
        """Override this to add extra info to the repr string."""
        return ""

    def __repr__(self) -> str:
        lines = [f"{self.__class__.__name__}("]
        extra = self.extra_repr()
        if extra:
            lines[0] = f"{self.__class__.__name__}({extra}"
            if not self._modules:
                lines[0] += ")"
                return lines[0]
            lines[0] += ","

        for name, module in self._modules.items():
            mod_str = repr(module).replace("\n", "\n  ")
            lines.append(f"  ({name}): {mod_str}")
        lines.append(")")
        return "\n".join(lines)

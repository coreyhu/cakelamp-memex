"""Base Optimizer class for CakeLamp.

All optimizers inherit from this class and implement the `step()` method.
Mirrors the torch.optim.Optimizer API.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


class Optimizer:
    """Base class for all optimizers.

    Parameters
    ----------
    params : iterable
        An iterable of :class:`cakelamp.nn.Parameter` (or tensors with
        ``requires_grad=True``) that define which tensors will be optimized.
        Can also be an iterable of dicts (param groups) with per-group
        hyperparameters.
    defaults : dict
        Default hyperparameter values for all parameter groups.
    """

    def __init__(self, params: Any, defaults: Dict[str, Any]) -> None:
        self.defaults = defaults
        self.state: Dict[int, Dict[str, Any]] = defaultdict(dict)
        self.param_groups: List[Dict[str, Any]] = []

        # Accept either a flat iterable of params or a list of param-group dicts.
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")

        if not isinstance(param_groups[0], dict):
            # Flat list of parameters -> single group.
            param_groups = [{"params": param_groups}]

        for group in param_groups:
            self.add_param_group(group)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Reset the gradients of all optimized parameters.

        Parameters
        ----------
        set_to_none : bool
            If ``True`` (default), set ``.grad`` to ``None`` instead of
            filling with zeros.  This is more memory-efficient and matches
            PyTorch >= 1.7 behavior.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    if set_to_none:
                        p.grad = None
                    else:
                        p.grad.zero_()

    def step(self) -> None:
        """Perform a single optimization step (parameter update).

        Must be implemented by every concrete optimizer.
        """
        raise NotImplementedError(
            "Subclasses must implement the step() method"
        )

    # ------------------------------------------------------------------
    # Parameter-group management
    # ------------------------------------------------------------------

    def add_param_group(self, param_group: Dict[str, Any]) -> None:
        """Add a parameter group to the optimizer.

        Parameters
        ----------
        param_group : dict
            Must contain a ``"params"`` key whose value is an iterable of
            parameters.  Any other keys are treated as group-specific
            hyperparameters and override the optimizer defaults.
        """
        if not isinstance(param_group, dict):
            raise TypeError(f"param_group must be a dict, got {type(param_group)}")

        params = list(param_group.get("params", []))
        if len(params) == 0:
            raise ValueError("a parameter group must contain at least one parameter")
        param_group["params"] = params

        # Fill in defaults for keys not provided in this group.
        for key, value in self.defaults.items():
            param_group.setdefault(key, value)

        # Sanity-check: no parameter should appear in multiple groups.
        existing_param_ids = {
            id(p) for g in self.param_groups for p in g["params"]
        }
        for p in params:
            if id(p) in existing_param_ids:
                raise ValueError(
                    "some parameters appear in more than one parameter group"
                )

        self.param_groups.append(param_group)

    # ------------------------------------------------------------------
    # Serialisation helpers (state_dict / load_state_dict)
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        """Return the optimizer state as a nested dict.

        Contains two entries:
        * ``"state"`` – per-parameter state (momentum buffers, etc.).
        * ``"param_groups"`` – list of param group dicts (without the
          actual parameter objects, replaced by integer indices).
        """
        # Build a mapping from parameter id -> flat index.
        param_to_idx: Dict[int, int] = {}
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                param_to_idx[id(p)] = idx
                idx += 1

        # Pack state keyed by integer index.
        packed_state = {
            param_to_idx[pid]: s for pid, s in self.state.items()
            if pid in param_to_idx
        }

        # Pack param groups (replace params with indices).
        packed_groups = []
        for group in self.param_groups:
            packed = {k: v for k, v in group.items() if k != "params"}
            packed["params"] = [param_to_idx[id(p)] for p in group["params"]]
            packed_groups.append(packed)

        return {"state": packed_state, "param_groups": packed_groups}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state from a dict produced by :meth:`state_dict`.

        Parameters
        ----------
        state_dict : dict
            Optimizer state.  The structure must match the current optimizer
            (same number of param groups and parameters).
        """
        groups = state_dict["param_groups"]
        saved_state = state_dict["state"]

        if len(groups) != len(self.param_groups):
            raise ValueError(
                f"loaded state dict has {len(groups)} param groups, "
                f"but optimizer has {len(self.param_groups)}"
            )

        # Build index -> parameter mapping.
        idx_to_param_id: Dict[int, int] = {}
        idx = 0
        for group in self.param_groups:
            for p in group["params"]:
                idx_to_param_id[idx] = id(p)
                idx += 1

        # Restore per-parameter state.
        self.state = defaultdict(dict)
        for idx_str, s in saved_state.items():
            idx_int = int(idx_str) if isinstance(idx_str, str) else idx_str
            pid = idx_to_param_id[idx_int]
            self.state[pid] = s

        # Restore group hyperparameters (but keep current params references).
        for current_group, saved_group in zip(self.param_groups, groups):
            for key, value in saved_group.items():
                if key != "params":
                    current_group[key] = value

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        pieces = [self.__class__.__name__ + " ("]
        for i, group in enumerate(self.param_groups):
            pieces.append(f"Parameter Group {i}")
            for k, v in sorted(group.items()):
                if k == "params":
                    pieces.append(f"    {k}: {len(v)} parameters")
                else:
                    pieces.append(f"    {k}: {v}")
        pieces.append(")")
        return "\n".join(pieces)

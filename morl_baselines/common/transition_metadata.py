"""Protocols and helpers for optional per-transition metadata in RL buffers."""

import inspect
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np
import torch as th


@runtime_checkable
class TransitionMetadataCollector(Protocol):
    """Collects optional per-transition metadata during environment interaction.

    Implementations are algorithm- or environment-specific (e.g. running sums for
    episodic shaping). The returned dict keys and shapes must match
    ``transition_extra_fields`` passed to the replay buffer / agent.
    """

    def reset(self, env_id: int = 0) -> None:
        """Reset collector state when an episode ends in ``env_id``."""
        ...

    def step(self, reward: np.ndarray, done: bool, env_id: int = 0) -> Dict[str, np.ndarray]:
        """Update state and return metadata to store with the current transition."""
        ...


def apply_reward_transforms(
    transforms: List[Callable[..., th.Tensor]],
    rewards: th.Tensor,
    weights: th.Tensor,
    **optional_kwargs: Any,
) -> th.Tensor:
    """Apply reward transforms; passes only kwargs each transform accepts (or all via **kwargs)."""
    result = rewards
    for transform in transforms:
        sig = inspect.signature(transform)
        accepts_var_kw = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        kwargs: Dict[str, Any] = {"rewards": result, "weights": weights}
        if accepts_var_kw:
            kwargs.update(optional_kwargs)
        else:
            for name in sig.parameters:
                if name in optional_kwargs:
                    kwargs[name] = optional_kwargs[name]
        result = transform(**kwargs)
    return result

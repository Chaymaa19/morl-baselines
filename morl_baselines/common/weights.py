"""Utilities related to weight vectors."""

from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
from pymoo.util.ref_dirs import get_reference_directions


def random_weights(
    dim: int, n: int = 1, dist: str = "dirichlet", seed: Optional[int] = None, rng: Optional[np.random.Generator] = None,  dist_config: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """Generate random normalized weight vectors from a Gaussian or Dirichlet distribution alpha=1.

    Args:
        dim: size of the weight vector
        n : number of weight vectors to generate
        dist: distribution to use, either 'gaussian', 'dirichlet' or 'well_spaced'. Default is 'dirichlet' as it is equivalent to sampling uniformly from the weight simplex.
        seed: random seed
        rng: random number generator
        dist_config: optional dictionary with distribution-specific parameters.
            For 'well_spaced': {'a': float} where 'a' is the power law exponent (default: 3.0)
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    if dist == "gaussian":
        w = rng.standard_normal((n, dim))
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1, keepdims=True)
    elif dist == "dirichlet":
        w = rng.dirichlet(np.ones(dim), n)
    elif dist == "well_spaced":
        a = dist_config.get("a", 3.0)  # Default to 3.0 if not specified
        w = _random_well_spaced_weights(n, dim, a, rng)
    else:
        raise ValueError(f"Unknown distribution {dist}")

    if n == 1:
        return w[0]
    return w


@lru_cache(maxsize=128)
def _get_k_weights(dim: int, a: float) -> np.ndarray:
    """Cache k_weights calculation for well_spaced distribution."""
    k_weights = 1 / (np.arange(2, dim + 1) ** a)
    return k_weights / k_weights.sum()


def _random_well_spaced_weights(n: int, dim: int, a: float, rng: np.random.Generator) -> np.ndarray:
    """Generate random well-spaced weight vectors with sparse structure.

    This generates weights where each vector has k active entries (where k is sampled
    from a power-law distribution), and the active entries are uniformly distributed.

    Args:
        n: number of weight vectors to generate
        dim: dimension of each weight vector
        a: power law exponent for k distribution (higher = more sparse)
        rng: random number generator
    """
    # Get cached k_weights
    k_weights = _get_k_weights(dim, a)
    k_values = np.arange(2, dim + 1)

    # Step 1: sample k values for each weight vector
    ks = rng.choice(k_values, size=n, p=k_weights)

    # Step 2: create binary mask with exactly k_i active entries per row
    rand_matrix = rng.random((n, dim))
    sorted_indices = np.argsort(rand_matrix, axis=1)

    # Create mask to generate sparse vectors
    max_k = ks.max()
    rows = np.arange(n)[:, None]
    mask = np.zeros((n, dim), dtype=bool)
    # Only set mask for indices up to max_k, then filter by ks
    mask[rows, sorted_indices[:, :max_k]] = (
        np.arange(max_k)[None, :] < ks[:, None]
    )

    # Step 3: generate random weights for all entries
    w = rng.random((n, dim))
    w *= mask  # zero out inactive entries

    # Step 4: normalize rows to sum to 1 (only active entries matter)
    row_sums = w.sum(axis=1, keepdims=True)
    w /= np.where(row_sums == 0, 1, row_sums)

    return w


@lru_cache
def equally_spaced_weights(dim: int, n: int, seed: int = 42) -> List[np.ndarray]:
    """Generate weight vectors that are equally spaced in the weight simplex.

    It uses the Riesz s-Energy method from pymoo: https://pymoo.org/misc/reference_directions.html

    Args:
        dim: size of the weight vector
        n: number of weight vectors to generate
        seed: random seed
    """
    return list(get_reference_directions("energy", dim, n, seed=seed))


def extrema_weights(dim: int) -> List[np.ndarray]:
    """Generate weight vectors in the extrema of the weight simplex. That is, one element is 1 and the rest are 0.

    Args:
        dim: size of the weight vector
    """
    return list(np.eye(dim, dtype=np.float32))

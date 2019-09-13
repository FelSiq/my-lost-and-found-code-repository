"""Permutation testing experiments."""
import typing as t

import numpy as np


def _shuffle(x1: np.ndarray,
             x2: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
    """Shuffle (not in-place) arrays ``x1`` and ``x2`` values."""
    aux = np.random.permutation(np.concatenate((x1, x2)))
    return aux[:x1.size], aux[x1.size:]


def check_diff_mean(pop_a: np.ndarray,
                    pop_b: np.ndarray,
                    M: int = 1000,
                    random_seed: t.Optional[int] = None) -> float:
    """Tests whether the average values of ``pop_a`` and ``pop_b`` match.

    The test works as following:
    H_0 (null hypothesis): the mean value of ``pop_a`` and ``pop_b`` are equal.
    H_1 (alt hypothesis): the mean value of ``pop_a`` and ``pop_b`` are different.

    Arguments
    ---------
    M : :obj:`int`, optional
        Number of permutations.

    Returns
    -------
    :obj:`float`
        p-value of the permutation test.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    if pop_a.ndim != 1:
        pop_a = pop_a.ravel()

    if pop_b.ndim != 1:
        pop_b = pop_b.ravel()

    truediff = np.abs(pop_a.mean() - pop_b.mean())

    as_extreme_val = 0
    for i in np.arange(M):
        sh_a, sh_b = _shuffle(pop_a, pop_b)
        as_extreme_val += truediff <= np.abs(sh_a.mean() - sh_b.mean())

    return (as_extreme_val + 1) / (M + 1)


if __name__ == "__main__":
    random_seed = 16
    np.random.seed(random_seed)

    pop_a = np.random.normal(size=500)
    pop_b = np.random.normal(size=500) + 0.5
    print("pop_a.mean = {}\npop_b.mean = {}".format(pop_a.mean(),
                                                    pop_b.mean()))

    p_val = check_diff_mean(pop_a, pop_b, M=1000, random_seed=random_seed)
    print("permutation test p-value: {}".format(p_val))

"""All-pair shortest paths with repeated 'matrix multiplication'.

This algorithm runs in Theta(V**3 logV).
"""
import typing as t

import numpy as np


def _extend_shortest_path(L: np.ndarray, v_ids: t.Sequence[int]) -> np.ndarray:
    L_new = np.full_like(L, np.inf)

    for i in v_ids:
        for j in v_ids:
            for k in v_ids:
                L_new[i, j] = min(L_new[i, j], L[i, k] + L[k, j])

    return L_new


def apsp_rm(adj_matrix: np.ndarray) -> np.ndarray:
    """."""
    power = 1
    num_vert = adj_matrix.shape[0]
    _v_ids = np.arange(num_vert)

    L = adj_matrix

    while power < num_vert - 1:
        L = _extend_shortest_path(L, v_ids=_v_ids)
        power *= 2

    if id(adj_matrix) == id(L):
        return np.copy(adj_matrix)

    return L


def _test() -> None:
    W = np.array([
        [0, 3, 8, np.inf, -4],
        [np.inf, 0, np.inf, 1, 7],
        [np.inf, 4, 0, np.inf, np.inf],
        [2, np.inf, -5, 0, np.inf],
        [np.inf, np.inf, np.inf, 6, 0],
    ])

    print(W)

    res = apsp_rm(W)

    print(res)


if __name__ == "__main__":
    _test()

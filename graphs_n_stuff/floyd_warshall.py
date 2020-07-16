"""Floyd-Warshall algorithm for all pair shortest distance.

Runs in Theta(V**3) using DP.
"""
import typing as t

import numpy as np


def _calc_predecessors(
    D: np.ndarray, adj_matrix: np.ndarray, iterator: np.ndarray
) -> np.ndarray:
    predecessors = np.full_like(D, -1)

    for i in iterator:
        for j in iterator:
            if i != j:
                for k in iterator:
                    if k != j and np.isclose(D[i, k] + adj_matrix[k, j], D[i, j]):
                        predecessors[i, j] = k

    return predecessors


def floyd_warshall(adj_matrix: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
    n = adj_matrix.shape[0]

    D = np.copy(adj_matrix)

    _iterator = np.arange(n)

    for k in _iterator:
        for i in _iterator:
            for j in _iterator:
                D[i, j] = min(D[i, j], D[i, k] + D[k, j])

    predecessors = _calc_predecessors(D, adj_matrix, _iterator)

    return D, predecessors


def _test() -> None:
    W = np.array(
        [
            [0, 3, 8, np.inf, -4],
            [np.inf, 0, np.inf, 1, 7],
            [np.inf, 4, 0, np.inf, np.inf],
            [2, np.inf, -5, 0, np.inf],
            [np.inf, np.inf, np.inf, 6, 0],
        ]
    )

    print(W)

    res, pred = floyd_warshall(W)

    print(res)
    print(pred)


if __name__ == "__main__":
    _test()

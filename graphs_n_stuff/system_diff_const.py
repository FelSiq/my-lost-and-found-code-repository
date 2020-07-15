"""Solve a system of difference constraints.

Uses Bellman-Ford algorithm.

This version runs in O(n**2 + nm) time, where n is the number of unknowns and
m is the number of constraints. There is a version of this algorithm which
runs in O(mn) time.
"""
import numpy as np

import bellman_ford


def solve(A: np.ndarray, b: np.ndarray, check_A: bool = True) -> np.ndarray:
    """Solve a system of difference constraints Ax <= b.

    The A matrix must have a single '1' and a single '-1' in each row. All
    other entries must be 0.
    """
    if b.ndim > 1:
        b = b.ravel()

    if A.shape[0] != b.size:
        raise ValueError("Number of rows in A must match the size of b.")

    if A.shape[1] > A.shape[0]:
        raise ValueError(
            "Number of constraints can't be smaller than number of variables."
        )

    graph = {i: [] for i in np.arange(A.shape[1])}

    for const, upper_b in zip(A, b):
        ind_pos = np.flatnonzero(const == +1)[0]
        ind_neg = np.flatnonzero(const == -1)[0]
        graph[ind_neg].append((ind_pos, upper_b))

    graph[-1] = [(i, 0) for i in np.arange(A.shape[1])]

    sol, success = bellman_ford.bellman_ford(graph, source=-1)

    if not success:
        raise ValueError("Negative weight cycle found: system has no solution.")

    # Remove the artificial node from the bellman-ford results
    sol.pop(-1)

    # Solution is the smallest distance from each node to the artificial node
    return np.fromiter((d for _, d in sol.values()), dtype=float)


def _test() -> None:
    constraints = np.array(
        [
            [1, -1, 0, 0, 0],
            [1, 0, 0, 0, -1],
            [0, 1, 0, 0, -1],
            [-1, 0, 1, 0, 0],
            [-1, 0, 0, 1, 0],
            [0, 0, -1, 1, 0],
            [0, 0, -1, 0, 1],
            [0, 0, 0, -1, 1],
        ],
        dtype=int,
    )

    upper_b = np.array([0, -1, 1, 5, 4, -1, -3, -3])

    print(constraints)
    print(upper_b)

    sol = solve(A=constraints, b=upper_b)

    print(sol)


if __name__ == "__main__":
    _test()

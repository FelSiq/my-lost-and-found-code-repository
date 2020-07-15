"""Find the tastiest ingredient combination.

The rules are the following:

1. There is a NxM garden, with each cell containing a distinct ingredient.
   All ingredients (i, j) have a positive value P(i, j) > 0.

2. The Chef will pick a coordinate (i, j), 0 < i < N-1 and 0 < j < M-1,
   dividing the garden into four quadrants:

   a) LT (left-top):     all cells (x, y) with x < i and y < j.
   b) LB (left-bottom):  all cells (x, y) with x < i and y > j.
   c) RT (right-top):    all cells (x, y) with x > i and y < j.
   d) RB (right-bottom): all cells (x, y) with x > i and y > j.

   Emphasizing that neither the row i nor the column j are not in any
   of the four quadrants.

   Also note that the coordinates (i, j) are such that every quadrant has
   at least one cell available.

3) The Chef will then pick a single ingredient from each of the quadrants.
   The taste quality of the cooked dish is measured by the product of the
   values of each ingredient.

4) Find the tastiest combination available in the garden efficiently.
"""
import numpy as np


def _max_lt(tastiness: np.ndarray, memo_lt: np.ndarray) -> float:
    n, m = tastiness.shape

    memo_lt[1, 1] = tastiness[0, 0]

    for a in np.arange(1, n - 1):
        for b in np.arange(1, m - 1):
            c = max(tastiness[a-1, b-1], memo_lt[a-1, b], memo_lt[a, b-1])
            memo_lt[a, b] = c

    return memo_lt


def _max_rt(tastiness: np.ndarray, memo_rt: np.ndarray) -> float:
    n, m = tastiness.shape

    memo_rt[1, m - 2] = tastiness[0, m - 1]

    for a in np.arange(1, n - 1):
        for b in np.arange(m - 2, 1, -1):
            c = max(tastiness[a-1, b+1], memo_rt[a-1, b], memo_rt[a, b+1])
            memo_rt[a, b] = c

    return memo_rt


def _max_rb(tastiness: np.ndarray, memo_lb: np.ndarray) -> float:
    n, m = tastiness.shape

    memo_lb[n - 2, m - 2] = tastiness[n - 1, m - 1]

    for a in np.arange(n - 2, 1, -1):
        for b in np.arange(m - 2, 1, -1):
            c = max(tastiness[a+1, b+1], memo_lb[a+1, b], memo_lb[a, b+1])
            memo_lb[a, b] = c

    return memo_lb


def _max_lb(tastiness: np.ndarray, memo_lb: np.ndarray) -> float:
    n, m = tastiness.shape

    memo_lb[n - 2, 1] = tastiness[n - 1, 0]

    for a in np.arange(n - 2, 1, -1):
        for b in np.arange(1, m - 1):
            c = max(tastiness[a+1, b-1], memo_lb[a+1, b], memo_lb[a, b-1])
            memo_lb[a, b] = c

    return memo_lb


def max_qd(tastiness: np.ndarray, qd: str) -> np.ndarray:
    memo = np.zeros_like(tastiness, dtype=float)

    n, m = tastiness.shape

    if qd == "LT":
        _max_lt(tastiness, memo)

    elif qd == "RB":
        _max_rb(tastiness, memo)

    elif qd == "LB":
        _max_lb(tastiness, memo)

    elif qd == "RT":
        _max_rt(tastiness, memo)

    return memo


def max_tastiness(tastiness: np.ndarray) -> np.ndarray:
    res = np.ones_like(tastiness, dtype=float)

    for qd in ("LT", "LB", "RT", "RB"):
        res *= max_qd(tastiness, qd)

    max_i, max_j, max_t = 0, 0, res[0, 0]

    n, m = tastiness.shape

    for i in np.arange(n):
        for j in np.arange(m):
            if res[i, j] > res[max_i, max_j]:
                max_i, max_j = i, j
                max_t = res[i, j]

    return max_t, max_i, max_j


def _test() -> None:
    np.random.seed(32)
    tastiness = np.random.randint(1, 19, size=(8, 4))
    print(tastiness)
    print(max_tastiness(tastiness))


if __name__ == "__main__":
    _test()

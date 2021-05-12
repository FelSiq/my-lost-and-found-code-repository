import numpy as np

import rref


class LinearSolver:
    def __init__(self, return_all_sol: bool = True):
        self.return_all_sol = return_all_sol

        self._rref = rref.RREF()

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        """Solve Xw = y.

        X shape is (n, m);
        w shape is (m, 1);
        y shape is (n, 1).

        Returns
        -------
        If `return_all_sol`=False, return a particular solution with
            shape (m,).

        If `return_all_sol`=True, return an array with shape (m, 1 + m - r),
        where `r` is rank(X), where the first column as a particular
        solution for Xw = b, and all subsequent columns form a basis to the
        Null Space of X. In other words, all possible solutions for Xw = b
        are the particular solution (res[:, 0]) plus all linear combinations
        of the special solutions (res[:, 1:]).
        """
        X, y = self._rref.transform(X, y)

        n, m = X.shape

        pivot_inds = self._rref.col_inds_pivot
        free_inds = self._rref.col_inds_free
        rank = self._rref.rank

        assert np.allclose(0.0, y[rank:]), "System unsolvable"

        sol_particular = np.zeros(m, dtype=float)
        sol_particular[pivot_inds] = y[:rank]

        if not self.return_all_sol:
            return sol_particular

        sols_special = np.vstack((-X[:rank, free_inds], np.eye(m - rank)))

        sols = np.column_stack((sol_particular, sols_special))

        return sols


def _test():
    solver = LinearSolver()
    np.random.seed(1)

    n, m = np.random.randint(1, 10, size=2)
    print(n, m)
    X = np.random.randn(n, m)
    coeffs = np.random.randn(m)
    y = np.zeros(n)

    res = solver.transform(X, y)

    assert np.allclose(y, X @ res[:, 0])

    if res.shape[1] > 1:
        assert np.allclose(0.0, X @ res[:, 1:])

    print(res)


if __name__ == "__main__":
    _test()

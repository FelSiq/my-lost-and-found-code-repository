import numpy as np


class RREF:
    def __init__(self, threshold: float = 1e-7, copy: bool = True):
        self.copy = copy

        self.rank = -1
        self.col_inds_pivot = []
        self.col_inds_free = []
        self.threshold = float(threshold)

    def _pick_pivot_ind(self, X, r: int, c: int) -> int:
        pivot_ind = r + int(np.argmax(np.abs(X[r:, c])))

        if np.abs(X[pivot_ind, c]) < self.threshold:
            return -1

        return pivot_ind

    @staticmethod
    def _switch_rows(X, ind_a, ind_b):
        if ind_a == ind_b:
            return

        X[[ind_a, ind_b], :] = X[[ind_b, ind_a], :]

    @staticmethod
    def _elimination(X, r, c, reverse: bool):
        n, _ = X.shape
        pivot = X[r, c]

        range_it = range(r + 1, n) if not reverse else range(r - 1, -1, -1)

        for i in range_it:
            elim_val = X[i, c] / pivot
            X[i, :] -= elim_val * X[r, :]

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.copy and y is None:
            X = np.array(X, dtype=float)

        X = np.asfarray(X)
        n, m = X.shape

        if y is not None:
            X = np.column_stack((X, y))

        self.rank = 0
        self.col_inds_pivot = []
        self.col_inds_free = []
        r = c = 0

        while r < n and c < m:
            pivot_ind = self._pick_pivot_ind(X, r, c)

            if pivot_ind < 0:
                self.col_inds_free.append(c)
                c += 1
                continue

            self._switch_rows(X, r, pivot_ind)
            self._elimination(X, r, c, reverse=False)
            X[r, :] /= X[r, c]

            self.col_inds_pivot.append(c)
            self.rank += 1
            r += 1
            c += 1

        r -= 1

        for c in reversed(self.col_inds_pivot):
            self._elimination(X, r, c, reverse=True)
            r -= 1

        if y is not None:
            rref_X, rref_y = X[:, :m], X[:, m:]
            return rref_X, rref_y.reshape(y.shape)

        return X


def _test():
    X = np.array(
        [
            [1, 2, 3],
            [7, 8, 1],
            [7, 8, 2],
        ]
    )

    rref = RREF()
    res_X = rref.transform(X)

    print(X)
    print(res_X)


if __name__ == "__main__":
    _test()

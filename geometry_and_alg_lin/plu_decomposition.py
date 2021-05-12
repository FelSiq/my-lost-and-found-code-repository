import typing as t

import numpy as np


def _search_partial_pivot_ind(U, r, c):
    partial_pivot_ind = r + np.argmax(np.abs(U[r:, c]))

    if np.abs(U[partial_pivot_ind, c]) < 1e-7:
        # Note: just an invalid index
        partial_pivot_ind = U.shape[0] + 1

    return partial_pivot_ind


def _eliminate_column(U, r, c):
    n = U.shape[0]
    pivot = U[r, c]

    coeffs = np.empty(n - r - 1, dtype=float)

    for j in range(r + 1, n):
        coeff = U[j, c] / pivot
        U[j, :] -= coeff * U[r, :]
        coeffs[j - r - 1] = coeff

    return coeffs


def _permute_rows(i, j, *args):
    if i == j:
        return

    for arr in args:
        arr[[i, j], :] = arr[[j, i], :]


def _permute_cols(i, j, *args):
    if i == j:
        return

    for arr in args:
        arr[:, [i, j]] = arr[:, [j, i]]


def plu(M):
    """Decompose M = PLU

    P is a permutation matrix;
    L is a lower triangular matrix; and
    U is a upper triangular matrix.
    """
    U = np.array(M, dtype=float)

    n, m = U.shape
    k = min(n, m)

    P = np.eye(n, dtype=np.uint8)
    L = np.eye(n, dtype=float)

    r, c = 0, 0

    while r < min(k, n - 1) and c < k:
        partial_pivot_ind = _search_partial_pivot_ind(U, r, c)

        if partial_pivot_ind < n:
            _permute_rows(partial_pivot_ind, r, U, L)
            _permute_cols(partial_pivot_ind, r, L, P)

            coeffs = _eliminate_column(U, r, c)
            L[r + 1 :, r] = coeffs

        r += 1
        c += 1

    L = L[:, :k]
    U = U[:k, :]

    return P, L, U


def _test():
    import scipy.linalg

    k = 20000

    np.random.seed(128)

    not_match = 0

    for _ in np.arange(k):
        r, c = np.random.randint(1, 6, size=2)
        arr = np.random.randn(r, c)
        arr *= np.random.randint(2, size=arr.shape)

        P, L, U = plu(arr)
        sP, sL, sU = scipy.linalg.lu(arr, check_finite=False)

        if not (
            np.allclose(sP, P.astype(float))
            and np.allclose(sL, L)
            and np.allclose(sU, U)
        ):
            not_match += 1

        res = P @ L @ U
        assert np.allclose(res, arr)

    print("Not matches:", not_match / k)


if __name__ == "__main__":
    _test()

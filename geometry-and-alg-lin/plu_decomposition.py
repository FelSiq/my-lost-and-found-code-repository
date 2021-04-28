import typing as t

import numpy as np


def _search_pivot_ind(U, r, c):
    pivot_ind = r + np.argmax(np.abs(U[r:, c]))

    if np.abs(U[pivot_ind, c]) < 1e-6:
        # Note: just an invalid index
        pivot_ind = U.shape[0]

    return pivot_ind


def _eliminate_column(U, r, c):
    n = U.shape[0]
    pivot = U[r, c]

    coeffs = np.zeros(n - r - 1, dtype=float)

    for j in range(r + 1, n):
        coeff = U[j, c] / pivot
        U[j, :] -= coeff * U[r, :]
        coeffs[j - r - 1] = coeff

    return coeffs


def _permute_rows(i, j, *args):
    if i == j:
        return

    for arr in args:
        arr[[i, j]] = arr[[j, i]]


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
    M = np.asfarray(M)
    n, m = M.shape

    P = np.eye(n, dtype=np.uint8)
    L = np.eye(n, dtype=float)
    U = np.copy(M)

    r, c = 0, 0

    while r < n - 1 and c < m:
        pivot_ind = _search_pivot_ind(U, r, c)

        if pivot_ind >= n:
            c += 1
            continue

        _permute_rows(pivot_ind, r, U, L)
        _permute_cols(pivot_ind, r, L, P)

        coeffs = _eliminate_column(U, r, c)
        L[r + 1 :, r] = coeffs

        r += 1
        c += 1

    k = min(n, m)
    L = L[:, :k]
    U = U[:k, :]

    return P, L, U


def _test():
    import scipy.linalg

    k = 1000

    np.random.seed(32)
    match = 0

    for _ in np.arange(k):
        r, c = np.random.randint(1, 20, size=2)
        arr = np.random.randn(r, c)
        arr *= np.random.randint(2, size=arr.shape)

        P, L, U = plu(arr)
        sP, sL, sU = scipy.linalg.lu(arr, check_finite=False)

        equal = (
            np.allclose(sP, P.astype(float))
            and np.allclose(sL, L)
            and np.allclose(sU, U)
        )
        if not equal:
            print()
            print(np.round(arr, 3))
            print(np.round(P, 3))
            print(np.round(sP, 3))
            print(np.round(L, 3))
            print(np.round(sL, 3))
            print(np.round(U, 3))
            print(np.round(sU, 3))
            print()

        match += int(equal)

        res = P @ L @ U
        assert np.allclose(res, arr)

    print(match / k)


if __name__ == "__main__":
    _test()

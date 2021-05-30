import numpy as np


def _prepare_mat(mat):
    mat = np.asfarray(mat)
    assert mat.shape[0] == mat.shape[1]
    assert np.allclose(mat, mat.T)
    return mat


def cholesky(mat):
    mat = _prepare_mat(mat)
    L = np.zeros_like(mat)
    n = mat.shape[0]

    for c in np.arange(n):
        col_slice = L[c, :c]
        L[c, c] = float(np.sqrt(mat[c, c] - np.dot(col_slice, col_slice)))

        if np.isclose(L[c, c], 0.0):
            raise ValueError("Cholesky failed.")

        for r in np.arange(c + 1, n):
            aux = float(np.dot(L[r, :c], col_slice))
            L[r, c] = (mat[r, c] - aux) / L[c, c]

    return L


def _test():
    n = 10000
    np.random.seed(32)
    for _ in np.arange(n):
        m = np.random.randint(1, 10)
        mat = np.random.randn(m, m)
        mat[np.diag_indices_from(mat)] = np.abs(np.diag(mat))
        mat += mat.T + (3 * m) * np.eye(m)

        cho = cholesky(mat)
        cho_ref = np.linalg.cholesky(mat)

        assert np.allclose(cho, cho_ref)


if __name__ == "__main__":
    _test()

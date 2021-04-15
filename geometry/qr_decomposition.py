import numpy as np

import gram_schmidt


def qr(mat: np.ndarray):
    # (n, m)
    Q = gram_schmidt.gram_schmidt(mat)
    # (n, k) -> Q.T: (k, n)
    R = np.dot(Q.T, mat)
    # (k, n) . (n, m): (k, m)
    return Q, R


def _test():
    mats = np.random.randn(30, 7, 6)
    for mat in mats:
        Q, R = qr(mat)
        dot = Q @ R
        assert np.allclose(dot, mat), dot - mat
        assert np.isclose(0, float(np.sum(R[np.tril_indices_from(R, k=-1)])))


if __name__ == "__main__":
    _test()

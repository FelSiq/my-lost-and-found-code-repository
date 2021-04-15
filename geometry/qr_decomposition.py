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
    mats = np.random.randn(1000, 2, 15)
    for mat in mats:
        Q_ref, R_ref = np.linalg.qr(mat)
        Q, R = qr(mat)
        assert np.allclose(np.abs(Q.T @ Q_ref), np.eye(Q.shape[1]))
        assert np.allclose(np.abs(R), np.abs(R_ref))


if __name__ == "__main__":
    _test()

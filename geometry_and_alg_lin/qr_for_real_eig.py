"""Compute eigenvalues for symmetric real matrices."""
import numpy as np

import qr_decomposition
import linear_solver


def _prepare_mat(mat):
    mat = np.asfarray(mat)
    assert mat.shape[0] == mat.shape[1]
    assert np.allclose(mat.T, mat)
    return mat


def compute_real_eig(mat, num_iter: int = 512):
    assert int(num_iter) > 0

    mat = _prepare_mat(mat)

    for _ in np.arange(num_iter):
        Q, R = qr_decomposition.qr(mat)
        mat = R @ Q

    n = mat.shape[0]

    eigenvalues = np.array(sorted(np.diag(mat), key=abs, reverse=True))
    eigenvectors = np.empty((n, n))

    solver = linear_solver.LinearSolver(return_all_sol=False)
    y = np.zeros(n)

    for i, eig_val in enumerate(eigenvalues):
        cur_mat = mat - eig_val * np.eye(n)
        eig_vec = solver.transform(cur_mat, y)
        eigenvectors[:, i] = eig_vec / (1e-8 + float(np.linalg.norm(eig_vec)))

    return eigenvalues, eigenvectors


def _test():
    np.random.seed(16)
    n = 100

    matches = 0

    for _ in np.arange(n):
        m = np.random.randint(1, 12)
        mat = np.random.randn(m, m)
        mat += mat.T
        e_val, e_vec = compute_real_eig(mat)
        e_val_ref, e_vec_ref = np.linalg.eig(mat)

        matches += np.allclose(sorted(e_val), sorted(e_val_ref), rtol=0.05)

    print(f"Eigenvalues matches prop: {100. * matches / n:.3f}")


if __name__ == "__main__":
    _test()

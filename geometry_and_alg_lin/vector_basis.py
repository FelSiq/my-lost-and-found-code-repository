"""Create a (non-orthonormal) vector basis from a set of vectors."""
import numpy as np

import rref


def _prepare_vectors(vectors):
    vectors = np.asfarray(vectors)
    assert vectors.ndim == 2
    return vectors


def create_basis_method_1(vectors):
    vectors = _prepare_vectors(vectors)
    transformer = rref.GaussJordanElimination()
    basis = transformer.transform(vectors.T).T
    basis = basis[:, transformer.col_inds_pivot]
    return basis


def create_basis_method_2(vectors):
    vectors = _prepare_vectors(vectors)
    transformer = rref.GaussJordanElimination()
    transformer.transform(vectors)
    basis = vectors[:, transformer.col_inds_pivot]
    return basis


def _test():
    np.random.seed(32)
    n = 1000

    for _ in np.arange(n):
        n, m = np.random.randint(2, 10, size=2)
        mat = np.random.randn(n, m)
        dup_quant = np.random.randint(0, 10)

        if dup_quant:
            dup_inds = np.random.randint(m, size=dup_quant)
            dup_coeffs = np.random.random(size=dup_quant)
            mat = np.hstack((mat, mat[:, dup_inds] * dup_coeffs))

        basis_1 = create_basis_method_1(mat)
        basis_2 = create_basis_method_2(mat)

        rank = min(n, m)

        assert basis_1.shape[1] == rank, (basis_1.shape, rank)
        assert basis_2.shape[1] == rank, (basis_2.shape, rank)


if __name__ == "__main__":
    _test()

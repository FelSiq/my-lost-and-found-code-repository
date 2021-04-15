import numpy as np

import vector_projection


def gram_schmidit(vectors: np.ndarray) -> np.ndarray:
    k = min(*vectors.shape)
    basis = np.copy(vectors[:, :k]).astype(float, copy=False)

    for i, vec in enumerate(basis.T):
        for prev_vec in basis.T[:i]:
            proj = vector_projection.project_u_onto_v(
                u=vec, v=prev_vec, assume_v_unitary=True
            )
            vec -= proj

        vec /= np.linalg.norm(vec)

    return basis


def _test():
    mats = np.random.randn(30, 7, 7)

    for mat in mats:
        res = gram_schmidit(mat)

        for i in range(res.shape[1] - 1):
            for j in range(i + 1, res.shape[1]):
                dot = np.dot(res[:, i], res[:, j])
                assert np.isclose(0.0, dot), dot


if __name__ == "__main__":
    _test()

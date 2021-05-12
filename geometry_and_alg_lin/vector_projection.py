"""Projection of vector u onto v."""
import numpy as np


def magnitude_of_projection_of_u_onto_v(u, v):
    # fact 1: cos(u, v) = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    # fact 2: cos(u, v) = np.linalg.norm(proj) / np.linalg.norm(u)
    # Therefore: np.linalg.norm(proj) = np.dot(u, v) / np.linalg.norm(v)
    return np.dot(u, v) / float(np.linalg.norm(v))


def direction_of_projection(v):
    # Since the projection of u onto v has the same direction of v,
    # the unity vector of the projection is the same unity vector of v.
    return v / float(np.linalg.norm(v))


def slow_project_u_onto_v(u, v):
    return magnitude_of_projection_of_u_onto_v(u, v) * direction_of_projection(v)


def project_u_onto_v(u, v, assume_v_unitary: bool = False):
    # Just combinint all functions above into one.
    proj = np.dot(u, v) * v

    if not assume_v_unitary:
        proj /= float(np.linalg.norm(v)) ** 2

    return proj


def projection_mat(v, assume_v_unitary: bool = False):
    proj_mat = np.outer(v, v)

    if not assume_v_unitary:
        proj_mat /= float(np.dot(v, v))

    return proj_mat


def _test():
    num_tests = 10000
    np.random.seed(32)
    for i, (u, v) in enumerate(np.random.randn(num_tests, 2, 17), 1):
        print(f"Test {i} ...")
        assert np.allclose(project_u_onto_v(u, v), slow_project_u_onto_v(u, v))
        assert np.allclose(projection_mat(v) @ u, project_u_onto_v(u, v))


if __name__ == "__main__":
    _test()

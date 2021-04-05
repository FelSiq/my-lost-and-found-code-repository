"""Projection of vector u onto v."""
import numpy as np


def magnitude_of_projection_of_u_onto_v(u, v):
    # fact 1: cos(u, v) = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    # fact 2: cos(u, v) = np.linalg.norm(proj) / np.linalg.norm(u)
    # Therefore: np.linalg.norm(proj) = np.dot(u, v) / np.linalg.norm(v)
    return np.dot(u, v) / np.linalg.norm(v)


def direction_of_projection(v):
    # Since the projection of u onto v has the same direction of v,
    # the unity vector of the projection is the same unity vector of v.
    return v / np.linalg.norm(v)


def slow_project_u_onto_v(u, v):
    return magnitude_of_projection_of_u_onto_v(u, v) * direction_of_projection(v)


def project_u_onto_v(u, v):
    # Just combinint all functions above into one.
    return np.dot(u, v) / (np.linalg.norm(v) ** 2) * v


def _test():
    num_tests = 10000
    np.random.seed(32)
    for i, (u, v) in enumerate(np.random.randn(num_tests, 2), 1):
        print(f"Test {i} ...")
        assert np.isclose(project_u_onto_v(u, v), slow_project_u_onto_v(u, v))


if __name__ == "__main__":
    _test()

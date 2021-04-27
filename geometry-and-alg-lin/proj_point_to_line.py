"""Project a 2-D point to a 2-D line."""
import matplotlib.pyplot as plt
import numpy as np


def proj_point_to_line_analytical(p, line_coeffs):
    b, a = line_coeffs
    x, y = p

    a_orthogonal = -1.0 / a
    b_orthogonal = y - a_orthogonal * x

    x_proj = -(b_orthogonal - b) / (a_orthogonal - a)
    y_proj = a_orthogonal * x_proj + b_orthogonal

    proj = np.asfarray([x_proj, y_proj])

    return proj


def proj_point_to_line_vector(p, line_coeffs):
    b, a = line_coeffs

    l_vec = np.asfarray([1.0, a])
    shift = np.asfarray([0.0, b])

    p = np.asfarray(p) - shift
    proj = float(np.dot(p, l_vec)) / float(np.dot(l_vec, l_vec)) * l_vec
    proj += shift

    return proj


def _test():
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    method = proj_point_to_line_analytical

    line_coeffs = [-5, 5 / 3]
    p = [-4, 5]
    res = method(p, line_coeffs)
    expected = (57 / 17, 10 / 17)
    assert np.allclose(res, expected)

    for i in np.arange(4):
        point = 16 * np.random.randn(2)
        b, a = 16 * np.random.randn(2)
        line_coeffs = [b, a]
        proj_point = method(point, line_coeffs)

        min_plot, max_plot = np.quantile([*point, *proj_point], (0, 1))
        min_plot -= 5
        max_plot += 5

        ax = axes[i // 2][i % 2]
        ax.set_xlim(min_plot, max_plot)
        ax.set_ylim(min_plot, max_plot)
        ax.plot(
            [min_plot, max_plot],
            [b + a * min_plot, b + a * max_plot],
            label="line",
            color="red",
        )
        ax.scatter(*point, label="original point", color="black")
        ax.scatter(*proj_point, label="projection", color="blue")
        ax.plot(
            [point[0], proj_point[0]],
            [point[1], proj_point[1]],
            linestyle="--",
            color="orange",
        )
        ax.legend()

        test = np.dot(proj_point - point, [1, a])
        assert np.isclose(0, test), test

    plt.show()


if __name__ == "__main__":
    _test()

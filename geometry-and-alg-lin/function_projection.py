import numpy as np
import scipy.integrate


def func_inner_product(f, g, dx):
    return scipy.integrate.trapezoid(f * np.conjugate(g), dx=dx)


def func_norm(f, dx):
    return np.sqrt(func_inner_product(f, f, dx))


def magnitude_of_projection_of_f_onto_g(f, g, dx):
    return func_inner_product(f, g, dx) / func_norm(g, dx)


def direction_of_projection(g, dx):
    return g / func_norm(g, dx)


def slow_project_f_onto_g(f, g, dx):
    return magnitude_of_projection_of_f_onto_g(f, g, dx) * direction_of_projection(
        g, dx
    )

def project_f_onto_g(f, g, dx):
    return func_inner_product(f, g, dx) / (func_norm(g, dx) ** 2) * g


def _test():
    import matplotlib.pyplot as plt

    t = np.linspace(0, 4, 100)
    dx = 4 / 100
    f = 0.5 * t + 0.1 * t ** 2 + 0.3
    g = 0.5 * np.cos(2 * np.pi * t) + 0.5 * np.sin(t) + 0.05 * t

    proj = project_f_onto_g(f, g, dx)
    proj_aux = slow_project_f_onto_g(f, g, dx)

    assert np.allclose(proj, proj_aux)

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    ax1.plot(f, label="f")
    ax1.plot(g, label="g")
    ax1.legend()

    ax2.plot(proj, label="proj")
    ax2.legend()

    ax3.plot(f, label="f")
    ax3.plot(g, label="g")
    ax3.plot(proj, label="proj")
    ax3.legend()

    plt.show()


if __name__ == "__main__":
    _test()

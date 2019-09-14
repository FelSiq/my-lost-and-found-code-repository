"""Tests with kernel density estimation."""
import typing as t

import numpy as np


def kernel_gaussian(x: np.ndarray) -> np.ndarray:
    """Gaussian kernel."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    _, dimension = x.shape

    aux = np.sum(np.square(x), axis=1)
    return np.exp(-0.5 * aux) * np.power(2.0 * np.pi, -0.5 * dimension)


def infinity_norm(x: t.Union[np.ndarray, float]) -> t.Union[np.ndarray, float]:
    return np.max(np.abs(x), axis=1)


def kernel_uniform(x: np.ndarray) -> np.ndarray:
    """Uniform kernel."""
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    _, dimension = x.shape

    aux = infinity_norm(x)
    return 0.5**dimension * (infinity_norm(x) < 1.0)


def kernel_epanechnikov(x: np.ndarray) -> np.ndarray:
    """Epanechnikov kernel."""
    if x.ndim != 1:
        raise ValueError("'x' vector must be one-dimensional!")

    return 0.75 * (1.0 - np.square(x)) * (np.abs(x) < 1.0)


def kernel_density_est(
        unknown_points: np.ndarray,
        kernel: t.Callable[[np.ndarray], np.ndarray], known_points: np.ndarray,
        kernel_bandwidth: t.Union[int, float, np.number]) -> np.ndarray:
    """Calculate the density estimation for ``unknown points`` using ``kernel``.

    Arguments
    ---------
    """
    if isinstance(unknown_points, (float, int, np.number)):
        unknown_points = np.array([unknown_points])

    dimension = 1 if known_points.ndim == 1 else known_points.shape[1]

    return np.array([
        np.sum(kernel((x - known_points) / kernel_bandwidth))
        for x in unknown_points
    ]) / (known_points.size * kernel_bandwidth**dimension)


def _experiment_01() -> None:
    """Kernel Density estimation experiment 01."""
    import matplotlib.pyplot as plt

    t = np.linspace(-3, 3, 100)
    plt.plot(t, kernel_gaussian(t), label="Gaussian")
    plt.plot(t, kernel_uniform(t), label="Uniform")
    plt.plot(t, kernel_epanechnikov(t), label="Epanechnikov")
    plt.title("Kernels")
    plt.legend()
    plt.show()


def _experiment_02() -> None:
    """Kernel Density estimation experiment 02."""
    import matplotlib.pyplot as plt

    t = np.linspace(-3, 3, 100)
    dim_1, dim_2 = np.meshgrid(t, t)

    z_gauss = np.zeros((t.size, t.size))
    z_uniform = np.zeros((t.size, t.size))

    for i in np.arange(t.size):
        aux_1 = dim_1[:, i].reshape(-1, 1)
        aux_2 = dim_2[:, i].reshape(-1, 1)
        vals = np.hstack((aux_1, aux_2))
        z_gauss[:, i] = kernel_gaussian(vals)
        z_uniform[:, i] = kernel_uniform(vals)

    plt.title("Kernels")
    plt.subplot(121)
    plt.contour(dim_1, dim_2, z_gauss)
    plt.subplot(122)
    plt.contour(dim_1, dim_2, z_uniform)
    plt.show()


def _experiment_03() -> None:
    """Estimate Kernel bandwidth."""
    import matplotlib.pyplot as plt
    from cross_validation import loo_cv

    def loss_function(x):
        """Mean of log-probabilities (0 <= x <= 1)."""
        return -np.mean(np.log(x))

    random_state = 16

    np.random.seed(random_state)

    candidate_bandwidths = np.linspace(0.01, 1.0, 25)
    known_points = np.random.normal(size=512)

    logls = np.zeros(candidate_bandwidths.size)

    for X_test, X_train in loo_cv(known_points):
        for ind_bandwidth in np.arange(candidate_bandwidths.size):
            kernel_est = kernel_density_est(
                unknown_points=X_test,
                known_points=X_train,
                kernel=kernel_gaussian,
                kernel_bandwidth=candidate_bandwidths[ind_bandwidth])

            logls[ind_bandwidth] += loss_function(kernel_est)

    bandwidth_opt = candidate_bandwidths[logls.argmin()]
    print("Optimal bandwidth: {} (loss: {})".format(bandwidth_opt,
                                                    logls.min()))

    t = np.linspace(-3, 3, 100)

    plt.hist(known_points, 32, density=True)
    plt.plot(
        t,
        kernel_density_est(
            unknown_points=t,
            known_points=known_points,
            kernel=kernel_gaussian,
            kernel_bandwidth=bandwidth_opt))
    plt.show()


if __name__ == "__main__":
    # _experiment_01()
    # _experiment_02()
    _experiment_03()

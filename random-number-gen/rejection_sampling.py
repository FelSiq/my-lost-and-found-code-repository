import typing as t

import numpy as np


def horners(x: t.Union[int, float], coeffs: np.ndarray) -> t.Union[int, float]:
    """Evaluates a polynomial in ``x``.

    Operates with time complexity Theta(n) and space
    complexity Theta(1).
    """
    val = 0.0 if isinstance(x, float) else 0

    for c in coeffs:
        val = c + x * val

    return val


def normal_pdf(x: t.Union[int, float], mean: float = 0.0, var=1.0) -> float:
    """Probability density of normal distribution."""
    return 1.0 / (2.0 * np.pi * var)**0.5 * np.exp(-(x - mean)**2.0 /
                                                   (2.0 * var))


def rejection_sampling(func_a: t.Callable[[t.Union[int, float]], float],
                       func_b: t.Callable[[t.Union[int, float]], float],
                       func_b_sampler: t.Callable[[], float],
                       func_b_coeff: t.Union[float, int] = 1.0,
                       samples_num: int = 1000,
                       random_seed: t.Optional[int] = None) -> np.ndarray:
    """Uses rejection sampling method to sample from a complicated function ``func_a``."""
    if random_seed is not None:
        np.random.seed(random_seed)

    vals = []  # type: t.List[float]

    while len(vals) < samples_num:
        random_point = func_b_sampler()

        new_val = np.random.random() * func_b_coeff * func_b(random_point)

        if new_val < func_a(random_point):
            vals.append(random_point)

    return np.array(vals)


def _experiment_01():
    """Rejection sampling experiment 01."""

    def complex_fun(x: t.Union[int, float, np.ndarray]) -> t.Union[np.ndarray, float]:
        if 2 <= x < 3:
            return 1.75 - 0.5 * x

        if 1 <= x < 2 or 3 <= x < 3.55:
            return 7 + 0.5 * x

        if -2 <= x < 0.5:
            return horners(x, [0.5, -0.25, 1])

        return 0

    N = 20000
    INTERVAL = (-10, 10)
    B_COEFF = 90
    SEED = 1234

    p = np.array([5, 0, 3, 5, 1])
    vals = np.linspace(*INTERVAL, num=N)

    res1 = np.array([complex_fun(x) for x in vals])

    mean = 1.2
    var = 5

    res2 = np.array([normal_pdf(x, mean=mean, var=var) for x in vals])

    norm_fact = max(res1.max(), B_COEFF * res2.max())

    plt.plot(vals, res1 / norm_fact)
    plt.plot(vals, B_COEFF * res2 / norm_fact)

    aux = rejection_sampling(
        func_a=complex_fun,
        func_b=lambda x: normal_pdf(x, mean=mean, var=var),
        func_b_sampler=lambda: mean + np.sqrt(var) * np.random.randn(),
        func_b_coeff=B_COEFF,
        samples_num=N,
        random_seed=SEED)

    plt.hist(aux, bins=np.ptp(INTERVAL), density=True)
    plt.show()


def _experiment_02() -> None:
    """Rejection sampling experiment 02."""

    def complex_fun(x: t.Union[int, float, np.ndarray]) -> t.Union[float, np.ndarray]:
        return (0.30 * normal_pdf(x, mean=-2.0, var=1.2) + 0.60 * normal_pdf(
            x, mean=2.0, var=0.8) - 0.10 * normal_pdf(x, mean=-2.0, var=0.35))

    INTERVAL = (-10, 10)
    N = 20000
    B_COEFF = 2.5
    SEED = 1444

    mean = 0
    var = 5

    vals = np.linspace(*INTERVAL, num=N)

    res1 = complex_fun(vals)  # type: np.ndarray
    res2 = B_COEFF * normal_pdf(vals, mean=mean, var=var)  # type: np.ndarray

    # The following plot must be always below the horizontal line y = 1.0.
    plt.plot(vals, res1 / res2)
    plt.hlines(1.0, *INTERVAL)
    plt.show()

    aux = rejection_sampling(
        func_a=complex_fun,
        func_b=lambda x: normal_pdf(x, mean=mean, var=var),
        func_b_sampler=lambda: mean + np.sqrt(var) * np.random.randn(),
        func_b_coeff=B_COEFF,
        samples_num=N,
        random_seed=SEED)

    plt.plot(vals, res1)
    plt.plot(vals, res2)
    plt.hist(aux, bins=np.ptp(INTERVAL), density=True)
    plt.show()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # _experiment_01()
    _experiment_02()

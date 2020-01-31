import typing as t

import matplotlib.pyplot as plt
import numpy as np


def logistic_map(survival_factor: int, num_it: int = 192,
                 burn_in_it: int = 92) -> t.Sequence[float]:
    """Returns ``survival_factor`` convergence vals. for the logistic map."""
    vals = set()

    z = 0.5
    for _ in np.arange(burn_in_it):
        z = survival_factor * z * (1 - z)

    for _ in np.arange(num_it - burn_in_it):
        z = survival_factor * z * (1 - z)
        vals.add(np.round(z, 8))

    return list(vals)


def plot_logistic_map(survival_factors: np.ndarray) -> None:
    """Plot the logistic map."""
    for it, sf in enumerate(survival_factors):
        vals = logistic_map(sf)

        plt.scatter(len(vals) * [sf], vals, color="black", s=0.05)

        if it % 100 == 0:
            print("{:.2f}%".format(100 * it / survival_factors.size))

    plt.title("Logistic map")
    plt.show()


def _test() -> None:
    plot_logistic_map(np.linspace(0, 4, 512))


if __name__ == "__main__":
    _test()

"""Implementation of Symmetric Metropolis-Hasting algorithm."""
import typing as t

import numpy as np


def symm_metropolis_hasting(initial_theta: float,
                            num_samples: int,
                            log_target: t.Callable[[float], float],
                            proposal_sampler: t.Callable[[float], float],
                            discard_warm_up: bool = True,
                            warm_up_frac: float = 0.5,
                            verbose: bool = False) -> np.ndarray:
    """Symmetric case of Metropolis-Hasting algorithm."""
    if num_samples <= 0:
        raise ValueError("'num_samples' must be a positive value.")

    if discard_warm_up and not 0 <= warm_up_frac < 1:
        raise ValueError("'warm_up_frac' must be in [0.0, 1.0) range.")

    theta = initial_theta
    theta_log_targ = log_target(theta)

    thetas = np.zeros(num_samples)

    hits = 0
    for ind in np.arange(num_samples):
        theta_proposed = proposal_sampler(theta)

        if np.log(np.random.uniform(
                0, 1)) < log_target(theta_proposed) - theta_log_targ:
            theta = theta_proposed
            theta_log_targ = log_target(theta)
            hits += 1

        thetas[ind] = theta

    if verbose:
        acceptance_rate = hits / num_samples
        print("Acceptance rate: {}".format(acceptance_rate))
        print("Theoretically expected: [0.23, 0.50] (results is {}.)".format(
            "optimal" if 0.23 <= acceptance_rate <= 0.50 else "not optimal"))

    if discard_warm_up:
        return thetas[int(warm_up_frac * thetas.size):]

    return thetas


def _experiment() -> None:
    """Experiment 01."""
    import matplotlib.pyplot as plt
    import scipy.stats

    random_seed = 16
    np.random.seed(random_seed)

    laplace_dist = scipy.stats.laplace(loc=0.0, scale=1.0 / (2**0.5))
    test_vals = np.linspace(-5, 5, 10000)

    for plot_id, scale in enumerate([0.1, 2.5, 10, 50]):
        thetas = symm_metropolis_hasting(
            initial_theta=0.0,
            num_samples=20000,
            log_target=lambda x: -np.abs(x),
            proposal_sampler=
            lambda theta: theta + np.random.normal(loc=0.0, scale=scale),
            discard_warm_up=True,
            warm_up_frac=0.5,
            verbose=True)

        plt.subplot(4, 2, plot_id * 2 + 1)
        plt.plot(thetas[::10], label=str(scale))
        plt.legend()

        plt.subplot(4, 2, plot_id * 2 + 2)
        plt.plot(test_vals, laplace_dist.pdf(test_vals))
        plt.hist(thetas, bins=128, density=True, label=str(scale))
        plt.legend()

    plt.show()


if __name__ == "__main__":
    _experiment()

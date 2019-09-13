"""Testing bootstrap."""
import typing as t

import numpy as np


def bootstrap(population: np.ndarray,
              n: int = 10,
              prop: float = 1.0,
              random_state: t.Optional[int] = None) -> np.ndarray:
    """Generator of bootstraps ``population``.

    Arguments
    ---------
    prop : :obj:`float`
        Proportion between the size of ``population`` and the sampled
        population. Must be in (0.0, 1.0] interval.

    n : :obj:`int`
        Number of pseudo-datasets generated.

    random_state : :obj:`int`, optional
        If given, set the random seed before the first iteration.

    Returns
    -------
    :obj:`np.ndarray`
        Sample of ``population`` constructed via bootstrap technique.
    """
    if not isinstance(prop, (np.number, float, int)):
        raise TypeError("'prop' must be numberic type (got {}).".format(
            type(prop)))

    if not 0.0 < prop <= 1.0:
        raise ValueError("'prop' must be a number in (0.0, 1.0] interval.")

    if random_state is not None:
        np.random.seed(random_state)

    bootstrap_size = int(population.size * prop)

    for _ in np.arange(n):
        cur_inds = np.random.randint(population.size, size=bootstrap_size)
        yield population[cur_inds]


def experiment() -> None:
    """Bootstrap experiment.

    To calculate the two-sided confidence interval (1.0 - alpha), for some
    parameter theta, can be obtained calculating the internal

        [h_{alpha/2}, h_{1-alpha/2)}]

    Where h_{x} denotes the x quantile of the bootstrap estimates for the
    parameter theta.

    To be more clear, we can calculate the some statistic theta for every
    bootstrapped pseudo-dataset, and then collect the two percentiles
    p_1 = h_{alpha/2} and p_2 = h_{1-alpha/2)} to form the two sided
    (1.0 - alpha) confidence internal [p_1, p_2].

    This experiment exemplifies this building the two-sided confidence
    interval for the mean of some dataset.
    """
    random_state = 16
    reps = 1000
    alpha = 0.05

    np.random.seed(random_state)

    pop = np.random.normal(size=100)

    print("Population mean: {}".format(pop.mean()))

    bootstrapper = bootstrap(pop, n=reps, random_state=random_state)

    means = np.zeros(reps)
    for i, pseudo_pop in enumerate(bootstrapper):
        means[i] = pseudo_pop.mean()

    percentiles = 100 * np.array([0.5 * alpha, 1.0 - 0.5 * alpha])
    it_min, it_max = np.percentile(means, percentiles)

    print("Confidence interval (alpha = {}): [{}, {}]".format(alpha, it_min, it_max))
    print("True mean in confidence internal: {}".format(it_min <= pop.mean() <= it_max))


if __name__ == "__main__":
    experiment()

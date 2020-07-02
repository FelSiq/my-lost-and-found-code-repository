"""Convex combination between two points."""
import numpy as np


def conv_comb(p1: np.ndarray, p2: np.ndarray, alpha: float) -> np.ndarray:
    """Convex combination between two points.

    Convex combination p3 between p1 and p2 is given by:
    p3 = alpha * p1 + (1 - alpha) * p2, for any alpha in [0, 1].
    """
    alpha = float(alpha)

    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"'alpha' must be in [0, 1] (got {alpha:.4f}).")

    return alpha * p1 + (1.0 - alpha) * p2


def _test() -> None:
    import matplotlib.pyplot as plt

    np.random.seed(16)

    p1, p2 = np.random.randn(4).reshape(2, 2)

    cb = np.array([conv_comb(p1, p2, alpha) for alpha in np.linspace(0, 1, 8)])

    fig, ax = plt.subplots(1)

    ax.set_title("Convex combination")
    ax.scatter([p1[0], p2[0]], [p1[1], p2[1]], color="black", s=128)
    ax.plot(*np.vstack((p1, p2)).T, linestyle="dotted")
    ax.scatter(cb[:, 0], cb[:, 1], color="red", s=16)

    plt.show()


if __name__ == "__main__":
    _test()

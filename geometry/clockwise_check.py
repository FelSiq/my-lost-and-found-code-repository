"""Check whether p1 is clockwise from p2 in respect to the origin."""
import numpy as np


def _det(p1: np.ndarray, p2: np.ndarray) -> bool:
    """Determinant of pair of points."""
    x1, y1 = p1
    x2, y2 = p2
    return x1 * y2 - y1 * x2


def is_clockwise(p1: np.ndarray, p2: np.ndarray) -> bool:
    """Check if p1 is clockwise from p2 in respect to the origin.

    The pair of points must be two-dimensional.
    """
    return _det(p1, p2) >= 0.0


def is_clockwise_seg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> bool:
    """Check if the segment p0->p1 is clockwise in respect to segment p0->p2."""
    return is_clockwise(p1 - p0, p2 - p0)


def is_coclockwise(p1: np.ndarray, p2: np.ndarray) -> bool:
    """Check if p1 is counter-clockwise from p2 in respect to the origin.

    The pair of points must be two-dimensional.
    """
    # Note: this is not the same as 'not is_clockwise(p1, p2)', since in the
    # boundary condition where the vectors are colinear p1 and p2 are
    # simultaneously clockwise and counter-clockwise to each other in respect
    # to the origin.
    return _det(p1, p2) <= 0.0


def is_coclockwise_seg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> bool:
    """Check if the segment p0->p1 is counter-clockwise in respect to p0->p2."""
    return is_coclockwise(p1 - p0, p2 - p0)


def is_colinear(p1: np.ndarray, p2: np.ndarray) -> bool:
    """Check if p1 and p2 are colinear.

    The pair of points must be two-dimensional.
    """
    return np.isclose(0.0, _det(p1, p2))


def is_colinear_seg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> bool:
    """Check if p1 and p2 are colinear.

    The pair of points must be two-dimensional.
    """
    return np.isclose(0.0, _det(p1 - p0, p2 - p0))


def is_left_turn(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> bool:
    """Check if there is a left turn in the p0->p1->p2 transition."""
    return is_coclockwise_seg(p0, p1, p2) and not is_colinear(p2 - p0, p1 - p0)


def is_right_turn(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> bool:
    """Check if there is a right turn in the p0->p1->p2 transition."""
    return is_clockwise_seg(p0, p1, p2) and not is_colinear(p2 - p0, p1 - p0)


def _test() -> None:
    def plot(p1, p2, ax, title):
        ax.scatter(*np.vstack((p1, p2)).T)
        ax.set_title(title)
        ax.annotate("p1", p1)
        ax.annotate("p2", p2)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ax.hlines(0, min(0, xmin), max(0, xmax))
        ax.vlines(0, min(0, ymin), max(0, ymax))

        ax.plot([0, p2[0]], [0, p2[1]], linestyle="dotted")
        ax.plot([0, p1[0]], [0, p1[1]], linestyle="dotted")

    def plot_seg(p1, p2, p3, ax, title):
        ax.scatter(*np.vstack((p1, p2, p3)).T)
        ax.set_title(title)
        ax.annotate("p1", p1)
        ax.annotate("p2", p2)
        ax.annotate("p3", p3)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        ax.hlines(0, min(0, xmin), max(0, xmax))
        ax.vlines(0, min(0, ymin), max(0, ymax))

        ax.plot([p1[0], p3[0]], [p1[1], p3[1]], linestyle="dotted")
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], linestyle="dotted")

    import matplotlib.pyplot as plt

    np.random.seed(32)

    p1, p2, p3 = np.random.randn(6).reshape(3, 2)
    p4 = p3 * 1.5

    fig, (ax1, ax2, ax3) = plt.subplots(3)

    plot(p1, p2, ax1, f"Is clockwise? {'yes!' if is_clockwise(p1, p2) else 'no.'}")
    plot(p3, p4, ax2, f"Is colinear? {'yes!' if is_colinear(p3, p4) else 'no.'}")
    plot_seg(p1, p2, p3, ax3, f"Is closer clockwise? {'yes!' if is_clockwise_seg(p1, p2, p3) else 'no.'}")

    assert is_clockwise_seg(p1, p2, p3) == is_right_turn(p1, p2, p3)
    assert is_coclockwise_seg(p1, p2, p3) == is_left_turn(p1, p2, p3)
    assert (not is_left_turn(np.zeros(2), p3, p4) and
            not is_right_turn(np.zeros(2), p3, p4) and
            is_colinear_seg(np.zeros(2), p3, p4))

    plt.show()


if __name__ == "__main__":
    _test()

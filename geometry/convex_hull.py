"""TODO."""
import functools

import numpy as np

import clockwise_check
import _utils


def _get_first_point(points: np.ndarray) -> int:
    """Get the point with minimal 'y' and the leftmost 'x' to break ties."""
    x, y = points.T
    min_ind = np.argmin(y)
    tied_pos = np.flatnonzero(np.isclose(y[min_ind], y))

    if tied_pos.size == 1:
        return min_ind

    # Note: break ties using the leftmost 'x'
    min_ind = tied_pos[np.argmin(x[tied_pos])]

    return min_ind


def _sort_by_orientation(points: np.ndarray, first_ind: int) -> int:
    """Sort points based on the polar angle relative to the first index.

    The points are sorted by counter-clockwise rotation starting from the
    point in the chosen first index.
    """
    p1 = points[first_ind, :]
    points = np.delete(points, first_ind, axis=0) - p1

    rem_inds = set()

    def _cmp(ind1: int, ind2: int) -> float:
        pb, pa = points[ind2, :], points[ind1, :]
        polar_angle = _utils.det(pb, pa)

        # Note: if polar_angle == 0, then both points have the same polar angle
        # in respect to a fixed origin. The correct procedure, in this case, is
        # to keep just the farthest point.
        if np.isclose(0.0, polar_angle):
            rem_inds.add(ind1
                         if np.linalg.norm(pa) <= np.linalg.norm(pb)
                         else ind2)

        return polar_angle

    ordered_inds = sorted(np.arange(points.shape[0]),
                          key=functools.cmp_to_key(_cmp))

    # Note: removing all points that has the same polar angle in respect
    # to the p1.
    ordered_inds = np.asarray([i for i in ordered_inds if i not in rem_inds])

    return np.vstack((p1, points[ordered_inds, :] + p1))


def graham_scan(points: np.ndarray) -> np.ndarray:
    """TODO."""
    if points.shape[0] <= 2:
        raise ValueError("Need 3 or more points.")

    first_ind = _get_first_point(points)
    points = _sort_by_orientation(points, first_ind)

    if points.shape[0] <= 2:
        raise ValueError("Need 3 or more points non-colinear points.")

    stack = list(points[:3, :])

    for i in np.arange(3, points.shape[0]):
        (p1, p2), p3 = stack[-2:], points[i, :]

        while not clockwise_check.is_left_turn(p1, p2, p3):
            stack.pop()
            p1, p2 = stack[-2:]

        stack.append(p3)

    return np.asarray(stack, dtype=float)


def _test(arr: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    first_ind = _get_first_point(arr)

    col = np.repeat("b", arr.shape[0])
    col[first_ind] = "r"

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.scatter(*arr.T, c=col)
    ax1.set_title("Chosen starting point")

    sorted_arr = _sort_by_orientation(arr, first_ind)

    ax2.scatter(*sorted_arr.T, label="points")

    conv_hull = graham_scan(arr)
    ax2.scatter(*conv_hull.T, c="g", label="CH vertices", s=128)
    ax2.plot(*np.vstack((conv_hull, conv_hull[0, :])).T, color="orange", label="CH")
    ax2.legend()

    for i, coord in enumerate(sorted_arr):
        ax2.annotate(i, coord, size=16)
    
    plt.show()


def _test_01() -> None:
    arr = np.vstack((np.arange(10), np.ones(10) + (np.arange(1, 1 + 10) % 2))).T
    _test(arr)


def _test_02() -> None:
    np.random.seed(16)
    arr = np.random.randn(30).reshape(-1, 2)
    _test(arr)


if __name__ == "__main__":
    _test_02()
    _test_01()

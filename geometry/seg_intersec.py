"""Check whether a pair of line segments insertect."""
import typing as t

import numpy as np

import _utils


def direction(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate the relative direction of p1->p2 and p1->p3.

    If < 0, then p1->p2 is to the left (counter-clockwise rotation) of p1->p3.
    If > 0, then p1->p2 is to the right (clockwise rotation) of p1->p3.
    If = 0, then p1->p2 and p1->p3 are colinear.
    """
    return _utils.det(p2 - p1, p3 - p1)


def on_segment(p1: np.ndarray,
               p2: np.ndarray,
               p3: np.ndarray,
               dir_: t.Optional[float] = None) -> bool:
    """Check if p3 is in the segment p1->p2."""
    if dir_ is None:
        dir_ = direction(p1, p2, p3)

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    x_min, x_max = (x1, x2) if x1 <= x2 else (x2, x1)
    y_min, y_may = (y1, y2) if y1 <= y2 else (y2, y1)

    # Note: checks if all the three points are colinear
    is_colinear = np.isclose(dir_, 0.0)

    return is_colinear and x_min <= x3 <= x_max and y_min <= y3 <= y_max


def do_intersec(p1: np.ndarray, p2: np.ndarray,
                p3: np.ndarray, p4: np.ndarray) -> bool:
    """Check whether the segments p1->p2 and p3->p4 do intersect."""
    dir1 = direction(p3, p4, p1)
    dir2 = direction(p3, p4, p2)
    dir3 = direction(p1, p2, p3)
    dir4 = direction(p1, p2, p4)

    # Note: a segment straddlers a line (or, in this case, another segment)
    # if each of its endpoints lies on distinct sides of the line (or the
    # other line segment).
    p1p2straddlers = dir1 * dir2 < 0
    p3p4straddlers = dir3 * dir4 < 0

    # Note: check if both segments straddlers
    if p1p2straddlers and p3p4straddlers:
        return True

    # Note: handling boundary cases where exists colinearity between
    # some point in distinct segments
    if (on_segment(p3, p4, p1, dir_=dir1) or
            on_segment(p3, p4, p2, dir_=dir2) or
            on_segment(p1, p2, p3, dir_=dir3) or
            on_segment(p1, p2, p4, dir_=dir4)):
        return True

    return False


def _test() -> None:
    p1, p2, p3, p4 = np.array([
        [0, 0],
        [1, 1],
        [1, 0],
        [0, 1],
    ])

    print("p1:", p1)
    print("p2:", p2)
    print("p3:", p3)
    print("p4:", p4)

    res1 =  do_intersec(p1, p2, p3, p4)
    res2 =  do_intersec(p1, p3, p2, p4)

    print("Intersection? p1->p2 and p3->p4:", res1)
    print("Intersection? p1->p3 and p2->p4:", res2)

    assert res1 is True
    assert res2 is False


if __name__ == "__main__":
    _test()

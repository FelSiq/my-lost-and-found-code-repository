"""Find a intersection point of a pair of line segments.

Does not handle vertical lines.
"""
import typing as t

import numpy as np


def det(p1: np.ndarray, p2: np.ndarray) -> bool:
    """Determinant of pair of points."""
    x1, y1 = p1
    x2, y2 = p2
    return x1 * y2 - y1 * x2


def direction(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Calculate the relative direction of p1->p2 and p1->p3.
    If < 0, then p1->p2 is to the left (counter-clockwise rotation) of p1->p3.
    If > 0, then p1->p2 is to the right (clockwise rotation) of p1->p3.
    If = 0, then p1->p2 and p1->p3 are colinear.
    """
    return det(p2 - p1, p3 - p1)


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
    y_min, y_max = (y1, y2) if y1 <= y2 else (y2, y1)

    # Note: checks if all the three points are colinear
    is_colinear = np.isclose(dir_, 0.0)

    return is_colinear and x_min <= x3 <= x_max and y_min <= y3 <= y_max


def intersec_where(pa1, pa2, pb1, pb2):
    ma = (pa2[1] - pa1[1]) / (pa2[0] - pa1[0])
    mb = (pb2[1] - pb1[1]) / (pb2[0] - pb1[0])

    if np.isclose(ma, mb):
        for pa in [pa1, pa2]:
            if on_segment(pa, pb1, pb2):
                return pa

        for pb in [pb1, pb2]:
            if on_segment(pb, pa1, pa2):
                return pb

    xa, ya = pa1
    xb, yb = pb1

    x_intersec = (yb - ya + ma * xa - mb * xb) / (ma - mb)
    y_intersec = ma * (x_intersec - xa) + ya

    p_intersec = (x_intersec, y_intersec)

    return p_intersec


def do_intersec(p1: np.ndarray, p2: np.ndarray,
                p3: np.ndarray, p4: np.ndarray) -> bool:
    """Check whether the segments p1->p2 and p3->p4 do intersect."""
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p3 = np.asarray(p3)
    p4 = np.asarray(p4)

    if np.allclose(p3, p4):
        if on_segment(p3, p1, p2):
            return p3

    if np.allclose(p1, p2):
        if on_segment(p1, p3, p4):
            return p1

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
        return intersec_where(p1, p2, p3, p4)

    # Note: handling boundary cases where exists colinearity between
    # some point in distinct segments
    if (on_segment(p3, p4, p1, dir_=dir1) or
            on_segment(p3, p4, p2, dir_=dir2) or
            on_segment(p1, p2, p3, dir_=dir3) or
            on_segment(p1, p2, p4, dir_=dir4)):
        return intersec_where(p1, p2, p3, p4)

    return None


if __name__ == "__main__":
    # aux1 = map(float, input().strip().split(" "))
    # aux2 = map(float, input().strip().split(" "))

    import random
    import matplotlib.pyplot as plt

    minv, maxv = -3, 3

    aux1 = [random.randint(minv, maxv), random.randint(minv, maxv), random.randint(minv, maxv), random.randint(minv, maxv)]
    aux2 = [random.randint(minv, maxv), random.randint(minv, maxv), random.randint(minv, maxv), random.randint(minv, maxv)]

    pa1, pa2 = aux1[:2], aux1[2:]
    pb1, pb2 = aux2[:2], aux2[2:]

    res = do_intersec(pa1, pa2, pb1, pb2)

    print(res, res)

    plt.plot([pa1[0], pa2[0]], [pa1[1], pa2[1]], color="black")
    plt.plot([pb1[0], pb2[0]], [pb1[1], pb2[1]], color="blue")

    if res is not None:
        plt.scatter([res[0]], [res[1]], color="red")
        plt.title("intersection")

    else:
        plt.title("No intersection")

    plt.show()

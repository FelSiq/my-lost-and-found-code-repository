"""Utility code."""
import numpy as np


def det(p1: np.ndarray, p2: np.ndarray) -> bool:
    """Determinant of pair of points."""
    x1, y1 = p1
    x2, y2 = p2
    return x1 * y2 - y1 * x2

"""."""
import typing as t

import numpy as np


class InfiniteArray:
    """Array indexable on all integers."""

    def __init__(self, values: t.Sequence[int], start_index: int):
        """Infinite array defined in some range starting at ``start_index``.

        Array's value outside the defined range is infinity (:obj:`np.inf`).
        """
        self.lim_inf = start_index
        self.lim_sup = start_index + len(values)
        self.values = values

    def __getitem__(self, ind: int) -> t.Union[int, float]:
        """Returns array's value at index ``ind``."""
        if not self.lim_inf <= ind < self.lim_sup:
            return np.inf

        return self.values[ind - self.lim_inf]

    def __setitem__(self, ind: int, value: t.Union[int, float]) -> None:
        if np.isinf(value):
            raise ValueError(
                "Infinity value is forbidden in array's defined range.")

        if not self.lim_inf <= ind < self.lim_sup:
            raise IndexError(
                "Array not defined in index {} (range is [{}, {})).".format(
                    ind, self.lim_inf, self.lim_sup))

        self.values[ind - self.lim_inf] = value

    def __str__(self):
        return "{} defined in [{}, {})".format(self.values.__str__(),
                                               self.lim_inf, self.lim_sup)


def _find_lim(array: InfiniteArray, ind: int) -> int:
    while np.isfinite(array[ind]):
        ind *= 2

    return ind


def bin_search(array: InfiniteArray, a: int, b: int) -> int:
    neg_side = b <= 0

    while a < b:
        middle = (a + b) // 2

        if (neg_side and np.isfinite(
                array[middle])) or (not neg_side and np.isinf(array[middle])):
            b = middle - 1

        else:
            a = middle + 1

    if neg_side:
        return a if np.isfinite(array[a]) else a + 1

    return a if np.isfinite(array[a]) else a - 1


def ext_bin_search(array: InfiniteArray, start_ind: int) -> t.Tuple[int, int]:
    """Find defined range of InfiniteArray in O(log_{n}) time complexity."""
    a = bin_search(array, _find_lim(array, -1), 0)
    b = bin_search(array, 0, _find_lim(array, +1))
    return a, b


if __name__ == "__main__":
    for _ in np.arange(5000):
        start_index = np.random.randint(-987, 0)
        values = np.arange(abs(start_index) + np.random.randint(378))
        array = InfiniteArray(values, start_index)
        range_ = ext_bin_search(array, 0)
        exp = (start_index, start_index + len(values) - 1)
        error = range_ != exp
        if error:
            print(range_, exp)
        assert not error

"""Segment Tree with Lazy propagation."""
import typing as t

import numpy as np


class SegmentTree:
    def __init__(self, values: t.Sequence[float]):
        self.size = len(values)

        size = self._next_power_of_two(4 * len(values))

        self.tree = np.full(size, fill_value=np.inf)
        self.lazy = np.zeros(size)

        self._values = values
        self.build_tree(0, len(values) - 1, 0)

    @classmethod
    def _next_power_of_two(cls, size: int) -> int:
        return 2 ** int(np.ceil(np.log2(size)))

    def build_tree(self, low: int, high: int, pos: int) -> None:
        if low == high:
            self.tree[pos] = self._values[low]
            return

        middle = low + (high - low) // 2

        self.build_tree(low, middle, 2 * pos + 1)
        self.build_tree(middle + 1, high, 2 * pos + 2)

        self.tree[pos] = min(self.tree[2 * pos + 1], self.tree[2 * pos + 2])

    def _search(
        self, qlow: int, qhigh: int, low: int, high: int, pos: int, inc: float
    ) -> float:
        if low > high:
            return np.inf

        if not np.isclose(self.lazy[pos], 0.0):
            self.tree[pos] += self.lazy[pos]
            if low != high:
                self.lazy[2 * pos + 1] += self.lazy[pos]
                self.lazy[2 * pos + 2] += self.lazy[pos]
            self.lazy[pos] = 0

        if qlow > high or qhigh < low:
            return np.inf

        if qlow <= low and high <= qhigh:
            if not np.isclose(inc, 0.0):
                self.tree[pos] += inc
                if low != high:
                    self.lazy[2 * pos + 1] += inc
                    self.lazy[2 * pos + 2] += inc

            return self.tree[pos]

        middle = low + (high - low) // 2

        ret = min(
            self._search(qlow, qhigh, low, middle, 2 * pos + 1, inc),
            self._search(qlow, qhigh, middle + 1, high, 2 * pos + 2, inc),
        )

        return ret

    def search(self, low: int, high: int) -> float:
        return self._search(low, high, 0, self.size - 1, 0, 0.0)

    def update(self, low: int, high: int, inc: float) -> None:
        self._search(low, high, 0, self.size - 1, 0, inc)


def _test():
    num_trees = 1000
    num_tests = 500
    max_size = 1000
    min_v = -100
    max_v = 100

    np.random.seed(16)

    for i in np.arange(num_trees):
        print(f"{i + 1} / {num_trees}...")
        size = np.random.randint(1, max_size)

        vals = (max_v - min_v) * np.random.random(size) + min_v
        tree = SegmentTree(vals)

        for j in np.arange(num_tests // 2):
            print(f"   test {j + 1} / {num_tests}...")
            aux = np.random.randint(0, vals.size, size=2)
            qlow = min(aux)
            qmax = max(aux)
            assert np.isclose(tree.search(qlow, qmax), np.min(vals[qlow : qmax + 1]))

        for j in np.arange(num_tests // 2, num_tests):
            print(f"   test {j + 1} / {num_tests}...")
            aux = np.random.randint(0, vals.size, size=2)
            qlow = min(aux)
            qmax = max(aux)
            delta = 200 * np.random.random() - 100
            tree.update(qlow, qmax, delta)
            assert np.isclose(
                tree.search(qlow, qmax), delta + np.min(vals[qlow : qmax + 1])
            )
            tree.update(qlow, qmax, -delta)


if __name__ == "__main__":
    _test()

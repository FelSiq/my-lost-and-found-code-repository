"""Builds a Binary Search Tree from a previously sorted array."""


class Node:
    def __init__(self, value, son_r=None, son_l=None):
        self.value = value
        self.son_r = son_r
        self.son_l = son_l


class SearchBinTree:
    def __init__(self, head=None):
        self.head = head

    def _search(self, value, cur_node):
        if not cur_node:
            return False

        if cur_node.value == value:
            return True

        if value > cur_node.value:
            return self._search(value, cur_node.son_l)

        return self._search(value, cur_node.son_r)

    def search(self, value):
        return self._search(value, self.head)


def _build_node(array, start, end):
    if start > end:
        return None

    middle = (start + end) // 2

    return Node(value=array[middle],
                son_r=_build_node(array, start, middle-1),
                son_l=_build_node(array, middle+1, end))


def build_sbt(array):
    return SearchBinTree(_build_node(array, 0, len(array)-1))


if __name__ == "__main__":
    import random

    for i in range(15000):
        t_1 = sorted([
            random.randint(-9999, 9999)
            for _ in range(random.randint(1, 155))
        ])

        bin_t = build_sbt(t_1)

        for v in t_1:
            assert bin_t.search(v)

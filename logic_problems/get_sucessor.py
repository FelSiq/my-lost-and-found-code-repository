"""Get in-order sucessor node of a given value/node of a Binary Search Tree."""


class Node:
    def __init__(self, value, parent=None):
        self.value = value
        self.son_l = None
        self.son_r = None
        self.parent = parent


class BST:
    def __init__(self):
        self.root = None
        self.in_order = []

    def _insert(self, node, value):
        if node.value == value:
            raise ValueError('No duplicated elements allowed')

        if node.value > value:
            if not node.son_l:
                node.son_l = Node(value, parent=node)
                return

            self._insert(node.son_l, value)
        else:

            if not node.son_r:
                node.son_r = Node(value, parent=node)
                return

            self._insert(node.son_r, value)

    def insert(self, value):
        if not self.root:
            self.root = Node(value)

        else:
            self._insert(self.root, value)

    def _travel(self, node):
        if not node:
            return

        self._travel(node.son_l)
        self.in_order.append(node.value)
        self._travel(node.son_r)

    def traversal_in_order(self):
        self.in_order = []
        self._travel(self.root)
        return self.in_order

    def _get_leftmost(self, node):
        if not node.son_l:
            return node

        return self._get_leftmost(node.son_l)

    def _get_parent(self, node):
        if node.parent is None:
            return None

        if node.parent.son_r == node:
            return self._get_parent(node.parent)

        return node.parent

    def travel_tree(self, node, value):
        if not node:
            return None

        if node.value == value:
            if node.son_r:
                return self._get_leftmost(node.son_r)

            if node.parent is None:
                return None

            if node.parent.son_l == node:
                return node.parent

            return self._get_parent(node)

        elif node.value > value:
            return self.travel_tree(node.son_l, value)

        return self.travel_tree(node.son_r, value)

    def get_sucessor(self, value):
        return self.travel_tree(tree.root, value)


if __name__ == '__main__':
    import numpy as np

    for i in range(10000):
        tree = BST()

        vals = np.random.randint(-9999, 9999, size=30 + i // 5)

        for val in vals:
            try:
                tree.insert(val)
            except ValueError:
                pass

        in_order_res = tree.traversal_in_order()

        for val in vals:
            res = tree.get_sucessor(val)

            try:
                expected = in_order_res[1 + in_order_res.index(val)]
                assert res.value == expected

            except IndexError:
                assert res is None

"""Given a Binary Search Tree, construct all array of values
that could have generated it.
"""


class Node:
    def __init__(self, value):
        self.value = value
        self.son_l = None
        self.son_r = None


class BST:
    def __init__(self):
        self.root = None

    def _insert(self, node, value):
        if node.value == value:
            raise ValueError('No duplicated elements allowed')

        if node.value > value:
            if not node.son_l:
                node.son_l = Node(value)
                return

            self._insert(node.son_l, value)
        else:

            if not node.son_r:
                node.son_r = Node(value)
                return

            self._insert(node.son_r, value)

    def insert(self, value):
        if not self.root:
            self.root = Node(value)

        else:
            self._insert(self.root, value)

    def _in_order_travel(self, node, ans):
        if not node:
            return
        self._in_order_travel(node.son_l, ans)
        ans.append(node.value)
        self._in_order_travel(node.son_r, ans)

    def in_order_travel(self):
        ans = []
        self._in_order_travel(self.root, ans)
        return ans

    def _build_possibilities(self, A, B, root):
        possibilities = []
        for p_a in A:
            for i in range(len(p_a)):
                for p_b in B:
                    aux = [root] + p_a[:(i + 1)] + p_b + p_a[(i + 1):]
                    possibilities.append(aux)

        return possibilities

    def _get_possibilities(self, node):
        if not node:
            return [[]]

        lp = self._get_possibilities(node.son_l)
        lr = self._get_possibilities(node.son_r)

        ans = (self._build_possibilities(lp, lr, node.value) +
               self._build_possibilities(lr, lp, node.value))

        if not ans:
            return [[node.value]]

        return ans

    def build_bst_arrays(self):
        return self._get_possibilities(self.root)


if __name__ == '__main__':
    """
    tree = BST()
    tree.insert(3)
    tree.insert(1)
    tree.insert(6)
    tree.insert(2)
    tree.insert(5)
    tree.insert(7)

    ans = tree.build_bst_arrays()

    for p in ans:
        print(", ".join(map(str, p)))
    """

    import numpy as np

    for i in range(100):
        print("Test case", i, "...")
        tree = BST()

        vals = np.random.randint(-9999, 9999, size=5 + i // 10)

        for val in vals:
            try:
                tree.insert(val)
            except ValueError:
                pass

        in_order_vals = tree.in_order_travel()

        ans = tree.build_bst_arrays()

        for config in ans:
            tree_2 = BST()

            for val in config:
                tree_2.insert(val)

            assert in_order_vals == tree_2.in_order_travel()

"""Builds up a Cartesian Tree from an array.

A Cartesian Tree is a structure that satisfy the (minimum)
heap property: every children node key is at least the parent
node key. This means that the minimal element of any subtree
of a Cartesian Subtree is always at the root, and this applies
recursively in every left and right cartesian subtree of any
given root.

The second property of a Cartesian Tree is that the in-order
travel around the tree will hold the original array that
was used to create the Cartesian Tree.
"""
import typing as t

import numpy as np


class _TreeNode:
    def __init__(self,
                 key: t.Union[float, int],
                 son_l: t.Optional["_TreeNode"] = None,
                 son_r: t.Optional["_TreeNode"] = None):
        self.key = key
        self.son_l = son_l
        self.son_r = son_r


class _CartesianBinTree:
    def __init__(self):
        self.num_node = 0
        self._cur_array_ind = -1
        self.root = None  # type: _TreeNode
        self._tree_son_l = None  # type: np.ndarray
        self._tree_son_r = None  # type: np.ndarray
        self._tree_key = None  # type: np.ndarray
        self._node_print_sep = None  # type: np.ndarray

    def _rec_rightmost_path(self, cur_node: t.Optional[_TreeNode],
                            key: t.Union[int, float]) -> bool:
        if cur_node is None:
            return False

        if self._rec_rightmost_path(cur_node.son_r, key=key):
            return True

        if cur_node.key < key:
            new_node = _TreeNode(key=key, son_l=cur_node.son_r, son_r=None)
            cur_node.son_r = new_node
            return True

        return False

    def _rec_travel(self, cur_node: _TreeNode, cur_node_sep: float) -> int:
        """Pre-order tree traversal."""
        if cur_node is None:
            return 0

        cur_node_ind = self._cur_array_ind
        self._tree_key[cur_node_ind] = cur_node.key
        self._node_print_sep[cur_node_ind] = cur_node_sep
        self._cur_array_ind += 1

        height = 0
        _new_sep = 0.5 * cur_node_sep

        if cur_node.son_l:
            self._tree_son_l[cur_node_ind] = self._cur_array_ind
            height = self._rec_travel(cur_node.son_l, _new_sep)

        if cur_node.son_r:
            self._tree_son_r[cur_node_ind] = self._cur_array_ind
            height = max(height, self._rec_travel(cur_node.son_r, _new_sep))

        return 1 + height

    def translate_cart_tree_model(
            self
    ) -> t.Tuple[t.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], int]:
        """Translate the underlying tree model to numpy arrays."""
        tree_height = 0
        self._tree_key = np.zeros(self.num_node)
        self._tree_son_l = np.full(self.num_node, -1, dtype=int)
        self._tree_son_r = np.full(self.num_node, -1, dtype=int)
        self._node_print_sep = np.zeros(self.num_node)

        self._cur_array_ind = 0
        tree_height = self._rec_travel(self.root, 0.5)

        self._node_print_sep = (self._node_print_sep * 2**
                                (1 + tree_height)).astype(int)

        return (self._tree_key, self._tree_son_l, self._tree_son_r,
                self._node_print_sep), tree_height

    def insert(self, key: int) -> None:
        """Insert a new node in the Cartesian Tree."""
        self.num_node += 1

        if self.root is None:
            self.root = _TreeNode(key=key)
            return

        if not self._rec_rightmost_path(cur_node=self.root, key=key):
            new_node = _TreeNode(key=key, son_l=self.root, son_r=None)
            self.root = new_node


class CartesianTree:
    """Build Cartesian Trees."""

    def __init__(self):
        self._tree_son_l = None  # type: np.ndarray
        self._tree_son_r = None  # type: np.ndarray
        self._tree_key = None  # type: np.ndarray
        self._in_order_array = None  # type: np.ndarray
        self.tree_height = -1
        self.num_node = -1
        self._walk_ind = -1
        self._node_print_sep = None  # type; np.ndarray

    def __str__(self):
        """Transform the undelrying tree into a tree string using BFS."""
        print_array = []  # type.List[str]

        cur_depth = 0
        queue = [(0, 0)]

        while queue:
            cur_node_ind, new_depth = queue.pop()

            if new_depth != cur_depth:
                print_array.append("\n")
                cur_depth = new_depth

            _blank_spaces_sep = self._node_print_sep[cur_node_ind] * " "
            print_array.append(_blank_spaces_sep)
            print_array.append("{:.0f}".format(self._tree_key[cur_node_ind]))
            print_array.append(_blank_spaces_sep)

            if self._tree_son_l[cur_node_ind] >= 0:
                queue.insert(0,
                             (self._tree_son_l[cur_node_ind], cur_depth + 1))

            if self._tree_son_r[cur_node_ind] >= 0:
                queue.insert(0,
                             (self._tree_son_r[cur_node_ind], cur_depth + 1))

        return "".join(print_array)

    def _walk_in_order(self, cur_node_ind: int) -> None:
        if self._tree_son_l[cur_node_ind] != -1:
            self._walk_in_order(self._tree_son_l[cur_node_ind])

        self._in_order_array[self._walk_ind] = self._tree_key[cur_node_ind]
        self._walk_ind += 1

        if self._tree_son_r[cur_node_ind] != -1:
            self._walk_in_order(self._tree_son_r[cur_node_ind])

    def walk_in_order(self) -> np.ndarray:
        """Visit all nodes in the Tree in-order (left_son -> cur_node -> right_son)."""
        self._in_order_array = np.zeros(self.num_node)
        self._walk_ind = 0
        self._walk_in_order(0)
        return self._in_order_array

    def fit(self, array: np.ndarray) -> "CartesianTree":
        """Build a Cartesian Tree from a given array."""
        cart_tree_model = _CartesianBinTree()

        for val in array:
            cart_tree_model.insert(val)

        arrays, self.tree_height = cart_tree_model.translate_cart_tree_model()

        self._tree_key, self._tree_son_l, self._tree_son_r, self._node_print_sep = arrays

        self.num_node = self._tree_key.size

        return self


def _test_01():
    np.random.seed(16)
    tree_values = 10 * np.random.random(size=10)
    model = CartesianTree().fit(tree_values)
    print("Original values:", tree_values)
    print("In-order CT walk:", model.walk_in_order())
    print("Tree model:\n", model, sep="")


def _test_02():
    np.random.seed(16)

    num_it = 5000
    for i in np.arange(num_it):
        tree_size = np.random.randint(1, 200)
        tree_values = -200 + 400 * np.random.random(size=tree_size)
        model = CartesianTree().fit(tree_values)
        assert np.allclose(model.walk_in_order(), tree_values)
        print("\r{}%".format(100 * i / num_it), end="")

    print("\rRandomized test done.")


if __name__ == "__main__":
    _test_01()
    _test_02()

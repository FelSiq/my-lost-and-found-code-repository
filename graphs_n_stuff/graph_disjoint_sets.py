"""Data structure for Disjoint Set operations.

This is a simplified version made just for quick future
reference.

Also, quoted from CLRS:
``In an actual implementation of this connected-components algorithm, the repre-
sentations of the graph and the disjoint-set data structure would need to reference
each other. That is, an object representing a vertex would contain a pointer to
the corresponding disjoint-set object, and vice versa. These programming details
depend on the implementation language, and we do not address them further here.``

Note: finding the connected components of an undirected graph
with static edges (i.e., edges that don't change over time) is
faster using the (classical repeated) Depth First Search (DFS)
algorithm.
"""
import typing as t

import numpy as np


class _TreeNode:
    """."""
    def __init__(self, key: int):
        self.key = key
        self.parent = self
        self.rank = 0
        """Rank is used in the ``Union by rank`` heuristic.

        The rank of the node is an upper bound of the height of
        the tree.

        Using this heuristic, we make the root node with the
        smallest rank points to the root with larger rank during
        the Union operation.
        """


class GraphDisjointSets:
    """Disjoint sets for forest graphs. Each tree in the graph is a set.

    Notes
    -----
    The implementation strategy follows the Disjoint-set forests strategy
    presented in CLRS, which is faster than the linked-list representation.
    In this case, every tree represents a tree in the original graph.

    This implementation also uses the two heuristics cited by CLRS, which
    improves the amortized cost of this data structure.
    1. Union by Rank
    2. Path compression
    """
    def __init__(self):
        self._nodes = []  # type: t.List[_TreeNode]

    def __str__(self) -> str:
        _fill_val = max(3, len(str(len(self._nodes))))
        _sep = "+-{0}-+-{0}-+".format(_fill_val * "-")

        str_ = [_sep, "| {:<{fill}} | {:<{fill}} |".format("ID", "Set", fill=_fill_val), _sep]

        for node in self._nodes:
            str_.append("| {:<{fill}} | {:<{fill}} |".format(node.key, self._find_set(node).key, fill=_fill_val))

        str_.append(_sep)
            
        return "\n".join(str_)

    def fit(self, graph: np.ndarray) -> "GraphDisjointSets":
        """."""
        self.num_nodes = graph.shape[0]

        if graph.shape[1] != self.num_nodes:
            raise ValueError("Graph adjacency matrix shape must be "
                             "square (got {}.)".format(graph.shape))

        self._nodes = []

        for index in np.arange(self.num_nodes):
            self.make_set(index)

        for ind_cur_node in np.arange(self.num_nodes):
            for ind_adj_node, weight in enumerate(graph[ind_cur_node, :]):
                if weight > 0 and not self.same_component(ind_cur_node, ind_adj_node):
                    self.union(ind_cur_node, ind_adj_node)

        return self

    def make_set(self, index: int) -> None:
        """Construct a new tree from a single node."""
        self._nodes.append(_TreeNode(key=index))

    def _find_set(self, node: _TreeNode) -> _TreeNode:
        if node != node.parent:
            # Note: using path compression heuristic.
            node.parent = self._find_set(node.parent)

        return node.parent

    def find_set(self, index: int) -> int:
        """Find the element representative of given ``index`` element.

        Notes
        -----
        Using ``path compression`` heuristic. It means that, in every
        call of ``find_set`` method, the parent node pointer of every
        element in the search path is updated to the representative
        element of the ``index`` element set.
        """
        return self._find_set(self._nodes[index]).key

    def _link(self, node_a: _TreeNode, node_b: _TreeNode) -> None:
        if node_a.rank > node_b.rank:
            node_b.parent = node_a

        else:
            node_a.parent = node_b
            if node_a.rank == node_b.rank:
                node_b.rank += 1

    def union(self, index_a: int, index_b: int) -> None:
        """Merge two trees.

        Notes:
        Using the ``Union by rank`` heuristic.
        """
        _node_a = self._nodes[self.find_set(index_a)]
        _node_b = self._nodes[self.find_set(index_b)]
        self._link(_node_a, _node_b)

    def same_component(self, index_a: int, index_b: int) -> bool:
        """Checks wheter two graph nodes are in the same connected component."""
        return self.find_set(index_a) == self.find_set(index_b)


def _test() -> None:
    # np.random.seed(16)
    graph = np.array([
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
    ])
    graph += graph.T

    d_set = GraphDisjointSets().fit(graph)
    print(d_set)


if __name__ == "__main__":
    _test()
